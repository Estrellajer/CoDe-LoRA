"""Batched greedy generation, grouped by the adapter that serves each example."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .. import distributed
from ..data.collate import pretruncate
from ..data.tasks import causal_prompt
from ..models import adapters
from ..models.backbone import Backbone


@dataclass(frozen=True)
class GenerationSettings:
    max_source_length: int
    max_target_length: int
    batch_size: int
    official: bool  # T5: pre-truncation round trip, ``max_length`` decoding with clean-up
    sort_by_length: bool = False


def generate_batch(
    model: nn.Module,
    tokenizer,
    backbone: Backbone,
    texts: Sequence[str],
    settings: GenerationSettings,
    on_encoded: Callable[[Mapping[str, torch.Tensor]], None] | None = None,
) -> list[str]:
    """Greedy-decode one batch with the currently active adapter(s)."""

    model.eval()
    device = next(model.parameters()).device
    causal = backbone.causal
    prompts = [causal_prompt(t) for t in texts] if causal else list(texts)
    official_seq2seq = backbone.seq2seq and settings.official
    if official_seq2seq:
        prompts = pretruncate(tokenizer, prompts, settings.max_source_length)
    padding_side = tokenizer.padding_side
    if causal:  # decoder-only generation aligns every prompt at its final token
        tokenizer.padding_side = "left"
    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=settings.max_source_length
    )
    tokenizer.padding_side = padding_side
    encoded = {k: v.to(device) for k, v in encoded.items()}
    if on_encoded is not None:
        on_encoded(encoded)
    length = {"max_length" if official_seq2seq else "max_new_tokens": settings.max_target_length}
    with torch.inference_mode():
        output = model.generate(**encoded, **length, do_sample=False, num_beams=1, pad_token_id=tokenizer.pad_token_id)
    if causal:
        output = output[:, encoded["input_ids"].shape[1] :]
    if official_seq2seq:
        return tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in output]


def generate_routed(
    model: nn.Module,
    tokenizer,
    backbone: Backbone,
    texts: Sequence[str],
    adapter_names: Sequence[str | tuple[str, ...]],
    settings: GenerationSettings,
) -> list[str]:
    """One output per input; only examples served by the same adapter share a batch.

    Under torchrun each group is strided over ranks and gathered back in the original order.
    """

    groups: dict[str | tuple[str, ...], list[tuple[int, str]]] = {}
    for index, (text, name) in enumerate(zip(texts, adapter_names, strict=True)):
        groups.setdefault(name, []).append((index, text))

    def generate_owned(rank: int, world: int) -> dict[int, str]:
        produced: dict[int, str] = {}
        for name, items in groups.items():
            items = items[rank::world]
            if settings.sort_by_length:  # similar prompt lengths share a batch: less padding
                items = sorted(items, key=lambda item: len(item[1]))
            adapters.activate(model, name, None)
            for start in range(0, len(items), settings.batch_size):
                chunk = items[start : start + settings.batch_size]
                outputs = generate_batch(model, tokenizer, backbone, [t for _, t in chunk], settings)
                produced.update((index, out) for (index, _), out in zip(chunk, outputs, strict=True))
        return produced

    context = distributed.get_context()
    merged: dict[int, str] = {}
    for part in distributed.all_gather_objects(generate_owned(context.rank, context.world_size)):
        merged.update(part)
    outputs = [merged[i] for i in range(len(texts))]
    return outputs
