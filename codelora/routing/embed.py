"""Task-routing features: masked-mean last hidden state of the frozen base model."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from contextlib import nullcontext

import torch
from torch import nn

from .. import distributed
from ..models.backbone import Backbone

_CACHE: dict[str, torch.Tensor] = {}


def base_embeddings(
    model: nn.Module,
    tokenizer,
    backbone: Backbone,
    texts: Sequence[str],
    *,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    """``[len(texts), hidden]`` fp32 embeddings with every adapter disabled.

    Adapters are disabled and the base weights are frozen for the whole run, so identical
    inputs always produce identical embeddings and repeated evaluations hit the cache.
    """

    key = hashlib.sha256("\0".join((str(id(model)), str(batch_size), str(max_length), *texts)).encode()).hexdigest()
    if key in _CACHE:
        return _CACHE[key]
    model.eval()
    device = next(model.parameters()).device

    def embed(shard: list[str]) -> list[torch.Tensor]:
        rows: list[torch.Tensor] = []
        disable = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
        with disable, torch.inference_mode():
            base = model.get_base_model()
            for start in range(0, len(shard), batch_size):
                encoded = tokenizer(
                    shard[start : start + batch_size],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                if backbone.seq2seq:
                    hidden = base.get_encoder()(**encoded, return_dict=True).last_hidden_state
                else:
                    output = base(
                        **encoded, output_hidden_states=True, use_cache=False, return_dict=True, logits_to_keep=1
                    )
                    hidden = output.hidden_states[-1]  # hidden states only: no (batch x length x vocab) logits
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                # clone: a row view would pickle its whole batch storage when gathered
                rows.extend(row.clone() for row in pooled.float().cpu().unbind(0))
        return rows

    result = torch.stack(distributed.sharded_map(embed, texts))
    _CACHE[key] = result
    return result
