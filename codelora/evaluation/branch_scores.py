"""Teacher-forced log-probabilities of candidate answers (the input of ``evaluation/fusion.py``).

For every example and every candidate answer ``y`` this scores ``log p(y | input)`` (sum over the answer tokens, EOS
included, no length normalisation) under a chosen adapter set.  Cost: one teacher-forced forward pass per
(example, answer, adapter).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .. import distributed
from ..data.tasks import Example
from ..models import adapters
from ..models.backbone import Backbone


def sequence_logprobs(model: nn.Module, batch: dict[str, torch.Tensor], causal: bool) -> list[float]:
    """Sum of token log-probabilities of the supervised (label != -100) positions of each batch row."""

    logits = model(**batch).logits.float()
    labels = batch["labels"]
    if causal:
        logits, labels = logits[:, :-1], labels[:, 1:]
    mask = labels.ne(-100)
    token = F.log_softmax(logits, dim=-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (token * mask).sum(dim=-1).tolist()


def label_logprobs(
    model: nn.Module,
    backbone: Backbone,
    collator: Callable[[Sequence[Example]], dict[str, torch.Tensor]],
    examples: Sequence[Example],
    labels: Sequence[str],
    adapter_names: Sequence[str | tuple[str, ...]],
    batch_size: int,
) -> torch.Tensor:
    """``[example, label]`` log-probabilities (float64) under the adapters ``adapter_names[example]``."""

    device = next(model.parameters()).device
    items = [(i, j) for i in range(len(examples)) for j in range(len(labels))]

    def score(shard: list[tuple[int, int]]) -> list[float]:
        out = [0.0] * len(shard)
        model.eval()
        by_adapter: dict[str | tuple[str, ...], list[int]] = {}
        for position, (i, _) in enumerate(shard):
            by_adapter.setdefault(adapter_names[i], []).append(position)
        with torch.inference_mode():
            for name, positions in by_adapter.items():
                adapters.activate(model, name, None)
                for start in range(0, len(positions), batch_size):
                    chunk = positions[start : start + batch_size]
                    rows = [
                        Example(examples[shard[p][0]].input_text, labels[shard[p][1]], examples[shard[p][0]].task)
                        for p in chunk
                    ]
                    batch = {k: v.to(device) for k, v in collator(rows).items()}
                    for p, value in zip(chunk, sequence_logprobs(model, batch, backbone.causal), strict=True):
                        out[p] = value
        return out

    flat = distributed.sharded_map(score, items)
    return torch.tensor(flat, dtype=torch.float64).reshape(len(examples), len(labels))
