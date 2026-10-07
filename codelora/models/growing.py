"""Growing-rank LoRA state (official O-LoRA / N-LoRA layout) on top of PEFT layers.

Each target layer keeps one adapter whose ``lora_A`` / ``lora_B`` are replaced by
block modules: a frozen history (registered as a buffer, so it never receives
gradients) and one trainable ``current`` block.  The forward pass is the
concatenation of both, so the effective update is ``scaling * (B_hist A_hist + B_cur A_cur)``
with PEFT's scaling unchanged as the rank grows.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import nn

from .adapters import lora_layers


class GrowingA(nn.Module):
    """Project into the frozen history rank block and the trainable current one."""

    def __init__(self, current: nn.Linear) -> None:
        super().__init__()
        self.register_buffer("history_weight", current.weight.detach().new_empty((0, current.in_features)))
        self.current: nn.Linear | None = current
        self.in_features = current.in_features

    @property
    def history_rank(self) -> int:
        return int(self.history_weight.shape[0])

    @property
    def weight(self) -> torch.Tensor:
        return torch.cat([self.history_weight, *([] if self.current is None else [self.current.weight])], dim=0)

    def commit(self) -> None:
        self.history_weight = torch.cat((self.history_weight, self.current.weight.detach()), dim=0).clone()
        self.current = None

    def restore(self, history: torch.Tensor) -> None:
        self.history_weight = history.to(self.history_weight).detach().clone()
        self.current = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = []
        if self.history_rank:
            outputs.append(F.linear(inputs, self.history_weight))
        if self.current is not None:
            outputs.append(self.current(inputs))
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)


class GrowingB(nn.Module):
    """Map the concatenated rank blocks back to the output width; history stays gradient-free."""

    def __init__(self, current: nn.Linear) -> None:
        super().__init__()
        self.register_buffer("history_weight", current.weight.detach().new_empty((current.out_features, 0)))
        self.current: nn.Linear | None = current
        self.out_features = current.out_features

    @property
    def history_rank(self) -> int:
        return int(self.history_weight.shape[1])

    @property
    def weight(self) -> torch.Tensor:
        return torch.cat([self.history_weight, *([] if self.current is None else [self.current.weight])], dim=1)

    def commit(self) -> None:
        self.history_weight = torch.cat((self.history_weight, self.current.weight.detach()), dim=1).clone()
        self.current = None

    def restore(self, history: torch.Tensor) -> None:
        self.history_weight = history.to(self.history_weight).detach().clone()
        self.current = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = inputs.new_zeros((*inputs.shape[:-1], self.out_features))
        if self.history_rank:
            result = result + F.linear(inputs[..., : self.history_rank], self.history_weight)
        if self.current is not None:
            result = result + self.current(inputs[..., self.history_rank :])
        return result


def _pairs(model: nn.Module, adapter: str) -> Iterator[tuple[str, GrowingA, GrowingB]]:
    for name, module in lora_layers(model, adapter):
        yield name, module.lora_A[adapter], module.lora_B[adapter]


def install_growing_rank(model: nn.Module, adapter: str) -> None:
    """Convert an initialised PEFT adapter into growing-rank blocks (its weights become block 0)."""

    for _, module in lora_layers(model, adapter):
        module.lora_A[adapter] = GrowingA(module.lora_A[adapter])
        module.lora_B[adapter] = GrowingB(module.lora_B[adapter])


def begin_task(model: nn.Module, adapter: str, rank: int) -> None:
    """Open a fresh current block of ``rank`` (N-LoRA re-initialisation: kaiming A, zero B).

    The official fork builds the historical A/B ``Linear`` modules before re-initialising the
    new block; the two throw-away constructions below consume the same random numbers so the
    initial weights match the upstream process-seed protocol.
    """

    for _, lora_a, lora_b in _pairs(model, adapter):
        history_rank = lora_a.history_rank
        lora_a.current = nn.Linear(lora_a.in_features, rank, bias=False)
        lora_b.current = nn.Linear(rank, lora_b.out_features, bias=False)
        nn.Linear(lora_a.in_features, history_rank, bias=False)
        nn.Linear(history_rank, lora_b.out_features, bias=False)
        nn.init.kaiming_uniform_(lora_a.current.weight, a=math.sqrt(5))
        nn.init.zeros_(lora_b.current.weight)
        lora_a.current.to(device=lora_a.history_weight.device, dtype=lora_a.history_weight.dtype)
        lora_b.current.to(device=lora_b.history_weight.device, dtype=lora_b.history_weight.dtype)


def current_factors(model: nn.Module, adapter: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Live ``(B, A)`` weights of the trainable block per layer."""

    return {name: (b.current.weight, a.current.weight) for name, a, b in _pairs(model, adapter)}


def history_factors(model: nn.Module, adapter: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {name: (b.history_weight, a.history_weight) for name, a, b in _pairs(model, adapter)}


def commit_task(model: nn.Module, adapter: str) -> int:
    """Freeze the current block into the history and return the new cumulative rank."""

    rank = 0
    for _, lora_a, lora_b in _pairs(model, adapter):
        lora_a.commit()
        lora_b.commit()
        rank = lora_a.history_rank
    return rank


def replace_history(model: nn.Module, adapter: str, factors: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
    """Replace the whole state by committed ``(B, A)`` factors and drop the current block."""

    for name, lora_a, lora_b in _pairs(model, adapter):
        b_new, a_new = factors[name]
        lora_a.restore(a_new)
        lora_b.restore(b_new)


def export_state(model: nn.Module, adapter: str) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {"A": a.history_weight.detach().cpu(), "B": b.history_weight.detach().cpu()}
        for name, a, b in _pairs(model, adapter)
    }
