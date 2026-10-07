"""CLoRA (Lu et al., ACL 2025): sequential LoRA with a fixed-subspace regulariser.

One shared adapter is trained on every task in turn, exactly as ``lora``.  Per LoRA layer the task loss gets

    lambda / 2 * (|| A P_in ||_F^2 + || B^T P_out ||_F^2),

with ``A: [r, in]`` and ``B: [out, r]`` the PEFT factors and ``P_in: [in, k]``, ``P_out: [out, k]`` fixed random orthogonal
bases (official ``CLoraWrapper`` of sutakori/CLoRA).  The bases are drawn once at setup and never trained; they are only
needed for training and are not stored in the checkpoint.  The wrapper of the official code is replaced by an explicit
penalty added to the loss, which is mathematically the same.
"""

from __future__ import annotations

import torch
from torch import nn

from .. import distributed
from ..training.loop import Branch
from .base import register
from .lora import SequentialLoRA


def subspace_penalty(
    a: torch.Tensor, b: torch.Tensor, input_basis: torch.Tensor, output_basis: torch.Tensor, weight: float
) -> torch.Tensor:
    """``weight / 2 * (|a @ input_basis|_F^2 + |b.T @ output_basis|_F^2)`` of one layer; the bases are constants."""

    input_basis, output_basis = input_basis.to(a.dtype), output_basis.to(b.dtype)
    overlap_in, overlap_out = a @ input_basis, b.transpose(0, 1) @ output_basis
    return 0.5 * float(weight) * (overlap_in.square().sum() + overlap_out.square().sum())


def _orthogonal(rows: int, columns: int, device: torch.device) -> torch.Tensor:
    """``nn.init.orthogonal_`` in float32 (QR is not available in bf16 everywhere); cast by the caller."""

    basis = torch.empty(rows, columns, device=device, dtype=torch.float32)
    nn.init.orthogonal_(basis)
    return basis


class SubspaceRegularizer:
    """Fixed bases of every ``shared`` LoRA layer of a PEFT model and the penalty over them."""

    def __init__(self, model: nn.Module, k: int, weight: float, seed: int, adapter: str = "shared") -> None:
        self.adapter, self.k, self.weight = adapter, k, float(weight)
        self.layers: list[tuple[nn.Module, torch.Tensor, torch.Tensor]] = []
        generator_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        for _, module in model.named_modules():
            lora_a, lora_b = getattr(module, "lora_A", None), getattr(module, "lora_B", None)
            if lora_a is None or lora_b is None or adapter not in lora_a:
                continue
            a, b = lora_a[adapter].weight, lora_b[adapter].weight
            self.layers.append(
                (
                    module,
                    _orthogonal(a.shape[1], k, a.device).to(a.dtype),
                    _orthogonal(b.shape[0], k, b.device).to(b.dtype),
                )
            )
        torch.random.set_rng_state(generator_state)
        if not self.layers:
            raise KeyError(f"adapter {adapter!r} has no LoRA layers")
        distributed.broadcast_parameters(
            [t for _, p_in, p_out in self.layers for t in (p_in, p_out)]
        )  # identical on every rank

    def __call__(self) -> torch.Tensor:
        terms = [
            subspace_penalty(m.lora_A[self.adapter].weight, m.lora_B[self.adapter].weight, p_in, p_out, self.weight)
            for m, p_in, p_out in self.layers
        ]
        return torch.stack(terms).sum()


@register("c-lora")
class CLoRA(SequentialLoRA):
    def setup(self) -> None:
        super().setup()
        c = self.cfg.clora_reg
        self.regularizer = SubspaceRegularizer(self.model, c.k, c.lambda_reg, self.cfg.seed)

    def branch(self) -> Branch:
        return Branch(["shared"], "shared", penalty=self.regularizer, rng_role="shared")
