"""Helpers over upstream PEFT LoRA layers.

Factors are handled as ``(B, A)`` pairs whose product is the *effective* dense update:
PEFT applies ``scaling * (B @ A)``, so extraction folds the scaling into ``B`` and
writing divides it out again.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn

from ..config import LoraCfg
from ..training.rng import isolated_rng
from .backbone import Backbone, lora_targets

Factors = tuple[torch.Tensor, torch.Tensor]  # (B, A)


def lora_config(base: nn.Module, backbone: Backbone, cfg: LoraCfg) -> LoraConfig:
    return LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=lora_targets(base, backbone, cfg.target_modules),
        bias="none",
        task_type=TaskType[backbone.task_type],
    )


def build_peft_model(base: nn.Module, config: LoraConfig, seed: int | None) -> nn.Module:
    """Wrap ``base`` with the persistent ``shared`` adapter.

    ``seed`` isolates the initialisation random stream; None draws from the global stream.
    """

    if seed is None:
        return get_peft_model(base, config, adapter_name="shared")
    with isolated_rng(seed, 0, "shared"):
        return get_peft_model(base, config, adapter_name="shared")


def add_adapter(model: nn.Module, name: str, config: LoraConfig, seed: int, index: int, role: str) -> None:
    """Create adapter ``name`` on its own random stream ``(seed, index, role)``."""

    with isolated_rng(seed, index, role):
        model.add_adapter(name, config)


def lora_layers(model: nn.Module, adapter: str) -> Iterator[tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        lora_a, lora_b = getattr(module, "lora_A", None), getattr(module, "lora_B", None)
        if (
            isinstance(lora_a, nn.ModuleDict)
            and isinstance(lora_b, nn.ModuleDict)
            and adapter in lora_a
            and adapter in lora_b
        ):
            yield name, module


def factor_parameters(model: nn.Module, adapter: str) -> list[nn.Parameter]:
    """The LoRA parameters of ``adapter`` (growing-rank history blocks are buffers and not included)."""

    return [
        parameter
        for _, module in lora_layers(model, adapter)
        for factor in (module.lora_A[adapter], module.lora_B[adapter])
        for parameter in factor.parameters()
    ]


def activate(model: nn.Module, active: str | Sequence[str], trainable: str | None) -> list[nn.Parameter]:
    """Activate adapters and make exactly ``trainable`` (if any) require gradients.

    PEFT's list activation makes every active adapter trainable, so the explicit global
    freeze followed by enabling the current adapter is part of the correctness contract.
    """

    names = [active] if isinstance(active, str) else list(active)
    model.base_model.set_adapter(names[0] if len(names) == 1 else names)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if trainable is None:
        return []
    parameters = factor_parameters(model, trainable)
    for parameter in parameters:
        parameter.requires_grad_(True)
    return parameters


class BranchSwitch:
    """Switch between the training branches of one task (``(active, trainable)`` pairs) without rescanning the model.

    ``activate`` freezes every parameter of the model and walks all modules to find the trainable ones, which costs
    tens of milliseconds per call.  During a task the adapters and their parameters do not change, so the switch looks
    them up once.  PEFT's ``set_adapter`` makes every *active* adapter trainable and every other one frozen, and the
    base model stays frozen, so only the active adapters other than the trainable one need freezing again.
    """

    def __init__(self, model: nn.Module, branches: Sequence[tuple[Sequence[str], str | None]]) -> None:
        self.model, self.branches = model, [(list(active), trainable) for active, trainable in branches]
        self.parameters = [activate(model, active, trainable) for active, trainable in self.branches]
        self._factors = {name: factor_parameters(model, name) for active, _ in self.branches for name in active}
        self._selected: int | None = len(self.branches) - 1

    def select(self, branch: int) -> None:
        if branch == self._selected:
            return
        active, trainable = self.branches[branch]
        self.model.base_model.set_adapter(active[0] if len(active) == 1 else active)
        for name in active:
            if name != trainable:
                for parameter in self._factors[name]:
                    parameter.requires_grad_(False)
        self._selected = branch


def freeze(model: nn.Module, adapter: str) -> None:
    for _, module in lora_layers(model, adapter):
        for factor in (module.lora_A[adapter], module.lora_B[adapter]):
            factor.requires_grad_(False)


def effective_factors(model: nn.Module, adapter: str) -> dict[str, Factors]:
    """Detached ``(B * scaling, A)`` per target layer."""

    return {
        name: (
            module.lora_B[adapter].weight.detach().clone() * float(module.scaling[adapter]),
            module.lora_A[adapter].weight.detach().clone(),
        )
        for name, module in lora_layers(model, adapter)
    }


def write_factors(model: nn.Module, adapter: str, factors: Mapping[str, Factors]) -> None:
    """Write effective ``(B, A)`` factors (rank <= adapter rank, zero padded) into ``adapter``."""

    with torch.no_grad():
        for name, module in lora_layers(model, adapter):
            b_new, a_new = factors[name]
            b_weight, a_weight = module.lora_B[adapter].weight, module.lora_A[adapter].weight
            rank = b_new.shape[1]
            b_weight.zero_()
            a_weight.zero_()
            b_weight[:, :rank].copy_((b_new / float(module.scaling[adapter])).to(b_weight))
            a_weight[:rank, :].copy_(a_new.to(a_weight))
