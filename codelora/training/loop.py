"""The training loop shared by every method except MoLE-CIE.

A ``Branch`` is one trainable adapter with its own optimizer, penalty and dropout
stream.  The ``grad_accum`` micro-batches of an optimizer step visit the branches in
order (CoDe-LoRA: shared, then expert), so the active adapters change once per branch
and step; afterwards each branch takes one optimizer step.  Under torchrun each rank
runs its slice of the global micro-batch, weighted by its share of the loss tokens,
and gradients are summed with one flat all-reduce.  Losses stay on the device until
the optimizer step.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from .. import distributed, profiling, tracking
from ..config import TrainCfg
from ..data.tasks import Example
from ..models import adapters
from .optim import build_adamw
from .rng import isolated_rng


@dataclass
class Branch:
    adapters: list[str]  # active during this branch's forward pass
    trainable: str  # the adapter whose LoRA factors are optimised
    penalty: Callable[[], torch.Tensor] | None = None  # regularizer added to the task loss
    rng_role: str | None = None  # isolated dropout stream; None draws from the global RNG

    @property
    def label(self) -> str:
        return self.rng_role or self.trainable


@dataclass
class TrainResult:
    steps: int
    loss: list[list[float]] = field(default_factory=list)  # [branch][micro-batch], global values
    penalty: list[list[float]] = field(default_factory=list)


def _backward(
    model: nn.Module,
    branch: Branch,
    batch: dict[str, torch.Tensor],
    *,
    weight: float,
    grad_accum: int,
    seed: int,
    micro_step: int,
    device: torch.device,
) -> torch.Tensor:
    """Forward/backward one (sharded) micro-batch of the selected branch; returns the weighted local ``(loss, penalty)``.

    Nothing is read back here: the values stay on the device until the optimizer step (``_read_out``).
    """

    model.train()
    values = torch.zeros(2, dtype=torch.float64, device=device)
    if weight > 0.0:
        role = None if branch.rng_role is None else f"optimization:{branch.rng_role}{distributed.rng_role_suffix()}"
        with (
            profiling.phase(f"forward_backward:{branch.label}"),
            isolated_rng(seed, micro_step, role) if role else nullcontext(),
        ):
            loss = model(**batch).loss
            penalty = None if branch.penalty is None else branch.penalty()
            objective = loss if penalty is None else loss + penalty
            (objective * weight / grad_accum).backward()
        values[0] = loss.detach().float()
        if penalty is not None:
            values[1] = penalty.detach().float()
        values *= weight
    return values


def _read_out(
    values: list[list[torch.Tensor]], norms: list[torch.Tensor]
) -> tuple[list[list[tuple[float, float]]], list[float]]:
    """The global ``(loss, penalty)`` of every branch and micro-batch of one optimizer step, and the branch gradient norms.

    One all-reduce and one device-to-host copy per optimizer step.  A non-finite value (loss, penalty or gradient norm)
    means the run has diverged and raises.
    """

    stats = distributed.all_reduce_tensor(torch.stack([v for branch in values for v in branch]))
    host = torch.cat([stats.flatten(), torch.stack(norms).double()]).tolist()
    if not all(map(math.isfinite, host)):
        raise RuntimeError("non-finite loss, penalty or gradient norm: the run has diverged")
    count = len(values) * len(values[0])
    pairs = list(zip(host[0 : 2 * count : 2], host[1 : 2 * count : 2], strict=True))
    per_branch = len(values[0])
    return [pairs[i * per_branch : (i + 1) * per_branch] for i in range(len(values))], host[2 * count :]


def train_task(
    model: nn.Module,
    branches: Sequence[Branch],
    examples: Sequence[Example],
    collator: Callable,
    cfg: TrainCfg,
    *,
    seed: int,
    label: str,
) -> TrainResult:
    """Train ``branches`` on ``examples`` for ``cfg.max_steps`` optimizer updates (or ``cfg.epochs``)."""

    device = next(model.parameters()).device
    switch = adapters.BranchSwitch(model, [(branch.adapters, branch.trainable) for branch in branches])
    parameters = switch.parameters
    optimizers = [build_adamw(params, cfg) for params in parameters]
    tracking.recorder().trainable_parameters = sum(p.numel() for ps in parameters for p in ps)
    loader = DataLoader(
        list(examples),
        batch_size=cfg.micro_batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(seed),
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )
    iterations = cfg.max_steps // len(branches) if cfg.max_steps else None
    steps_per_epoch = math.ceil(len(loader) / cfg.grad_accum)
    epochs = math.ceil(iterations / steps_per_epoch) if iterations else cfg.epochs
    result = TrainResult(0, [[] for _ in branches], [[] for _ in branches])
    micro_step = 0
    for _ in range(epochs):
        group: list[tuple[int, dict[str, torch.Tensor], float]] = []  # (micro step, batch, loss weight) of one update
        for batch_index, raw in enumerate(profiling.timed_iter("data", loader)):
            local, weight = distributed.shard_batch(raw, model)
            group.append((micro_step, {k: v.to(device, non_blocking=True) for k, v in local.items()}, weight))
            micro_step += 1
            if len(group) < cfg.grad_accum and batch_index + 1 < len(loader):
                continue
            values: list[list[torch.Tensor]] = []  # per branch and micro-batch: weighted (loss, penalty), on the device
            for i, branch in enumerate(branches):
                with profiling.phase("activate"):
                    switch.select(i)
                values.append(
                    [
                        _backward(
                            model,
                            branch,
                            batch,
                            weight=weight,
                            grad_accum=cfg.grad_accum,
                            seed=seed,
                            micro_step=step,
                            device=device,
                        )
                        for step, batch, weight in group
                    ]
                )
            accumulated = len(group)
            group.clear()
            norms = []
            for params, optimizer in zip(parameters, optimizers, strict=True):
                with profiling.phase("grad_allreduce"):
                    distributed.all_reduce_gradients(params)
                with profiling.phase("clip_step"):
                    if accumulated < cfg.grad_accum:  # short last batch: rescale to a full accumulation
                        for p in params:
                            if p.grad is not None:
                                p.grad.mul_(cfg.grad_accum / accumulated)
                    norms.append(torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            result.steps += 1
            with profiling.phase("loss_sync"):
                per_branch, norms = _read_out(values, norms)
            for i, pairs in enumerate(per_branch):
                result.loss[i].extend(loss for loss, _ in pairs)
                result.penalty[i].extend(penalty for _, penalty in pairs)
            if distributed.is_main():
                record = [
                    {
                        "adapter": b.trainable,
                        "loss": result.loss[i][-1],
                        "penalty": result.penalty[i][-1],
                        "grad_norm": norms[i],
                    }
                    for i, b in enumerate(branches)
                ]
                tracking.recorder().train_step(label, result.steps, record, optimizers[0].param_groups[0]["lr"])
            if iterations and result.steps >= iterations:
                return result
    return result
