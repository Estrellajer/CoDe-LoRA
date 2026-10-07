"""Single-node data parallelism: replicated model, sharded work.

Every rank holds identical adapter weights and runs the same control flow, so
consolidation, routing statistics and bookkeeping need no rank-0-only branches.
Only the expensive loops are split:

* training: each micro-batch is split across ranks and adapter gradients are
  summed with one flat all-reduce before clipping;
* generation and embedding: items are strided across ranks and gathered back in
  the original order.

The model is not wrapped in ``DistributedDataParallel``: the trainable branch
changes inside a step (shared, then expert) and the active adapter set changes
between forwards, which DDP's reducer does not support.  Adapters are a few MB,
so a flat all-reduce is equivalent and simpler.

With ``WORLD_SIZE`` unset (or 1) every helper is an identity.
"""

from __future__ import annotations

import datetime
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistContext:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


_CONTEXT = DistContext()


def get_context() -> DistContext:
    return _CONTEXT


def is_main() -> bool:
    return _CONTEXT.is_main


def init_from_env(timeout_minutes: int = 30) -> DistContext:
    """Join the process group described by torchrun environment variables."""

    global _CONTEXT
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return _CONTEXT
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend, timeout=datetime.timedelta(minutes=timeout_minutes))
    _CONTEXT = DistContext(rank, world_size, local_rank)
    return _CONTEXT


def shutdown() -> None:
    global _CONTEXT
    if _CONTEXT.enabled:
        dist.destroy_process_group()
    _CONTEXT = DistContext()


def rng_role_suffix() -> str:
    """Make per-rank dropout streams differ (empty when not distributed)."""

    return f"@rank{_CONTEXT.rank}" if _CONTEXT.enabled else ""


def barrier() -> None:
    if _CONTEXT.enabled:
        dist.barrier()


# ---------------------------------------------------------------- training


def _loss_tokens(labels: torch.Tensor, causal: bool) -> int:
    """Number of target positions HF's mean cross-entropy normalises by."""

    if labels.shape[0] == 0:
        return 0
    return int((labels[:, 1:] if causal else labels).ne(-100).sum())


def shard_batch(batch: Mapping[str, torch.Tensor], model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], float]:
    """This rank's contiguous slice of a global micro-batch and its loss weight.

    The weight is the slice's share of the global loss tokens, so that
    ``sum_r weight_r * mean_loss_r`` equals the token-mean loss of the whole
    micro-batch, exactly what a single process computes.  A slice may be empty
    when the last batch of an epoch has fewer samples than ranks.
    """

    if not _CONTEXT.enabled:
        return dict(batch), 1.0
    labels = batch["labels"]
    causal = not model.config.is_encoder_decoder
    count, rank, world = labels.shape[0], _CONTEXT.rank, _CONTEXT.world_size
    low, high = count * rank // world, count * (rank + 1) // world
    local = {key: value[low:high] for key, value in batch.items()}
    total = _loss_tokens(labels, causal)
    if total == 0:
        raise ValueError("micro-batch has no supervised target tokens")
    return local, _loss_tokens(local["labels"], causal) / total


def all_reduce_tensor(values: torch.Tensor) -> torch.Tensor:
    """Sum a tensor over ranks (returned as is when not distributed)."""

    if _CONTEXT.enabled:
        dist.all_reduce(values)
    return values


def all_reduce_sum(*values: float, device: torch.device) -> tuple[float, ...]:
    """Sum python scalars over ranks in float64."""

    if not _CONTEXT.enabled:
        return values
    return tuple(all_reduce_tensor(torch.tensor(values, dtype=torch.float64, device=device)).tolist())


def all_reduce_gradients(parameters: Sequence[torch.nn.Parameter]) -> None:
    """Sum gradients over ranks in one flat buffer (missing grads count as 0)."""

    if not _CONTEXT.enabled:
        return
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in parameters]
    flat = torch.cat([grad.reshape(-1) for grad in grads])
    dist.all_reduce(flat)
    offset = 0
    for parameter in parameters:
        parameter.grad = flat[offset : offset + parameter.numel()].view_as(parameter)
        offset += parameter.numel()


def broadcast_parameters(tensors: Sequence[torch.Tensor], src: int = 0) -> None:
    """Overwrite ``tensors`` on every rank with rank ``src``'s values."""

    if not _CONTEXT.enabled or not tensors:
        return
    with torch.no_grad():
        flat = torch.cat([t.reshape(-1) for t in tensors])
        dist.broadcast(flat, src=src)
        offset = 0
        for tensor in tensors:
            tensor.copy_(flat[offset : offset + tensor.numel()].view_as(tensor))
            offset += tensor.numel()


def sync_adapters(model: torch.nn.Module) -> None:
    """Re-synchronise every LoRA weight and buffer (frozen ones included) from rank 0.

    Called after steps that mutate weights without a collective (adapter
    creation, SVD consolidation) so a nondeterministic kernel can never leave
    ranks with different weights.  Growing-rank history blocks are buffers.
    """

    if _CONTEXT.enabled:
        named = [*model.named_parameters(), *model.named_buffers()]
        broadcast_parameters([t for name, t in named if "lora" in name])


def sync_module_parameters(modules: Sequence[torch.nn.Module], skip_prefix: str = "base_layer.") -> None:
    """Same for wrapper modules that hold their own weights next to a frozen base layer (MoLE-CIE)."""

    if _CONTEXT.enabled:
        broadcast_parameters(
            [p for m in modules for name, p in m.named_parameters() if not name.startswith(skip_prefix)]
        )


# -------------------------------------------------------------- evaluation


def all_gather_objects(local: Any) -> list[Any]:
    if not _CONTEXT.enabled:
        return [local]
    gathered: list[Any] = [None] * _CONTEXT.world_size
    dist.all_gather_object(gathered, local)
    return gathered


def main_computes(fn: Callable[[], Any]) -> Any:
    """Run ``fn`` on rank 0 only and hand its (picklable) result to every rank."""

    if not _CONTEXT.enabled:
        return fn()
    box = [fn() if _CONTEXT.is_main else None]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def sharded_map(fn: Callable[[list[Any]], list[Any]], items: Sequence[Any]) -> list[Any]:
    """``fn(items)`` computed cooperatively: rank r handles ``items[r::world]``.

    ``fn`` maps a list to a same-length list and must accept an empty list.
    The result is in the original item order on every rank.
    """

    items = list(items)
    if not _CONTEXT.enabled:
        return fn(items)
    world = _CONTEXT.world_size
    parts = all_gather_objects(fn(items[_CONTEXT.rank :: world]))
    merged: list[Any] = [None] * len(items)
    for rank, part in enumerate(parts):
        merged[rank::world] = part
    return merged
