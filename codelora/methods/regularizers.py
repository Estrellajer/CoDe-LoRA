"""O-LoRA and N-LoRA penalties on the block trained at the current task."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch

Factors = tuple[torch.Tensor, torch.Tensor]  # (B, A)


def factor_product_l1(b: torch.Tensor, a: torch.Tensor, weight: float, reduction: str) -> torch.Tensor:
    """Upstream N-LoRA ``|A_new @ B_new|_1``, extended to rectangular layers.

    Upstream computes ``torch.norm(torch.mm(A, B), p=1)`` with ``A: [r, in]`` and ``B: [out, r]``;
    shape-wise ``mm`` only exists for ``in == out`` (T5 and Llama q/v).  For ``in != out`` the
    contraction runs over the feature axis of the longer factor in blocks the length of the
    shorter one (a trailing partial block uses the overlapping coordinates only).  Square layers
    are exactly the upstream expression; divisible dimensions (GQA ``q_proj`` in Qwen3) reduce to
    whole blocks.

    ``b`` and ``a`` may carry a leading batch dimension (layers of equal shape stacked); the result is then the sum of the
    penalties of the stacked layers.
    """

    in_dim, out_dim = a.shape[-1], b.shape[-2]
    if out_dim >= in_dim:
        products = [a[..., : block.shape[-2]] @ block for block in b.split(in_dim, dim=-2)]
    else:
        products = [block @ b[..., : block.shape[-1], :] for block in a.split(out_dim, dim=-1)]
    dense = torch.cat(products, dim=-2).abs()
    value = dense.sum() if reduction == "sum" else dense.mean(dim=(-2, -1)).sum()
    return float(weight) * value


def nlora_penalty(factors: Iterable[Factors], weight: float, reduction: str) -> torch.Tensor:
    """Sum of ``factor_product_l1`` over layers; layers of equal shape are evaluated together in one batched product."""

    groups: dict[tuple[torch.Size, torch.Size], list[Factors]] = {}
    for b, a in factors:
        groups.setdefault((b.shape, a.shape), []).append((b, a))
    values = [
        factor_product_l1(torch.stack([b for b, _ in group]), torch.stack([a for _, a in group]), weight, reduction)
        for group in groups.values()
    ]
    return torch.stack(values).sum()


def olora_penalty(
    current: Mapping[str, Factors], history_a: Mapping[str, torch.Tensor], lambda_orth: float, lambda_l2: float
) -> torch.Tensor:
    """``lambda_orth * |A_hist A_cur^T|_1 + lambda_l2 * (|A_cur|_2 + |B_cur|_2)`` summed over layers."""

    total = None
    for name, (b, a) in current.items():
        orth = history_a[name].detach().to(device=a.device, dtype=a.dtype).matmul(a.transpose(0, 1)).abs().sum()
        l2 = torch.linalg.vector_norm(a) + torch.linalg.vector_norm(b)
        value = float(lambda_orth) * orth + float(lambda_l2) * l2
        total = value if total is None else total + value
    return total


def dense_update_l1(b: torch.Tensor, a: torch.Tensor, scaling: float, weight: float, reduction: str) -> torch.Tensor:
    """Scale-aware N-LoRA penalty on the effective update: ``weight * |scaling * B @ A|_1`` (sum or mean)."""

    dense = (b * float(scaling)) @ a
    value = dense.abs().sum() if reduction == "sum" else dense.abs().mean()
    return float(weight) * value


def olora_multi_penalty(
    current: Mapping[str, Factors],
    histories: Iterable[Mapping[str, torch.Tensor]],
    lambda_orth: float,
    lambda_l2: float,
) -> torch.Tensor:
    """O-LoRA over one PEFT adapter per task: orthogonality of the new A to every earlier adapter's A, per layer."""

    histories = list(histories)
    total = None
    for name, (b, a) in current.items():
        orth = a.sum() * 0.0
        for history in histories:
            if name in history:
                orth = (
                    orth
                    + history[name].detach().to(device=a.device, dtype=a.dtype).matmul(a.transpose(0, 1)).abs().sum()
                )
        l2 = torch.linalg.vector_norm(a) + torch.linalg.vector_norm(b)
        value = float(lambda_orth) * orth + float(lambda_l2) * l2
        total = value if total is None else total + value
    return total
