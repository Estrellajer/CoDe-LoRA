"""Task-boundary consolidation of the Co branch: boundary coefficients, null-space projection, low-rank SVD retraction."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

Factors = tuple[torch.Tensor, torch.Tensor]  # (B, A)


def retract_coefficients(scaling: str, task_index: int) -> tuple[float, float]:
    """``(c_t, s_t)`` of the boundary recurrence ``Co_t = rank_r(c_t Co_{t-1} + s_t dW_t)``.

    ``additive``: ``c = s = 1``.  ``sqrt``: ``c = sqrt((t-1)/t)``, ``s = 1/sqrt(t)``, which keeps the norm of the history
    bounded (``c^2 + s^2 = 1``) and gives the first task the whole branch (``c_1 = 0``).
    """

    if scaling == "additive":
        return 1.0, 1.0
    if scaling == "sqrt":
        t = float(task_index)
        return math.sqrt((t - 1.0) / t), 1.0 / math.sqrt(t)
    raise ValueError(f"colora.scaling must be additive | sqrt; got {scaling!r}")


def null_space_project(history: Factors, update: Factors, rel_tol: float = 1e-6) -> Factors:
    """Remove from ``update = (B, A)`` the part that acts along the column space of ``history = (B_h, A_h)``.

    ``B_orth = B - U U^T B`` with ``U`` the left singular vectors of ``B_h A_h`` above ``rel_tol * s_max`` (QR of ``B_h``,
    SVD of the small core), so ``U^T (B_orth A) = 0``.  An empty history (the zero ``Co_0``) leaves the update unchanged.
    Returns ``(B_orth, A)``; deterministic (the projector ``U U^T`` ignores the sign of each column).
    """

    (b_h, a_h), (b, a) = history, update
    work = torch.float64 if b.dtype == torch.float64 else torch.float32
    q, r = torch.linalg.qr(b_h.to(work), mode="reduced")
    u_core, singular, _ = torch.linalg.svd(r @ a_h.to(work), full_matrices=False)
    if singular[0] == 0:
        return b, a
    u = q @ u_core[:, singular > rel_tol * singular[0]]
    b_work = b.to(work)
    return (b_work - u @ (u.transpose(0, 1) @ b_work)).to(b.dtype), a


def pad_rank(factors: Factors, rank: int) -> Factors:
    b, a = factors
    missing = rank - b.shape[1]
    if missing <= 0:
        return factors
    return torch.cat([b, b.new_zeros(b.shape[0], missing)], dim=1), torch.cat(
        [a, a.new_zeros(missing, a.shape[1])], dim=0
    )


def factorize_low_rank_sum(terms: Sequence[Factors], rank: int) -> Factors:
    """Exact rank-``rank`` truncated SVD of ``sum_i B_i @ A_i``, returned as ``(B, A)`` with ``B = U sqrt(S)``.

    The joined factors are reduced with QR and the only SVD runs on the small core, which is
    mathematically the truncated SVD of the dense sum without ever materialising it.  The
    decomposition runs in float32 (float64 for float64 inputs).
    """

    rows, columns = terms[0][0].shape[0], terms[0][1].shape[1]
    output_dtype = terms[0][0].dtype
    work = torch.float64 if any(t.dtype == torch.float64 for term in terms for t in term) else torch.float32
    keep_rank = min(rank, rows, columns)
    joined_b = torch.cat([b.to(dtype=work) for b, _ in terms], dim=1)
    joined_a = torch.cat([a.to(dtype=work) for _, a in terms], dim=0)
    q_b, r_b = torch.linalg.qr(joined_b, mode="reduced")
    q_a, r_a = torch.linalg.qr(joined_a.transpose(0, 1), mode="reduced")
    u_core, singular, vh_core = torch.linalg.svd(r_b @ r_a.transpose(0, 1), full_matrices=False)
    keep = min(keep_rank, singular.numel())
    root = singular[:keep].sqrt()
    b = (q_b @ u_core[:, :keep] * root.unsqueeze(0)).to(dtype=output_dtype)
    a = (root.unsqueeze(1) * (vh_core[:keep, :] @ q_a.transpose(0, 1))).to(dtype=output_dtype)
    if keep < keep_rank:
        b = torch.cat((b, b.new_zeros((rows, keep_rank - keep))), dim=1)
        a = torch.cat((a, a.new_zeros((keep_rank - keep, columns))), dim=0)
    return b, a


def svd_truncate(dense: torch.Tensor, rank: int) -> Factors:
    """Optimal rank-``rank`` factorisation of a dense matrix (``B = U sqrt(S)``), zero padded to ``rank``."""

    work = dense.to(dtype=torch.float64 if dense.dtype == torch.float64 else torch.float32)
    u, s, vh = torch.linalg.svd(work, full_matrices=False)
    keep = min(rank, int((s > 0).sum().item()))
    root = s[:keep].sqrt()
    b = (u[:, :keep] * root.unsqueeze(0)).to(dtype=dense.dtype)
    a = (root.unsqueeze(1) * vh[:keep, :]).to(dtype=dense.dtype)
    return pad_rank((b, a), rank)
