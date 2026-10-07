import pytest
import torch

from codelora.methods.consolidation import (
    factorize_low_rank_sum,
    null_space_project,
    pad_rank,
    retract_coefficients,
    svd_truncate,
)


def truncated_svd(dense: torch.Tensor, rank: int) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(dense.double(), full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vh[:rank]


@pytest.mark.parametrize("rank", [1, 4, 8, 16])
def test_low_rank_sum_equals_truncated_svd_of_the_dense_sum(rank):
    torch.manual_seed(0)
    terms = [(torch.randn(24, 8), torch.randn(8, 40)) for _ in range(2)]
    b, a = factorize_low_rank_sum(terms, rank)
    dense = sum(tb @ ta for tb, ta in terms)
    assert b.shape == (24, rank) and a.shape == (rank, 40)
    assert torch.allclose((b @ a).double(), truncated_svd(dense, rank), atol=2e-4)


def test_factors_are_balanced():
    """``B = U sqrt(S)`` and ``A = sqrt(S) V^T``: equal column / row norms."""

    torch.manual_seed(1)
    b, a = factorize_low_rank_sum([(torch.randn(16, 4), torch.randn(4, 16))], 4)
    assert torch.allclose(b.norm(dim=0), a.norm(dim=1), atol=1e-4)


def test_rank_deficient_sum_is_zero_padded():
    x = torch.randn(10, 1)
    b, a = factorize_low_rank_sum([(x, x.T), (x, -x.T)], 3)  # sums to zero
    assert b.shape == (10, 3) and torch.allclose(b @ a, torch.zeros(10, 10), atol=1e-6)


def test_output_keeps_the_input_dtype():
    terms = [(torch.randn(8, 2, dtype=torch.bfloat16), torch.randn(2, 8, dtype=torch.bfloat16))]
    b, a = factorize_low_rank_sum(terms, 2)
    assert b.dtype == a.dtype == torch.bfloat16


def test_svd_truncate_is_the_best_low_rank_approximation():
    torch.manual_seed(2)
    dense = torch.randn(12, 20)
    b, a = svd_truncate(dense, 5)
    assert torch.allclose((b @ a).double(), truncated_svd(dense, 5), atol=1e-4)
    b, a = svd_truncate(torch.zeros(6, 6), 3)  # exactly zero matrix: zero padded
    assert b.shape == (6, 3) and not (b @ a).any()


def test_pad_rank():
    b, a = pad_rank((torch.ones(4, 2), torch.ones(2, 5)), 4)
    assert b.shape == (4, 4) and a.shape == (4, 5) and not b[:, 2:].any() and not a[2:].any()


def test_boundary_coefficients():
    assert retract_coefficients("additive", 5) == (1.0, 1.0)
    assert retract_coefficients("sqrt", 1) == (0.0, 1.0)  # the first task takes the whole branch
    c, s = retract_coefficients("sqrt", 4)
    assert (c, s) == pytest.approx((0.75**0.5, 0.5)) and c**2 + s**2 == pytest.approx(1.0)
    with pytest.raises(ValueError, match=r"colora\.scaling"):
        retract_coefficients("linear", 2)


def test_null_space_projection_removes_the_component_along_the_history():
    torch.manual_seed(3)
    history = (torch.randn(20, 3), torch.randn(3, 30))
    update = (torch.randn(20, 4), torch.randn(4, 30))
    b, a = null_space_project(history, update)
    u = torch.linalg.svd(history[0] @ history[1], full_matrices=False)[0][:, :3]
    assert a is update[1] and torch.allclose(
        u.T @ b, torch.zeros(3, 4), atol=1e-5
    )  # nothing left along the column space
    assert torch.allclose(update[0] - b, u @ (u.T @ update[0]), atol=1e-5)  # exactly the history part was removed
    assert torch.allclose(null_space_project(history, (b, a))[0], b, atol=1e-5)  # idempotent
    inside = (history[0] @ torch.randn(3, 4), update[1])  # an update inside the history's column space vanishes
    assert null_space_project(history, inside)[0].abs().max() < 1e-4


def test_null_space_projection_of_an_empty_history_is_the_identity():
    update = (torch.randn(8, 2), torch.randn(2, 6))
    b, a = null_space_project((torch.zeros(8, 4), torch.zeros(4, 6)), update)
    assert torch.equal(b, update[0]) and torch.equal(a, update[1])
