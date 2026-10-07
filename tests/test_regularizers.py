import torch

from codelora.methods.regularizers import factor_product_l1, nlora_penalty, olora_penalty


def test_square_layer_is_the_upstream_expression():
    torch.manual_seed(0)
    a, b = torch.randn(4, 16), torch.randn(16, 4)  # A: [r, in], B: [out, r]
    expected = torch.norm(torch.mm(a, b), p=1)  # upstream N-LoRA: torch.norm(torch.mm(A, B), p=1)
    assert torch.allclose(factor_product_l1(b, a, 0.4, "sum"), 0.4 * expected)
    assert torch.allclose(factor_product_l1(b, a, 1.0, "mean"), (a @ b).abs().mean())


def test_rectangular_layers_contract_in_blocks():
    torch.manual_seed(1)
    r = 3
    # out > in (divisible): two A @ B_block products
    a, b = torch.randn(r, 4), torch.randn(8, r)
    blocks = [a @ b[:4], a @ b[4:]]
    assert torch.allclose(factor_product_l1(b, a, 1.0, "sum"), torch.cat(blocks).abs().sum())
    # in > out: blocks of A of width out
    a, b = torch.randn(r, 8), torch.randn(4, r)
    blocks = [a[:, :4] @ b, a[:, 4:] @ b]
    assert torch.allclose(factor_product_l1(b, a, 1.0, "sum"), torch.cat(blocks).abs().sum())
    # trailing partial block uses the overlapping coordinates only
    a, b = torch.randn(r, 4), torch.randn(6, r)
    expected = torch.cat([a @ b[:4], a[:, :2] @ b[4:]]).abs().sum()
    assert torch.allclose(factor_product_l1(b, a, 1.0, "sum"), expected)


def test_batched_layers_equal_the_sum_of_single_layers():
    torch.manual_seed(3)
    for out_dim, in_dim in ((8, 8), (12, 4), (4, 12), (6, 4)):  # square, divisible both ways, partial block
        layers = [
            (torch.randn(out_dim, 3, requires_grad=True), torch.randn(3, in_dim, requires_grad=True)) for _ in range(4)
        ]
        layers.append((torch.randn(5, 3, requires_grad=True), torch.randn(3, 5, requires_grad=True)))  # another shape
        for reduction in ("sum", "mean"):
            expected = sum(factor_product_l1(b, a, 0.4, reduction) for b, a in layers)
            value = nlora_penalty(layers, 0.4, reduction)
            assert torch.allclose(value, expected, rtol=1e-5)
            grads = torch.autograd.grad(value, [t for pair in layers for t in pair])
            reference = torch.autograd.grad(expected, [t for pair in layers for t in pair])
            assert all(torch.allclose(g, r, rtol=1e-4, atol=1e-6) for g, r in zip(grads, reference, strict=True))


def test_penalty_gradients_reach_the_current_block_only():
    a, b = torch.randn(2, 8, requires_grad=True), torch.randn(8, 2, requires_grad=True)
    nlora_penalty([(b, a)], 0.4, "sum").backward()
    assert a.grad is not None and b.grad is not None and a.grad.abs().sum() > 0


def test_olora_penalty_matches_definition():
    torch.manual_seed(2)
    history = {"l0": torch.randn(3, 8), "l1": torch.randn(3, 8)}
    current = {n: (torch.randn(8, 2, requires_grad=True), torch.randn(2, 8, requires_grad=True)) for n in history}
    value = olora_penalty(current, history, lambda_orth=0.5, lambda_l2=0.01)
    expected = sum(0.5 * (history[n] @ a.T).abs().sum() + 0.01 * (a.norm() + b.norm()) for n, (b, a) in current.items())
    assert torch.allclose(value, expected)
    value.backward()
    assert all(a.grad is not None for _, a in current.values())


def test_olora_penalty_is_zero_orthogonality_without_history():
    current = {"l": (torch.randn(8, 2), torch.randn(2, 8))}
    value = olora_penalty(current, {"l": torch.zeros(0, 8)}, lambda_orth=0.5, lambda_l2=0.0)
    assert value == 0
