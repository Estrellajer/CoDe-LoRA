"""CLoRA: the fixed-subspace penalty on hand-computed values, frozen bases, finite gradients."""

import torch
from torch import nn

from codelora.config import load_config
from codelora.methods import METHODS
from codelora.methods.clora import SubspaceRegularizer, subspace_penalty


def test_penalty_matches_the_hand_computed_formula():
    a = torch.tensor([[1.0, 0.0, 2.0], [0.0, -1.0, 1.0]], requires_grad=True)  # [r=2, in=3]
    b = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [0.0, -1.0]], requires_grad=True)  # [out=4, r=2]
    p_in = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # [in, k=2]
    p_out = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])  # [out, k=2]
    # A P_in = [[1, 0], [0, -1]] -> 2;  B^T P_out = [[1, 0], [0, 2]] -> 5;  0.25 / 2 * (2 + 5)
    penalty = subspace_penalty(a, b, p_in, p_out, weight=0.25)
    assert penalty.item() == 0.125 * 7
    penalty.backward()
    assert torch.equal(a.grad, 0.25 * (a @ p_in) @ p_in.T)
    assert torch.equal(b.grad, 0.25 * p_out @ (b.T @ p_out).T)


class _Layer(nn.Module):
    def __init__(self, n_in: int, n_out: int, r: int) -> None:
        super().__init__()
        self.lora_A = nn.ModuleDict({"shared": nn.Linear(n_in, r, bias=False)})
        self.lora_B = nn.ModuleDict({"shared": nn.Linear(r, n_out, bias=False)})


def test_bases_are_fixed_orthogonal_and_gradients_are_finite():
    model = nn.Sequential(_Layer(8, 6, 2), _Layer(6, 6, 2))
    regularizer = SubspaceRegularizer(model, k=4, weight=1.0, seed=3)
    assert len(regularizer.layers) == 2
    before = [t.clone() for _, p_in, p_out in regularizer.layers for t in (p_in, p_out)]
    for p in model.parameters():
        nn.init.normal_(p)
    penalty = regularizer()
    penalty.backward()
    assert penalty.item() > 0 and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    after = [t for _, p_in, p_out in regularizer.layers for t in (p_in, p_out)]
    assert all(torch.equal(x, y) and not y.requires_grad for x, y in zip(before, after, strict=True))
    p_in = regularizer.layers[0][1]  # [8, 4]: orthonormal rows
    assert torch.allclose(p_in @ p_in.T, torch.eye(8), atol=1e-5) or torch.allclose(
        p_in.T @ p_in, torch.eye(4), atol=1e-5
    )
    again = SubspaceRegularizer(model, k=4, weight=1.0, seed=3)
    assert torch.equal(again.layers[0][1], regularizer.layers[0][1])  # deterministic in the seed


def test_c_lora_is_registered_with_its_config():
    cfg = load_config(["orders/standard-order1", "protocols/standard/t5-large", "methods/c-lora"])
    assert "c-lora" in METHODS and cfg.method == "c-lora"
    assert (cfg.clora_reg.k, cfg.clora_reg.lambda_reg, cfg.lora.dropout) == (512, 1.0, 0.1)
    assert cfg.train.lr == 1e-3  # T5-large: every method at 1e-3 except sequential LoRA
    small = load_config(["orders/standard-order1", "protocols/standard/qwen3-0.6b", "methods/c-lora"])
    assert small.train.lr == 1e-4  # a shared-adapter method, like lora
