"""CoDe-LoRA: one-pass training, the fold into Co, the replayed expert context and threshold routing."""

from __future__ import annotations

import pytest
import torch
import yaml

from codelora.config import load_config
from codelora.data.tasks import Example
from codelora.methods import build_method
from codelora.methods.codelora import CoDeLoRA
from codelora.methods.consolidation import retract_coefficients
from codelora.models import adapters
from codelora.runner import Benchmark, build_context
from codelora.training.rng import seed_everything


def make_method(lab, **overrides) -> CoDeLoRA:
    """CoDe-LoRA on the tiny Qwen and a three-task benchmark (alpha, beta, gamma), LDA routing, 4 updates per task."""

    fragment = lab["root"] / "codelora.yaml"
    fragment.write_text(
        yaml.safe_dump(
            {
                "defaults": ["tiny/smoke", "methods/code-lora"],
                "output_dir": str(lab["root"] / "runs"),
                "backbone": {"name": "qwen3-0.6b", "path": str(lab["qwen"])},
                "lora": {"target_modules": ["q_proj", "v_proj"], "r": 4, "alpha": 16},
                "data": {"root": str(lab["data"]), "max_target_length": 3},
                "tasks": [
                    {"name": "alpha", "path": "TC/alpha"},
                    {"name": "beta", "path": "SC/beta"},
                    {"name": "gamma", "path": "TC/gamma"},
                ],
                "router": {"kind": "lda", "threshold": 0.0, "embedding_input": "raw_source", "lda_samples": 30},
                "train": {"max_steps": 4},
                "eval": {"batch_size": 4},
            }
        )
    )
    cfg = load_config([fragment], [f"{k}={v}" for k, v in overrides.items()])
    seed_everything(cfg.seed)
    method = build_method(build_context(cfg))
    method.setup()
    return method


def learn(method, count: int) -> list[dict]:
    bench = method.ctx.benchmark = Benchmark(method.cfg)
    return [method.learn_task(i, method.cfg.tasks[i - 1], bench.train(i)) for i in range(1, count + 1)]


def dense(factors):
    return {name: b.double() @ a.double() for name, (b, a) in factors.items()}


def truncated_svd(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vh[:rank]


def test_one_optimizer_update_per_step_and_no_penalty(lab):
    method = make_method(lab)
    assert method.branches == 1
    stats = learn(method, 2)
    assert [s["steps"] for s in stats] == [4, 4]  # max_steps counts optimizer updates
    assert "penalty" not in stats[0] and "shared_fro_norm" in stats[0]["consolidation"]


def test_the_first_task_becomes_the_shared_branch(lab):
    method = make_method(lab)
    learn(method, 1)
    expert1 = dense(adapters.effective_factors(method.model, "expert_001_alpha"))
    shared1 = dense(adapters.effective_factors(method.model, "shared"))
    for (
        name,
        matrix,
    ) in shared1.items():  # c_1 = 0, empty history: Co_1 is the (rank-preserving) re-factorisation of the expert
        assert torch.allclose(matrix, expert1[name], atol=2e-4)
    assert method.de.experts == {"alpha": "expert_001_alpha"}


@pytest.mark.parametrize(
    ("scaling", "projection"),
    [("sqrt", "null_space"), ("sqrt", "none"), ("additive", "none"), ("additive", "null_space")],
)
def test_fold_is_the_scaled_null_space_retraction(lab, scaling, projection):
    method = make_method(lab, **{"colora.scaling": scaling, "colora.projection": projection})
    learn(method, 3)
    c, s = retract_coefficients(scaling, 3)
    context = dense(adapters.effective_factors(method.model, method.contexts["gamma"]))  # Co_2
    expert = dense(adapters.effective_factors(method.model, method.de.experts["gamma"]))
    shared = dense(adapters.effective_factors(method.model, "shared"))  # Co_3
    for name, history in context.items():
        update = expert[name]
        if projection == "null_space":
            u, sv, _ = torch.linalg.svd(history, full_matrices=False)
            u = u[:, sv > 1e-6 * sv[0]]
            update = update - u @ (u.T @ update)
            assert torch.allclose(u.T @ update, torch.zeros(u.shape[1], update.shape[1]).double(), atol=1e-4)
        assert torch.allclose(shared[name], truncated_svd(c * history + s * update, method.cfg.lora.r), atol=2e-4)


def test_the_expert_is_trained_with_the_frozen_history_active(lab, monkeypatch):
    from codelora.methods import codelora

    seen = []
    train = codelora.train_task
    monkeypatch.setattr(
        codelora,
        "train_task",
        lambda model, branches, *a, **k: seen.append(branches) or train(model, branches, *a, **k),
    )
    learn(make_method(lab), 2)
    assert [[(b.adapters, b.trainable, b.penalty) for b in branches] for branches in seen] == [
        [(["expert_001_alpha"], "expert_001_alpha", None)],
        [(["shared", "expert_002_beta"], "expert_002_beta", None)],
    ]


def test_context_is_the_shared_branch_before_the_task(lab):
    method = make_method(lab)
    learn(method, 1)
    before = dense(adapters.effective_factors(method.model, "shared"))
    method.learn_task(2, method.cfg.tasks[1], method.ctx.benchmark.train(2))
    assert set(method.contexts) == {"beta"}  # task 1 was trained on the bare base: no context
    context = dense(adapters.effective_factors(method.model, method.contexts["beta"]))
    assert all(torch.allclose(context[name], matrix, atol=1e-6) for name, matrix in before.items())
    assert method.serving_adapters("alpha") == "expert_001_alpha"
    assert method.serving_adapters("beta") == ("context_002", "expert_002_beta")
    assert method.serving_adapters("shared") == "shared"


def test_loading_replays_the_folds_bit_exactly_and_keeps_predictions(lab):
    method = make_method(lab)
    learn(method, 3)
    state = method.state_dict()
    assert set(state) == {"experts", "adapters", "router"}  # only experts and router statistics are stored

    restored = [make_method(lab) for _ in range(2)]  # the fold is deterministic: two replays agree with each other ...
    for other in restored:
        other.load_state_dict(state)
        other.ctx.benchmark = method.ctx.benchmark
    assert restored[0].de.experts == method.de.experts and set(restored[0].contexts) == {"beta", "gamma"}
    for other in restored:  # ... and with the folds of the training run
        for task, name in method.contexts.items():
            original = adapters.effective_factors(method.model, name)
            rebuilt = adapters.effective_factors(other.model, other.contexts[task])
            assert all(
                torch.equal(rebuilt[k][0], original[k][0]) and torch.equal(rebuilt[k][1], original[k][1])
                for k in original
            )
        final = adapters.effective_factors(method.model, "shared")
        replayed = adapters.effective_factors(other.model, "shared")
        assert all(torch.equal(replayed[k][0], final[k][0]) and torch.equal(replayed[k][1], final[k][1]) for k in final)

    examples = method.ctx.benchmark.eval(3)
    assert [(p.text, p.route) for p in restored[0].predict(examples)] == [
        (p.text, p.route) for p in method.predict(examples)
    ]


class FixedRouter:
    def __init__(self, task, confidence):
        self.answer = (task, confidence)

    def route(self, embedding):
        return self.answer


def test_low_confidence_falls_back_to_the_shared_branch(lab):
    method = make_method(lab, **{"router.threshold": 0.5, "fusion.enabled": False})
    method.de.new_expert(1, "alpha")
    examples = [Example("w1 w2 w3", "cat", "alpha", source="w1 w2 w3")] * 3
    method.de.router = FixedRouter("alpha", 0.5)  # strict ">": equality goes to the shared branch
    assert {p.route for p in method.predict(examples)} == {"shared"}
    method.de.router = FixedRouter("alpha", 0.51)
    predictions = method.predict(examples)
    assert {p.route for p in predictions} == {"alpha"} and predictions[0].confidence == 0.51
