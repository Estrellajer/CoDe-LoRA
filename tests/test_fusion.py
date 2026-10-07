"""PMI-calibrated Co/expert fusion: the rule on hand-made scores, and its wiring into CoDe-LoRA on the tiny model."""

from __future__ import annotations

import pytest
import torch

from codelora.evaluation import fusion
from codelora.evaluation.branch_scores import label_logprobs
from tests.test_codelora import learn, make_method

GRID = [0.0, 0.1, 0.25, 0.5, 1.0]


def lp(*rows):
    return torch.log(torch.tensor(rows, dtype=torch.float64))


def test_prior_is_the_mean_predicted_distribution_and_pmi_removes_it():
    scores = lp([0.9, 0.1], [0.7, 0.3])  # a branch that always favours label 0
    assert fusion.prior(scores).tolist() == pytest.approx([0.8 + 1e-9, 0.2 + 1e-9])
    calibrated = fusion.pmi(scores, fusion.prior(scores)).exp()
    assert calibrated.sum(dim=-1).tolist() == pytest.approx([1.0, 1.0])
    assert calibrated[0, 0] < 0.9  # the prior is divided out: the favoured label loses its head start


def test_unnormalised_sequence_scores_are_renormalised_over_the_candidates():
    raw = torch.tensor([[-3.0, -4.0], [-10.0, -9.5]], dtype=torch.float64)  # sums of token log-probs, not distributions
    assert fusion.normalize(raw).exp().sum(dim=-1).tolist() == pytest.approx([1.0, 1.0])
    assert fusion.prior(raw).sum().item() == pytest.approx(1.0, abs=1e-8)


def test_alpha_is_chosen_on_dev_and_ties_go_to_the_smaller_alpha():
    gold = [0, 1, 0, 1]
    co = lp([0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.4, 0.6])  # always right
    expert = lp([0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.6, 0.4])  # always wrong
    assert fusion.fit(co, expert, gold, GRID).alpha == 1.0
    assert fusion.fit(expert, co, gold, GRID).alpha == 0.0
    rule = fusion.fit(co, co, gold, GRID)  # identical branches: every alpha is equally good
    assert rule.alpha == 0.0 and rule.dev_accuracy == 1.0


def test_fused_prediction_follows_the_rule_on_unseen_examples():
    dev_co = lp([0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.4, 0.6])
    dev_de = lp([0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.6, 0.4])
    rule = fusion.fit(dev_co, dev_de, [0, 1, 0, 1], GRID)  # Co is the better branch: alpha = 1
    test_co, test_de = lp([0.9, 0.1], [0.2, 0.8]), lp([0.1, 0.9], [0.8, 0.2])
    assert rule.predict(test_co, test_de) == [0, 1]
    manual = rule.alpha * fusion.pmi(test_co, rule.co_prior) + (1 - rule.alpha) * fusion.pmi(test_de, rule.expert_prior)
    assert torch.equal(rule.scores(test_co, test_de), manual)


def test_code_lora_scores_candidates_under_both_branches_and_fuses(lab):
    method = make_method(lab, **{"fusion.enabled": True, "fusion.dev_samples": 6})
    learn(method, 2)
    bench = method.ctx.benchmark
    examples, labels = bench.eval(2), bench.labels("beta")
    predictions = method.predict(examples)

    assert {p.route for p in predictions} <= {"alpha", "beta"}
    assert all(p.text in labels for p in predictions)
    rule = method._fusions["beta"]
    assert len(bench.dev("beta")) == 6 and rule.alpha in GRID
    for p in predictions:
        assert p.extra["fusion"]["alpha"] == rule.alpha and len(p.extra["fusion"]["co"]) == len(labels)

    # the stored scores are the raw branch log-probabilities, and the prediction is the rule applied to them
    co = torch.tensor([p.extra["fusion"]["co"] for p in predictions], dtype=torch.float64)
    expert = torch.tensor([p.extra["fusion"]["expert"] for p in predictions], dtype=torch.float64)
    shared = label_logprobs(
        method.model, method.ctx.backbone, method.ctx.collator, examples, labels, ["shared"] * len(examples), 4
    )
    assert torch.allclose(co, shared)
    assert [labels[i] for i in rule.predict(co, expert)] == [p.text for p in predictions]


def test_fusion_is_refitted_when_the_model_changes_and_can_be_switched_off(lab):
    method = make_method(lab, **{"fusion.enabled": True, "fusion.dev_samples": 6})
    learn(method, 1)
    method.predict(method.ctx.benchmark.eval(1))
    assert set(method._fusions) == {"alpha"}
    method.learn_task(2, method.cfg.tasks[1], method.ctx.benchmark.train(2))
    assert not method._fusions  # priors and alpha belong to one state of Co / the experts

    unseen = method.predict(method.ctx.benchmark.eval(3))  # gamma is not learned yet: generated, its dev split unread
    assert all(p.extra is None for p in unseen) and "gamma" not in method._fusions

    plain = make_method(lab, **{"fusion.enabled": False})
    learn(plain, 1)
    assert all(p.extra is None for p in plain.predict(plain.ctx.benchmark.eval(1)))


def test_fusion_stages_final_generates_before_the_last_stage_and_all_fuses_every_stage(lab):
    for stages, fused_early in (("final", False), ("all", True)):
        method = make_method(lab, **{"fusion.enabled": True, "fusion.dev_samples": 6, "fusion.stages": stages})
        learn(method, 1)
        n_tasks = len(method.cfg.tasks)
        method.ctx.stage = 1  # an intermediate stage of a longer run
        early = method.predict(method.ctx.benchmark.eval(1))
        assert (early[0].extra is not None) == fused_early
        method.ctx.stage = n_tasks  # the last stage
        assert method.predict(method.ctx.benchmark.eval(1))[0].extra is not None
