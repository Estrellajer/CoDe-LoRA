"""Run recording, result tables and run artifacts (docs/RESULTS.md)."""

from __future__ import annotations

import json

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from codelora.config import LogCfg
from codelora.evaluation.branch_scores import sequence_logprobs
from codelora.evaluation.results import forward_transfer_score, per_class_scores, routing_counts
from codelora.tracking import Recorder, TensorBoardSink

from .test_smoke import run_cli, write_config


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def step(rec, n, loss=1.0):
    rec.train_step("task", n, [{"adapter": "shared", "loss": loss, "penalty": 0.5, "grad_norm": 2.0}], 1e-3)


def test_recorder_windows_the_train_events_and_prints_every_step(tmp_path, capsys):
    rec = Recorder(tmp_path, LogCfg(interval=3, stdout_interval=2))
    rec.index = 4
    for n in range(1, 8):
        step(rec, n, loss=float(n))
    rec.close()
    events = lines(tmp_path / "log.jsonl")
    assert [(e["event"], e["step"]) for e in events] == [("train", 3), ("train", 6)]
    assert events[0]["loss_mean"] == pytest.approx(2.0) and events[1]["loss_mean"] == pytest.approx(
        5.0
    )  # mean over the window
    assert events[0]["index"] == 4 and events[0]["lr"] == 1e-3 and events[0]["branches"][0]["penalty"] == 0.5
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [p["step"] for p in printed] == [2, 4, 6] and printed[0]["event"] == "train_step"
    assert "train" in (tmp_path / "train.log").read_text()


def test_null_recorder_only_prints(capsys):
    rec = Recorder()
    step(rec, 1)
    assert json.loads(capsys.readouterr().out)["step"] == 1


def test_tensorboard_sink_needs_the_optional_package(tmp_path):
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            TensorBoardSink(tmp_path)
    else:
        sink = TensorBoardSink(tmp_path)
        sink.scalars({"train/loss": 1.0}, 1)
        sink.close()
        assert any((tmp_path).iterdir())


def test_result_helpers():
    records = [
        {"target": "a", "score": 1.0, "route": "t1", "confidence": 0.9},
        {"target": "a", "score": 0.0, "route": "shared", "confidence": 0.5},
        {"target": "b", "score": 1.0, "route": "t1", "confidence": 0.7},
    ]
    assert per_class_scores(records) == {"a": {"n": 2, "score": 0.5}, "b": {"n": 1, "score": 1.0}}
    routing = routing_counts(records)
    assert routing["counts"] == {"shared": 1, "t1": 2} and routing["mean_confidence"] == pytest.approx(0.7)
    assert routing_counts([{"route": None, "confidence": None}]) is None
    many = [{"target": str(i), "score": 0.0} for i in range(51)]
    assert per_class_scores(many) is None  # generation task: no per-class table


def test_forward_transfer_score_uses_the_score_before_learning():
    delta = {"after_task_001": {"b": 0.2, "c": 0.4}, "after_task_002": {"c": 0.1}}
    assert forward_transfer_score(delta, ["a", "b", "c"]) == pytest.approx((0.2 + 0.1) / 2)
    assert forward_transfer_score({}, ["a"]) is None


@pytest.mark.parametrize("kind", ["t5", "qwen"])
def test_sequence_logprobs_match_the_model_loss(lab, kind):
    from transformers import AutoTokenizer

    from codelora.data.collate import CausalLMCollator, Seq2SeqCollator
    from codelora.data.tasks import Example

    tok = AutoTokenizer.from_pretrained(lab[kind])
    if kind == "qwen":
        tok.pad_token = tok.eos_token
        model, collator = (
            AutoModelForCausalLM.from_pretrained(lab[kind], dtype=torch.float32),
            CausalLMCollator(tok, 32, 6),
        )
    else:
        model, collator = (
            AutoModelForSeq2SeqLM.from_pretrained(lab[kind], dtype=torch.float32),
            Seq2SeqCollator(tok, 32, 6),
        )
    model.eval()
    batch = collator([Example("w1 w2 w3", "cat", "t"), Example("w4", "dog bird", "t")])
    with torch.no_grad():
        scores = sequence_logprobs(model, batch, causal=kind == "qwen")
        for row in range(2):
            single = {k: v[row : row + 1] for k, v in batch.items()}
            tokens = (single["labels"][:, 1:] if kind == "qwen" else single["labels"]).ne(-100).sum().item()
            loss = model(**single).loss.item()
            assert scores[row] == pytest.approx(-loss * tokens, rel=1e-4)


def test_checkpoint_policies(lab, tmp_path):
    for policy, expected in {
        "none": [],
        "final": ["checkpoint.pt"],
        "all": ["checkpoint.pt", "checkpoints/after_task_001.pt", "checkpoints/after_task_002.pt"],
        "every:2": ["checkpoint.pt", "checkpoints/after_task_002.pt"],
    }.items():
        root = tmp_path / policy.replace(":", "_")
        root.mkdir()
        run_cli("codelora.train", [str(write_config(lab, root, "t5", "lora", save_checkpoint=policy))])
        run_dir = root / "runs" / "t5-lora"
        found = sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*.pt"))
        assert found == sorted(expected), policy


def test_full_artifact_set_of_a_code_lora_run(lab, tmp_path):
    config = write_config(
        lab,
        tmp_path,
        "t5",
        "code-lora",
        eval={"matrix": "full", "forward_transfer": True},
        log={"interval": 2, "profile": True},
        router={"kind": "cosine", "threshold": 0.0},
        fusion={"enabled": True},
    )
    run_cli("codelora.train", [str(config)])
    run_dir = tmp_path / "runs" / "t5-code-lora"
    run = json.loads((run_dir / "run.json").read_text())
    assert run["schema_version"] == 1 and run["environment"]["torch"] and run["parameters"]["base"] > 0
    assert (
        list(run["parameters"]["adapter_after_task"]) == ["alpha", "beta", "gamma"]
        and run["finished_at"] >= run["started_at"]
    )

    events = lines(run_dir / "log.jsonl")
    kinds = [e["event"] for e in events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    assert kinds.count("task_start") == kinds.count("task_end") == 3 and kinds.count("stage_end") == 3
    train = [e for e in events if e["event"] == "train"]
    assert train and all(e["step"] % 2 == 0 and len(e["branches"]) == 1 for e in train)  # CoDe-LoRA: one pass
    end = next(e for e in events if e["event"] == "task_end")
    assert end["trainable_parameters"] > 0 and end["adapter_parameters"] > 0 and end["train_seconds"] > 0
    assert (run_dir / "train.log").read_text().count("task_end") == 3

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["schema_version"] == 1 and summary["efficiency"]["adapter_parameters"] > 0
    assert summary["FWT"] is not None and "forward_transfer_delta_matrix" in summary
    assert (
        len(summary["performance_matrix"]["after_task_003"]) == 3
        and len(summary["performance_matrix"]["after_task_002"]) == 2
    )

    results = json.loads((run_dir / "results.json").read_text())
    assert (
        results["schema_version"] == 1
        and results["matrix_kind"] == "full"
        and results["metrics"]["FWT"] == summary["FWT"]
    )
    efficiency = results["efficiency"]
    assert efficiency == {**summary["efficiency"], "per_task": efficiency["per_task"]}
    assert efficiency["total_seconds"] >= efficiency["train_seconds"] + efficiency["eval_seconds"] > 0
    assert 0 < efficiency["adapter_fraction"] < 1
    per_task = efficiency["per_task"]
    assert [t["task"] for t in per_task] == ["alpha", "beta", "gamma"]
    assert all(t["train_seconds"] > 0 and t["eval_seconds"] >= sum(t["eval_task_seconds"].values()) for t in per_task)
    phases = per_task[1]["train_phases"]
    assert {
        "data",
        "activate",
        "forward_backward:expert",
        "loss_sync",
        "grad_allreduce",
    } <= set(phases)
    assert {"clip_step", "consolidation", "routing_registration"} <= set(phases)
    assert {"route", "generate"} <= set(per_task[1]["eval_phases"])  # fusion.stages=final: intermediate stages generate
    assert "label_scores" not in per_task[1]["eval_phases"] and "label_scores" in per_task[2]["eval_phases"]
    assert set(results["label_sets"]) == {"alpha", "beta", "gamma"} and results["label_sets"]["beta"] == [
        "positive",
        "negative",
    ]
    assert set(results["per_class"]["after_task_003"]["alpha"]) <= set(results["label_sets"]["alpha"])
    assert sum(results["routing"]["after_task_003"]["beta"]["counts"].values()) == 12

    records = json.loads((run_dir / "evaluations" / "after_task_003" / "beta.json").read_text())
    assert len(records) == 12 and all(r["index"] is not None and r["target"] and "prediction" in r for r in records)
    first = records[0]["fusion"]
    assert len(first["co"]) == 2 and len(first["expert"]) == 2 and all(v <= 0 for v in first["co"] + first["expert"])
    assert first["alpha"] in (0.0, 0.1, 0.25, 0.5, 1.0) and records[0]["prediction"] in ("positive", "negative")


def test_default_artifacts_are_light(lab, tmp_path):
    run_cli("codelora.train", [str(write_config(lab, tmp_path, "t5", "lora"))])
    run_dir = tmp_path / "runs" / "t5-lora"
    assert (
        not (run_dir / "routing").exists() and not (run_dir / "tb").exists() and not (run_dir / "checkpoints").exists()
    )
    record = json.loads((run_dir / "evaluations" / "after_task_003" / "alpha.json").read_text())[0]
    assert set(record) == {"index", "input", "target", "prediction", "route", "confidence", "score"}
    assert json.loads((run_dir / "results.json").read_text())["routing"] == {}  # unrouted method


def test_only_rank_zero_writes_and_sharded_scores_are_merged(lab, tmp_path):
    config = write_config(lab, tmp_path, "t5", "code-lora", fusion={"enabled": True})
    run_cli("codelora.train", [str(config)], world=2)
    run_dir = tmp_path / "runs" / "t5-code-lora"
    events = lines(run_dir / "log.jsonl")
    assert [e["event"] for e in events].count("run_start") == 1 and [e["event"] for e in events].count("task_end") == 3
    records = json.loads((run_dir / "evaluations" / "after_task_003" / "alpha.json").read_text())
    assert len(records) == 12 and [r["index"] for r in records] == sorted(r["index"] for r in records)
    assert all(len(r["fusion"]["co"]) == 4 for r in records)  # alpha has four candidate answers; every row present
    assert json.loads((run_dir / "run.json").read_text())["world_size"] == 2
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == ["t5-code-lora"]
