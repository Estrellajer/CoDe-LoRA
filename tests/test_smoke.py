"""End-to-end CPU smoke tests: the real entry points on tiny models, single process and gloo 2 ranks."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from codelora.methods import METHODS
from tests import launch

REPO = Path(__file__).resolve().parents[1]
ENV = {
    **os.environ,
    "PYTHONPATH": str(REPO),
    "OMP_NUM_THREADS": "1",
    "HF_HUB_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
TARGETS = {"t5": ["q", "v"], "qwen": ["q_proj", "v_proj"]}
NAMES = {"t5": "t5-large", "qwen": "qwen3-0.6b"}


def write_config(lab, tmp_path, backbone: str, method: str, **extra) -> Path:
    config = {
        "defaults": ["tiny/smoke", f"methods/{method}"],
        "name": f"{backbone}-{method}",
        "output_dir": str(tmp_path / "runs"),
        "backbone": {"name": NAMES[backbone], "path": str(lab[backbone])},
        "lora": {"target_modules": TARGETS[backbone]},
        "data": {"root": str(lab["data"])},
        "tasks": [
            {"name": "alpha", "path": "TC/alpha"},
            {"name": "beta", "path": "SC/beta"},
            {"name": "gamma", "path": "TC/gamma"},
        ],
        "router": {"kind": "lda", "threshold": 0.0, "embedding_input": "raw_source", "lda_samples": 30},
        "mole": {"memory_budget_per_task": 6, "prototype_num_samples": 12},
        **extra,
    }
    path = tmp_path / f"{backbone}-{method}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def run_cli(module: str, args: list[str], world: int = 1) -> subprocess.CompletedProcess:
    done = subprocess.run(
        [*launch.python(world), "-m", module, *args], cwd=REPO, env=ENV, capture_output=True, text=True
    )
    assert done.returncode == 0, f"{done.stdout[-2000:]}\n{done.stderr[-3000:]}"
    return done


def first_step_loss(stdout: str) -> float:
    for line in stdout.splitlines():
        if line.startswith('{"event": "train_step"'):
            return json.loads(line)["branches"][0]["loss"]
    raise AssertionError("no train_step logged")


@pytest.mark.parametrize("method", sorted(METHODS))
def test_train_then_audit_and_reload(lab, tmp_path, method):
    done = run_cli("codelora.train", [str(write_config(lab, tmp_path, "t5", method))])
    run_dir = tmp_path / "runs" / f"t5-{method}"
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "passed" and summary["world_size"] == 1
    assert [t["task"] for t in summary["tasks"]] == ["alpha", "beta", "gamma"]
    assert set(summary["performance_matrix"]) == {"after_task_001", "after_task_002", "after_task_003"}
    assert len(summary["performance_matrix"]["after_task_003"]) == 3  # diagonal_final: a_ii plus the whole final row
    assert 0.0 <= summary["average_performance"] <= 1.0
    assert (run_dir / "config.yaml").is_file() and (run_dir / "checkpoint.pt").is_file()
    assert done.stdout.count('"event": "train_step"') >= 3

    audit = json.loads(run_cli("codelora.evaluate", [str(run_dir), "--reload"]).stdout)
    assert audit["max_cell_error"] < 1e-12
    assert audit["AP"] == pytest.approx(audit["stored_AP"]) and audit["BWT"] == pytest.approx(audit["stored_BWT"])
    assert audit["reload_max_abs_diff"] == 0.0  # the checkpoint reproduces the run exactly


def test_decoder_backbone_runs_every_method_family(lab, tmp_path):
    for method in ("lora", "code-lora", "mole-cie"):
        run_cli("codelora.train", [str(write_config(lab, tmp_path, "qwen", method))])
        summary = json.loads((tmp_path / "runs" / f"qwen-{method}" / "summary.json").read_text())
        assert summary["status"] == "passed"


@pytest.mark.parametrize("method", ["lora", "code-lora", "mole-cie"])
def test_two_ranks_match_one_rank_statistically(lab, tmp_path, method):
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    single = run_cli("codelora.train", [str(write_config(lab, tmp_path / "one", "t5", method))])
    double = run_cli("codelora.train", [str(write_config(lab, tmp_path / "two", "t5", method))], world=2)
    summary = json.loads((tmp_path / "two" / "runs" / f"t5-{method}" / "summary.json").read_text())
    assert summary["world_size"] == 2 and summary["status"] == "passed"
    # same global micro-batch, split in two: the first loss agrees up to bf16 shape-dependent rounding
    assert first_step_loss(double.stdout) == pytest.approx(first_step_loss(single.stdout), rel=0.05)


def test_code_lora_with_pmi_fusion_trains_audits_and_reloads(lab, tmp_path):
    config = write_config(lab, tmp_path, "t5", "code-lora", fusion={"enabled": True})
    run_cli("codelora.train", [str(config)])
    run_dir = tmp_path / "runs" / "t5-code-lora"
    assert json.loads((run_dir / "summary.json").read_text())["status"] == "passed"
    records = json.loads((run_dir / "evaluations" / "after_task_003" / "alpha.json").read_text())
    assert all(r["fusion"]["alpha"] in (0.0, 0.1, 0.25, 0.5, 1.0) for r in records)
    audit = json.loads(run_cli("codelora.evaluate", [str(run_dir), "--reload"]).stdout)
    assert audit["reload_max_abs_diff"] == 0.0  # replaying the folds and refitting the fusion on dev reproduces the run


def test_code_lora_on_generation_tasks_ignores_fusion(lab, tmp_path):
    """TRACE-style open generation has no candidate labels: CoDe-LoRA trains, routes and generates as usual."""

    config = write_config(lab, tmp_path, "t5", "code-lora", fusion={"enabled": True})
    spec = yaml.safe_load(config.read_text())
    spec["data"] = {"root": str(lab["trace"])}
    spec["tasks"] = [
        {"name": "C-STANCE", "path": "C-STANCE", "adapter": "trace"},
        {"name": "MeetingBank", "path": "MeetingBank", "adapter": "trace"},
    ]
    config.write_text(yaml.safe_dump(spec))
    run_cli("codelora.train", [str(config)])
    summary = json.loads((tmp_path / "runs" / "t5-code-lora" / "summary.json").read_text())
    assert summary["status"] == "passed" and len(summary["performance_matrix"]["after_task_002"]) == 2
