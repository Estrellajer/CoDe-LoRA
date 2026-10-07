"""Forward-transfer callback: wiring, matrices and the untrained baseline."""

from __future__ import annotations

import json

import pytest

from codelora.config import load_config
from codelora.evaluation.callbacks import Callback, active_callbacks
from codelora.evaluation.forward_transfer import ForwardTransfer
from codelora.runner import run
from tests.test_smoke import write_config


def test_callbacks_are_off_by_default_and_enabled_by_config():
    assert active_callbacks(load_config(["methods/lora"])) == []
    (callback,) = active_callbacks(load_config(["methods/lora"], ["eval.forward_transfer=true"]))
    assert isinstance(callback, ForwardTransfer) and isinstance(callback, Callback)


def test_matrices_baseline_and_deltas(lab, tmp_path):
    path = write_config(
        lab, tmp_path, "t5", "lora", eval={"forward_transfer": True, "batch_size": 5, "matrix": "diagonal_final"}
    )
    run_dir = run(load_config(path))
    summary = json.loads((run_dir / "summary.json").read_text())
    baseline, matrix, delta = (
        summary[k]
        for k in ("forward_transfer_baseline_matrix", "forward_transfer_matrix", "forward_transfer_delta_matrix")
    )
    assert set(baseline) == {"alpha", "beta", "gamma"}
    assert set(matrix) == {"after_task_001", "after_task_002", "after_task_003"}
    assert set(matrix["after_task_001"]) == {"beta", "gamma"} and set(matrix["after_task_002"]) == {"gamma"}
    assert matrix["after_task_003"] == {} and delta["after_task_003"] == {}  # nothing left unseen
    for stage, row in matrix.items():
        for task, score in row.items():
            assert delta[stage][task] == pytest.approx(score - baseline[task])


def test_disabled_runs_carry_no_forward_transfer_entries(lab, tmp_path):
    run_dir = run(load_config(write_config(lab, tmp_path, "t5", "lora")))
    assert "forward_transfer_matrix" not in json.loads((run_dir / "summary.json").read_text())
