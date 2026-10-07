import pytest

from codelora.evaluation.metrics import stage_key, stage_tasks, summarize_matrix


def test_ap_and_bwt_from_the_performance_matrix():
    matrix = {
        stage_key(1): {"a": 0.9},
        stage_key(2): {"a": 0.7, "b": 0.8},
        stage_key(3): {"a": 0.5, "b": 0.6, "c": 0.7},
    }
    metrics = summarize_matrix(matrix, ["a", "b", "c"])
    assert metrics.average_performance == pytest.approx((0.5 + 0.6 + 0.7) / 3)
    assert metrics.backward_transfer == pytest.approx(((0.5 - 0.9) + (0.6 - 0.8)) / 2)
    assert metrics.diagonal_scores == {"a": 0.9, "b": 0.8, "c": 0.7}


def test_single_task_has_zero_bwt():
    assert summarize_matrix({stage_key(1): {"a": 0.4}}, ["a"]).backward_transfer == 0.0


def test_stage_task_selection():
    assert list(stage_tasks("full", 2, 4)) == [1, 2]
    assert list(stage_tasks("diagonal_final", 2, 4)) == [2]
    assert list(stage_tasks("diagonal_final", 4, 4)) == [1, 2, 3, 4]
