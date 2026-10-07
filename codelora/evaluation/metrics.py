"""Continual-learning metrics from the performance matrix ``performance[after_task_XXX][task]``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ContinualMetrics:
    average_performance: float  # mean final score over all tasks (AP)
    backward_transfer: float  # mean(final - just-learned) over all but the last task (BWT)
    diagonal_average: float
    final_scores: dict[str, float]
    diagonal_scores: dict[str, float]


def mean(values: Sequence[float]) -> float:
    """Plain left-to-right mean (0.0 when empty), the same summation order everywhere scores are averaged."""

    total = 0.0
    for value in values:
        total += value
    return total / len(values) if values else 0.0


def stage_key(index: int) -> str:
    return f"after_task_{index:03d}"


def summarize_matrix(performance: Mapping[str, Mapping[str, float]], task_names: Sequence[str]) -> ContinualMetrics:
    final = {name: float(performance[stage_key(len(task_names))][name]) for name in task_names}
    diagonal = {name: float(performance[stage_key(i)][name]) for i, name in enumerate(task_names, start=1)}
    backward = [final[name] - diagonal[name] for name in task_names[:-1]]
    return ContinualMetrics(
        average_performance=sum(final.values()) / len(final),
        backward_transfer=sum(backward) / len(backward) if backward else 0.0,
        diagonal_average=sum(diagonal.values()) / len(diagonal),
        final_scores=final,
        diagonal_scores=diagonal,
    )


def stage_tasks(matrix: str, task_index: int, task_count: int) -> range | tuple[int, ...]:
    """1-based indices of the learned tasks scored after task ``task_index``."""

    if matrix == "full" or task_index == task_count:
        return range(1, task_index + 1)
    return (task_index,)
