"""Analysis tables of a run (``results.json``): per-answer scores and routing counts per evaluated (stage, task)."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA_VERSION = 1
MAX_CLASSES = 50  # tasks with more distinct answers (generation, TRACE) have no per-class table


def per_class_scores(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]] | None:
    """``{answer: {n, score}}`` (mean score per gold answer) for classification-like tasks, else None."""

    groups: dict[str, list[float]] = defaultdict(list)
    for record in records:
        groups[record["target"]].append(record["score"])
        if len(groups) > MAX_CLASSES:
            return None
    return {label: {"n": len(scores), "score": sum(scores) / len(scores)} for label, scores in sorted(groups.items())}


def routing_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """How many examples went to each route and the mean router confidence (None for unrouted methods)."""

    routes = [r["route"] for r in records]
    if all(route is None for route in routes):
        return None
    confidences = [r["confidence"] for r in records if r["confidence"] is not None]
    return {
        "counts": dict(sorted(Counter(routes).items())),
        "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
    }


class ResultsBuilder:
    def __init__(self, per_class: bool) -> None:
        self.per_class_enabled = per_class
        self.per_class: dict[str, dict[str, Any]] = {}
        self.routing: dict[str, dict[str, Any]] = {}

    def add(self, stage: str, task: str, records: Sequence[Mapping[str, Any]]) -> None:
        if self.per_class_enabled and (table := per_class_scores(records)) is not None:
            self.per_class.setdefault(stage, {})[task] = table
        if (counts := routing_counts(records)) is not None:
            self.routing.setdefault(stage, {})[task] = counts

    def build(self, **fixed: Any) -> dict[str, Any]:
        result = {"schema_version": SCHEMA_VERSION, **fixed}
        result["per_class"] = self.per_class
        result["routing"] = self.routing
        return result


def forward_transfer_score(delta_matrix: Mapping[str, Mapping[str, float]], task_names: Sequence[str]) -> float | None:
    """FWT: mean over tasks ``i >= 2`` of the score on task ``i`` before it is learned minus the untrained model's score."""

    values = [
        delta_matrix[f"after_task_{i:03d}"][task_names[i]]
        for i in range(1, len(task_names))
        if f"after_task_{i:03d}" in delta_matrix
    ]
    return sum(values) / len(values) if values else None
