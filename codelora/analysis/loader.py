"""Discover run directories and read the numbers analyses need (tolerating missing optional files)."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def stage_key(index: int) -> str:
    return f"after_task_{index:03d}"


@dataclass
class Run:
    """One finished run: ``summary.json`` + ``config.yaml`` (+ ``results.json``, ``run.json`` when present)."""

    path: Path
    config: dict
    summary: dict
    results: dict | None = None
    info: dict | None = None
    _records: dict = field(default_factory=dict, repr=False)

    @property
    def method(self) -> str:
        return self.summary.get("method") or self.config.get("method", "?")

    @property
    def seed(self) -> int:
        return int(self.summary.get("seed", self.config.get("seed", 0)))

    @property
    def backbone(self) -> str:
        return self.config.get("backbone", {}).get("name", "?")

    @property
    def tasks(self) -> list[str]:
        if self.config.get("tasks"):
            return [t["name"] for t in self.config["tasks"]]
        return list(self.summary.get("diagonal_scores", {}))

    @property
    def benchmark(self) -> str:
        return benchmark_label(self.tasks)

    @property
    def performance(self) -> dict[str, dict[str, float]]:
        return self.summary["performance_matrix"]

    @property
    def matrix_kind(self) -> str:
        """``full`` when every learned task is scored after every stage, else ``diagonal_final``."""

        tasks = self.tasks
        full = all(len(self.performance.get(stage_key(k), {})) == k for k in range(1, len(tasks) + 1))
        return "full" if full else "diagonal_final"

    @property
    def ap(self) -> float:
        return float(self.summary["average_performance"])

    @property
    def bwt(self) -> float:
        return float(self.summary["backward_transfer"])

    def diagonal(self) -> dict[str, float]:
        return {t: self.performance[stage_key(i)][t] for i, t in enumerate(self.tasks, start=1)}

    def final(self) -> dict[str, float]:
        return dict(self.performance[stage_key(len(self.tasks))])

    def peak(self) -> dict[str, float]:
        """Best score a task ever had in the stored matrix (the diagonal for ``diagonal_final``)."""

        peaks = {}
        for i, task in enumerate(self.tasks, start=1):
            seen = [
                self.performance[stage_key(k)][task]
                for k in range(i, len(self.tasks) + 1)
                if task in self.performance.get(stage_key(k), {})
            ]
            peaks[task] = max(seen)
        return peaks

    def curve(self) -> list[float]:
        """AP after learning ``k`` tasks, ``k = 1..T`` (needs the full matrix)."""

        if self.matrix_kind != "full":
            raise ValueError(f"{self.path}: AP-vs-tasks curves need eval.matrix=full")
        return [
            sum(self.performance[stage_key(k)][t] for t in self.tasks[:k]) / k for k in range(1, len(self.tasks) + 1)
        ]

    def forward_transfer(self) -> dict[str, dict[str, float]] | None:
        return self.summary.get("forward_transfer_delta_matrix") or None

    @property
    def fwt(self) -> float | None:
        """Mean over tasks ``i >= 2`` of the gain of task ``i`` over the untrained model before it is learned."""

        stored = self.summary.get("FWT")
        if stored is not None:
            return float(stored)
        delta = self.forward_transfer()
        if not delta:
            return None
        values = [
            delta[stage_key(i - 1)][t]
            for i, t in enumerate(self.tasks, start=1)
            if i >= 2 and t in delta.get(stage_key(i - 1), {})
        ]
        return sum(values) / len(values) if values else None

    def routing_counts(self, stage: str | None = None) -> dict[str, dict[str, int]]:
        """``{gold task: {route: count}}`` at ``stage`` (default: the last), from results.json or the evaluation records."""

        stage = stage or stage_key(len(self.tasks))
        if self.results and self.results.get("routing", {}).get(stage):
            return {t: dict(v["counts"]) for t, v in self.results["routing"][stage].items()}
        counts: dict[str, dict[str, int]] = {}
        for task in self.tasks:
            records = self.records(stage, task)
            if records and records[0].get("route") is not None:
                tally: dict[str, int] = defaultdict(int)
                for record in records:
                    tally[record["route"]] += 1
                counts[task] = dict(tally)
        return counts

    def records(self, stage: str, task: str) -> list[dict] | None:
        key = (stage, task)
        if key not in self._records:
            self._records[key] = _read_json(self.path / "evaluations" / stage / f"{task}.json")
        return self._records[key]

    def per_task_efficiency(self) -> list[dict]:
        """Per task: train / evaluation seconds, peak memory, adapter parameters and the phase breakdown (``results.json``)."""

        return (self.results or {}).get("efficiency", {}).get("per_task", [])

    def efficiency(self) -> dict[str, float]:
        stored = (self.results or {}).get("efficiency") or self.summary.get("efficiency")
        if stored:
            return {k: float(v) for k, v in stored.items() if isinstance(v, (int, float))}
        tasks = self.summary.get("tasks", [])
        found = {
            "train_seconds": sum(t.get("train_seconds", 0.0) for t in tasks),
            "eval_seconds": sum(self.summary.get("evaluation_seconds", {}).values()),
            "peak_memory_mib": float(self.summary.get("peak_allocated_mib", 0.0)),
        }
        adapters = (self.info or {}).get("parameters", {}).get("adapter_after_task", {})
        if adapters:
            found["adapter_parameters"] = float(list(adapters.values())[-1])
        return found

    def embeddings(self, kind: str, task: str):
        """``routing/<kind>_<task>.pt`` tensor ([n, d]) or None."""

        path = self.path / "routing" / f"{kind}_{task}.pt"
        if not path.is_file():
            return None
        import torch

        return torch.load(path, map_location="cpu", weights_only=True)["embeddings"].float()


def benchmark_label(tasks: list[str]) -> str:
    return ",".join(tasks) if len(tasks) <= 6 else f"{len(tasks)} tasks: {tasks[0]}...{tasks[-1]}"


def load_run(path: Path) -> Run:
    config_path = path / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    return Run(
        path,
        config,
        _read_json(path / "summary.json"),
        _read_json(path / "results.json"),
        _read_json(path / "run.json"),
    )


def discover(paths: list[str | Path]) -> list[Run]:
    """Every directory below ``paths`` that holds a ``summary.json`` (a path may itself be a run directory)."""

    found: dict[Path, Run] = {}
    for root in map(Path, paths):
        candidates = (
            [root] if (root / "summary.json").is_file() else sorted(p.parent for p in root.rglob("summary.json"))
        )
        for directory in candidates:
            run = load_run(directory)
            if run.summary.get("status", "passed") == "passed":
                found[directory.resolve()] = run
    return list(found.values())


def group_runs(runs: list[Run], keys: tuple[str, ...] = ("backbone", "benchmark", "method")) -> dict[tuple, list[Run]]:
    groups: dict[tuple, list[Run]] = defaultdict(list)
    for run in runs:
        groups[tuple(getattr(run, k) for k in keys)].append(run)
    return dict(groups)


def mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    return mean, (math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0)
