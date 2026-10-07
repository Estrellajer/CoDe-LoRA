"""``python -m codelora.evaluate RUN_DIR [--reload]``: audit a finished run.

Always: recompute AP / BWT from the stored performance matrix and check that every stored cell equals
the mean of its per-example scores.  With ``--reload``: rebuild the model from ``checkpoint.pt``, score
the final row of the matrix again (``--split`` selects test or dev) and report the difference to the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import distributed
from .config import load_config
from .evaluation.metrics import stage_key, summarize_matrix
from .methods import build_method
from .runner import Benchmark, build_context, evaluate_task
from .training.rng import seed_everything


def check_matrix(run_dir: Path) -> dict:
    """Recompute the headline metrics and verify matrix cells against the per-example records."""

    summary = json.loads((run_dir / "summary.json").read_text())
    matrix = summary["performance_matrix"]
    names = list(summary["diagonal_scores"])
    metrics = summarize_matrix(matrix, names)
    worst = 0.0
    for stage, row in matrix.items():
        for task, score in row.items():
            records = json.loads((run_dir / "evaluations" / stage / f"{task}.json").read_text())
            mean = sum(r["score"] for r in records) / len(records)
            worst = max(worst, abs(mean - score))
    return {
        "AP": metrics.average_performance,
        "BWT": metrics.backward_transfer,
        "stored_AP": summary["average_performance"],
        "stored_BWT": summary["backward_transfer"],
        "max_cell_error": worst,
    }


def reload_and_score(run_dir: Path, split: str | None = None) -> dict[str, float]:
    """Final-row scores of the checkpointed model on every task."""

    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    cfg = load_config(run_dir / "config.yaml")
    if split:
        cfg.eval.split = split
    seed_everything(cfg.seed)
    ctx = build_context(cfg)
    method = build_method(ctx)
    method.setup()
    method.load_state_dict(checkpoint["state"])
    bench = ctx.benchmark = Benchmark(cfg)
    return {
        task.name: evaluate_task(method, bench.eval(index), cfg.official_protocol)[0]
        for index, task in enumerate(cfg.tasks, start=1)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--reload", action="store_true", help="re-score the final checkpoint")
    parser.add_argument(
        "--split", choices=["test", "dev"], help="score another split with --reload (default: the run's)"
    )
    args = parser.parse_args()
    distributed.init_from_env()
    report = check_matrix(args.run_dir)
    if args.reload:
        summary = json.loads((args.run_dir / "summary.json").read_text())
        final = summary["performance_matrix"][stage_key(len(summary["diagonal_scores"]))]
        scores = reload_and_score(args.run_dir, args.split)
        report["reloaded"] = scores
        if not args.split:  # a different split is not comparable with the stored scores
            report["reload_max_abs_diff"] = max(abs(scores[t] - final[t]) for t in final)
    if distributed.is_main():
        print(json.dumps(report, indent=2))
    distributed.shutdown()


if __name__ == "__main__":
    main()
