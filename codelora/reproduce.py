"""Expand a paper table into runs, launch them, and collect the results.

``python -m codelora.reproduce table1``                    print the commands (dry run)
``python -m codelora.reproduce table1 --run --gpus 8``     run them one after another (finished runs are skipped)
``python -m codelora.reproduce table1 --generation``      the same with CoDe-LoRA generating (no PMI fusion)
``python -m codelora.reproduce table1 --summarize``        AP / BWT as mean +- std over seeds -> results.{md,csv,json}

Tables: ``table1`` (T5-large, Standard CL order1-3 and Long order4-6) and ``table2`` (Qwen3-0.6B, Llama-2-7B,
Qwen3.5-4B on Standard, Long and TRACE).  Every run takes the shipped settings of its protocol and method fragments
(1 epoch per task, effective batch 64, bf16, per-backbone / per-method lr); CoDe-LoRA's PMI fusion on closed-label
tasks is part of ``methods/code-lora`` (TRACE has none).
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

METHODS = ["lora", "o-lora", "n-lora", "co-lora", "de-lora", "code-lora", "mole-cie"]
STANDARD = ["standard-order1", "standard-order2", "standard-order3"]
LONG = ["long-order4", "long-order5", "long-order6"]
DECODERS = ["qwen3-0.6b", "llama2-7b", "qwen3.5-4b"]


@dataclass(frozen=True)
class Run:
    table: str
    backbone: str
    benchmark: str
    method: str
    seed: int
    fuse: bool = True  # False: CoDe-LoRA generates (fragment evaluation/no-fusion) instead of fusing label scores

    @property
    def name(self) -> str:
        return f"{self.table}/{self.backbone}/{self.benchmark}/{self.method}/s{self.seed}"

    def fragments(self) -> list[str]:
        family = self.benchmark.split("-")[0]
        fragments = [f"orders/{self.benchmark}", f"protocols/{family}/{self.backbone}", f"methods/{self.method}"]
        return fragments if self.fuse or self.method != "code-lora" else [*fragments, "evaluation/no-fusion"]

    def overrides(self, output_dir: Path) -> list[str]:
        return [f"seed={self.seed}", f"name={self.name}", f"output_dir={output_dir}"]

    def command(self, output_dir: Path, gpus: int) -> list[str]:
        launcher = ["python"] if gpus == 1 else ["torchrun", f"--nproc_per_node={gpus}"]
        module = ["-m", "codelora.train"]
        return [*launcher, *module, *self.fragments(), *self.overrides(output_dir)]

    def summary_path(self, output_dir: Path) -> Path:
        return output_dir / self.name / "summary.json"


def plan(
    table: str,
    methods: list[str] | None,
    benchmarks: list[str] | None,
    backbones: list[str] | None,
    seeds: list[int],
    fuse: bool = True,
) -> list[Run]:
    runs: list[Run] = []
    if table == "table1":
        for benchmark in benchmarks or STANDARD + LONG:
            for method in methods or METHODS:
                runs += [Run(table, "t5-large", benchmark, method, seed, fuse) for seed in seeds]
    elif table == "table2":
        for backbone in backbones or DECODERS:
            for benchmark in benchmarks or [*STANDARD, *LONG, "trace"]:
                for method in methods or ["o-lora", "n-lora", "mole-cie", "code-lora"]:
                    runs += [Run(table, backbone, benchmark, method, seed, fuse) for seed in seeds]
    else:
        raise SystemExit(f"unknown table {table!r}")
    return runs


def summarize(runs: list[Run], output_dir: Path) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for run in runs:
        path = run.summary_path(output_dir)
        if path.is_file():
            groups[(run.backbone, run.benchmark, run.method)].append(json.loads(path.read_text()))
    rows = []
    for (backbone, benchmark, method), summaries in groups.items():
        ap = [100 * s["average_performance"] for s in summaries]
        bwt = [100 * s["backward_transfer"] for s in summaries]
        spread = lambda xs: statistics.stdev(xs) if len(xs) > 1 else 0.0  # noqa: E731
        rows.append(
            {
                "backbone": backbone, "benchmark": benchmark, "method": method, "seeds": len(summaries),
                "AP": statistics.mean(ap), "AP_std": spread(ap), "BWT": statistics.mean(bwt), "BWT_std": spread(bwt),
            }
        )  # fmt: skip
    return rows


def write_results(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(rows, indent=2))
    with (output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["backbone"])
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| backbone | benchmark | method | seeds | AP | BWT |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['backbone']} | {r['benchmark']} | {r['method']} | {r['seeds']} | "
            f"{r['AP']:.2f} ± {r['AP_std']:.2f} | {r['BWT']:.2f} ± {r['BWT_std']:.2f} |"
        )
    (output_dir / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("table", choices=["table1", "table2"])
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--benchmarks", nargs="+", help="e.g. standard-order1 long-order4 trace")
    parser.add_argument("--backbones", nargs="+", help="table2 only: qwen3-0.6b llama2-7b qwen3.5-4b")
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--generation",
        action="store_true",
        help="CoDe-LoRA generates instead of fusing candidate-label scores (adds the fragment evaluation/no-fusion)",
    )
    parser.add_argument("--gpus", type=int, default=1, help="processes per run (torchrun when > 1)")
    parser.add_argument("--run", action="store_true", help="execute the runs (default: print the commands)")
    parser.add_argument("--summarize", action="store_true", help="collect finished runs into results.{md,csv,json}")
    args = parser.parse_args()

    runs = plan(args.table, args.methods, args.benchmarks, args.backbones, args.seeds, not args.generation)
    if args.summarize:
        write_results(summarize(runs, args.output_dir), args.output_dir / args.table.replace("/", "_"))
        return
    for run in runs:
        command = run.command(args.output_dir, args.gpus)
        if not args.run:
            print(shlex.join(command))
        elif run.summary_path(args.output_dir).is_file():
            print(f"skip {run.name} (finished)")
        else:
            print(f"run  {run.name}", flush=True)
            subprocess.run([sys.executable if command[0] == "python" else command[0], *command[1:]], check=True)


if __name__ == "__main__":
    main()
