"""``python -m codelora.analysis COMMAND RUN_OR_DIR... --out DIR``."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import plots, tables
from .loader import Run, discover

COMMANDS = ["curves", "forgetting", "routing", "fwt", "efficiency", "table", "compare", "all"]
SUFFIX = {"md": ".md", "latex": ".tex", "csv": ".csv"}


def _emit(text: str, out: Path, name: str, fmt: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}{SUFFIX[fmt]}"
    path.write_text(text, encoding="utf-8")
    print(text)
    return path


def run_command(command: str, runs: list[Run], args: argparse.Namespace) -> list[Path]:
    out, fmt = Path(args.out), args.format
    if command == "table":
        return [_emit(tables.render(tables.summary_rows(runs, args.methods), fmt), out, "summary", fmt)]
    if command == "compare":
        rows = tables.compare_rows(runs, args.reference, args.methods)
        return [_emit(tables.render_compare(rows, args.reference, fmt), out, "compare", fmt)]
    if command == "efficiency":
        return [
            _emit(tables.render_efficiency(tables.efficiency_rows(runs, args.methods), fmt), out, "efficiency", fmt)
        ]
    if command == "curves":
        return plots.curves(runs, out)
    if command == "forgetting":
        return plots.forgetting(runs, out)
    if command == "routing":
        return plots.routing(runs, out)
    if command == "fwt":
        return plots.fwt(runs, out)
    raise SystemExit(f"unknown command {command!r}")


def run_all(runs: list[Run], args: argparse.Namespace) -> list[Path]:
    written: list[Path] = []
    for command in ("table", "compare", "efficiency"):
        written += run_command(command, runs, args)
    for command in ("curves", "forgetting", "routing", "fwt"):
        try:
            written += run_command(command, runs, args)
        except (
            ValueError,
            SystemExit,
        ) as error:  # a command whose inputs this directory lacks is skipped, with the reason
            print(f"[{command}] skipped: {error}")
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m codelora.analysis", description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("paths", nargs="+", help="run directories or directories searched recursively for summary.json")
    parser.add_argument("--out", default="analysis", help="output directory")
    parser.add_argument("--format", choices=sorted(SUFFIX), default="md", help="table format")
    parser.add_argument("--methods", nargs="+", help="method order of the tables")
    parser.add_argument("--reference", default="code-lora", help="reference method of `compare`")
    args = parser.parse_args(argv)
    runs = discover(args.paths)
    if not runs:
        raise SystemExit("no finished run (summary.json) found")
    print(f"{len(runs)} runs")
    written = run_all(runs, args) if args.command == "all" else run_command(args.command, runs, args)
    print("wrote:\n" + "\n".join(f"  {p}" for p in written))
