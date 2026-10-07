"""Markdown / LaTeX / CSV tables: multi-seed summary, method comparison, efficiency."""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence

from .loader import Run, group_runs, mean_std


def _order(methods: list[str], preferred: Sequence[str] | None) -> list[str]:
    if not preferred:
        return sorted(methods)
    return [m for m in preferred if m in methods] + sorted(m for m in methods if m not in preferred)


def summary_rows(runs: list[Run], methods: Sequence[str] | None = None) -> list[dict]:
    """One row per (backbone, benchmark, method): mean +- std over seeds of AP / BWT in percent."""

    groups = group_runs(runs)
    rows = []
    blocks = sorted({(b, o) for b, o, _ in groups})
    for backbone, benchmark in blocks:
        for method in _order([m for b, o, m in groups if (b, o) == (backbone, benchmark)], methods):
            members = groups[(backbone, benchmark, method)]
            ap, ap_std = mean_std([100 * r.ap for r in members])
            bwt, bwt_std = mean_std([100 * r.bwt for r in members])
            rows.append({"backbone": backbone, "benchmark": benchmark, "method": method, "seeds": len(members),
                         "AP": ap, "AP_std": ap_std, "BWT": bwt, "BWT_std": bwt_std})  # fmt: skip
    return rows


def _cell(mean: float, std: float, bold: bool, latex: bool, seeds: int) -> str:
    text = f"{mean:.2f}" + (f" $\\pm$ {std:.2f}" if latex else f" ± {std:.2f}") if seeds > 1 else f"{mean:.2f}"
    return (f"\\textbf{{{text}}}" if latex else f"**{text}**") if bold else text


def _best(rows: list[dict], key: str) -> float:
    return max(r[key] for r in rows)


def render(rows: list[dict], fmt: str = "md") -> str:
    """``md`` / ``latex`` / ``csv``; the best AP and BWT of every (backbone, benchmark) block are bold."""

    if fmt == "csv":
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=list(rows[0]) if rows else ["method"])
        writer.writeheader()
        writer.writerows(rows)
        return out.getvalue()
    blocks: dict[tuple, list[dict]] = {}
    for row in rows:
        blocks.setdefault((row["backbone"], row["benchmark"]), []).append(row)
    latex = fmt == "latex"
    lines: list[str] = []
    if latex:
        lines += ["\\begin{tabular}{llcc}", "\\toprule", "Method & Seeds & AP (\\%) & BWT (\\%) \\\\"]
    for (backbone, benchmark), block in blocks.items():
        best_ap, best_bwt = _best(block, "AP"), _best(block, "BWT")
        if latex:
            lines += [
                "\\midrule",
                f"\\multicolumn{{4}}{{l}}{{\\emph{{{backbone}, {benchmark}}}}} \\\\".replace("_", "\\_"),
            ]
        else:
            lines += [
                "",
                f"**{backbone} - {benchmark}**",
                "",
                "| Method | Seeds | AP (%) | BWT (%) |",
                "|---|---|---|---|",
            ]
        for r in block:
            ap = _cell(r["AP"], r["AP_std"], r["AP"] == best_ap, latex, r["seeds"])
            bwt = _cell(r["BWT"], r["BWT_std"], r["BWT"] == best_bwt, latex, r["seeds"])
            lines.append(
                f"{r['method']} & {r['seeds']} & {ap} & {bwt} \\\\"
                if latex
                else f"| {r['method']} | {r['seeds']} | {ap} | {bwt} |"
            )
    if latex:
        lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines).lstrip("\n") + "\n"


def compare_rows(runs: list[Run], reference: str, methods: Sequence[str] | None = None) -> list[dict]:
    rows = summary_rows(runs, methods)
    base = {(r["backbone"], r["benchmark"]): r for r in rows if r["method"] == reference}
    for row in rows:
        ref = base.get((row["backbone"], row["benchmark"]))
        row["dAP"] = row["AP"] - ref["AP"] if ref else float("nan")
        row["dBWT"] = row["BWT"] - ref["BWT"] if ref else float("nan")
    return rows


def render_compare(rows: list[dict], reference: str, fmt: str = "md") -> str:
    if fmt == "csv":
        return render(rows, "csv")
    latex = fmt == "latex"
    sep = " & " if latex else " | "
    head = [
        "Method",
        "AP",
        f"$\\Delta$AP vs {reference}" if latex else f"ΔAP vs {reference}",
        "BWT",
        "$\\Delta$BWT" if latex else "ΔBWT",
    ]
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule", sep.join(head) + " \\\\"] if latex else []
    last = None
    for r in rows:
        block = (r["backbone"], r["benchmark"])
        if block != last:
            title = f"{r['backbone']}, {r['benchmark']}"
            if latex:
                lines += ["\\midrule", f"\\multicolumn{{5}}{{l}}{{\\emph{{{title}}}}} \\\\".replace("_", "\\_")]
            else:
                lines += ["", f"**{title}**", "", "| " + " | ".join(head) + " |", "|---|---|---|---|---|"]
            last = block
        cells = [r["method"], f"{r['AP']:.2f}", f"{r['dAP']:+.2f}", f"{r['BWT']:.2f}", f"{r['dBWT']:+.2f}"]
        lines.append(sep.join(cells) + " \\\\" if latex else "| " + " | ".join(cells) + " |")
    if latex:
        lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines).lstrip("\n") + "\n"


EFFICIENCY_KEYS = (
    "train_seconds",
    "eval_seconds",
    "total_seconds",
    "peak_memory_mib",
    "adapter_parameters",
    "adapter_fraction",
    "trainable_parameters",
)


def efficiency_rows(runs: list[Run], methods: Sequence[str] | None = None) -> list[dict]:
    """One row per (backbone, benchmark, method): mean (``key``) and std (``key_std``) over seeds."""

    groups = group_runs(runs)
    rows = []
    for backbone, benchmark in sorted({(b, o) for b, o, _ in groups}):
        for method in _order([m for b, o, m in groups if (b, o) == (backbone, benchmark)], methods):
            members = [r.efficiency() for r in groups[(backbone, benchmark, method)]]
            row = {"backbone": backbone, "benchmark": benchmark, "method": method, "seeds": len(members)}
            for key in EFFICIENCY_KEYS:
                values = [m[key] for m in members if key in m]
                row[key], row[f"{key}_std"] = mean_std(values) if values else (float("nan"), float("nan"))
            rows.append(row)
    return rows


def render_efficiency(rows: list[dict], fmt: str = "md") -> str:
    if fmt == "csv":
        return render(rows, "csv")
    latex = fmt == "latex"
    columns = [  # (header, key, scale, decimals)
        ("Train (min)", "train_seconds", 60, 1),
        ("Eval (min)", "eval_seconds", 60, 1),
        ("Total (min)", "total_seconds", 60, 1),
        ("Peak mem (GiB)", "peak_memory_mib", 1024, 1),
        ("Adapter params (M)", "adapter_parameters", 1e6, 2),
        ("Adapter (% of base)", "adapter_fraction", 0.01, 2),
    ]
    head = ["Method", *(c[0] for c in columns)]
    sep = " & " if latex else " | "
    pm = " $\\pm$ " if latex else " ± "
    lines = ["\\begin{tabular}{l" + "r" * len(columns) + "}", "\\toprule", sep.join(head) + " \\\\"] if latex else []
    last = None
    for r in rows:
        block = (r["backbone"], r["benchmark"])
        if block != last:
            title = f"{r['backbone']}, {r['benchmark']}"
            if latex:
                lines += [
                    "\\midrule",
                    f"\\multicolumn{{{len(head)}}}{{l}}{{\\emph{{{title}}}}} \\\\".replace("_", "\\_"),
                ]
            else:
                lines += ["", f"**{title}**", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            last = block
        cells = [r["method"]]
        for _, key, scale, decimals in columns:
            text = f"{r[key] / scale:.{decimals}f}"
            if r["seeds"] > 1 and r[f"{key}_std"] == r[f"{key}_std"]:
                text += f"{pm}{r[f'{key}_std'] / scale:.{decimals}f}"
            cells.append(text)
        lines.append(sep.join(cells) + " \\\\" if latex else "| " + " | ".join(cells) + " |")
    if latex:
        lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines).lstrip("\n") + "\n"
