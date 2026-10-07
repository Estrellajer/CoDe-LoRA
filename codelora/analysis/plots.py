"""Figures (matplotlib, imported lazily; every figure is saved as PNG and PDF)."""

from __future__ import annotations

from pathlib import Path

from .loader import Run, group_runs, mean_std


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            "plotting needs matplotlib: pip install 'codelora[analysis]'  (tables work without it)"
        ) from None
    return plt


def _save(fig, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", dpi=200)
    _plt().close(fig)
    return paths


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)[:80]


def curves(runs: list[Run], out: Path) -> list[Path]:
    """AP after learning k tasks against k, one figure per (backbone, benchmark), one line per method (mean +- std over seeds).

    Runs evaluated with ``diagonal_final`` have no intermediate AP: they are drawn as the final AP at the last task only.
    """

    plt = _plt()
    written = []
    for (backbone, benchmark), by_method in _by_block(runs).items():
        fig, ax = plt.subplots(figsize=(5, 3.5))
        partial = False
        for method, members in by_method.items():
            full = [r for r in members if r.matrix_kind == "full"]
            if full:
                matrix = [[100 * v for v in r.curve()] for r in full]
                ks = list(range(1, len(matrix[0]) + 1))
                stats = [mean_std([row[k] for row in matrix]) for k in range(len(ks))]
                ax.errorbar(ks, [m for m, _ in stats], yerr=[s for _, s in stats], marker="o", capsize=2, label=method)
            else:
                partial = True
                m, s = mean_std([100 * r.ap for r in members])
                ax.errorbar(
                    [len(members[0].tasks)], [m], yerr=[s], marker="s", capsize=2, label=f"{method} (final only)"
                )
        ax.set_xlabel("number of learned tasks")
        ax.set_ylabel("AP (%)")
        ax.set_title(
            f"{backbone}, {benchmark}" + ("\n(eval.matrix=full gives full curves)" if partial else ""), fontsize=8
        )
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        written += _save(fig, out / f"curves_{_slug(backbone)}_{_slug(benchmark)}")
    return written


def forgetting_numbers(run: Run) -> dict[str, dict[str, float]]:
    """Per task: accuracy right after learning, peak, final and forgetting (peak - final)."""

    after, peak, final = run.diagonal(), run.peak(), run.final()
    return {
        t: {"after": after[t], "peak": peak[t], "final": final[t], "forgetting": peak[t] - final[t]} for t in run.tasks
    }


def forgetting(runs: list[Run], out: Path) -> list[Path]:
    """Per task bars (peak vs final) for the first seed of every method and block, titled with the mean forgetting."""

    plt = _plt()
    written = []
    for (backbone, benchmark), by_method in _by_block(runs).items():
        fig, axes = plt.subplots(1, len(by_method), figsize=(3.6 * len(by_method), 3.2), squeeze=False)
        for ax, (method, members) in zip(axes[0], by_method.items(), strict=True):
            numbers = forgetting_numbers(members[0])
            tasks = list(numbers)
            xs = range(len(tasks))
            ax.bar([x - 0.2 for x in xs], [100 * numbers[t]["peak"] for t in tasks], 0.4, label="peak")
            ax.bar([x + 0.2 for x in xs], [100 * numbers[t]["final"] for t in tasks], 0.4, label="final")
            mean_forgetting = 100 * sum(n["forgetting"] for n in numbers.values()) / len(numbers)
            ax.set_title(f"{method}: forgetting {mean_forgetting:.1f}", fontsize=8)
            ax.set_xticks(list(xs), tasks, rotation=60, ha="right", fontsize=6)
            ax.set_ylabel("score (%)")
            ax.legend(fontsize=6)
        fig.suptitle(f"{backbone}, {benchmark}", fontsize=8)
        written += _save(fig, out / f"forgetting_{_slug(backbone)}_{_slug(benchmark)}")
    return written


def routing_matrix(run: Run, stage: str | None = None) -> tuple[list[str], list[str], list[list[float]]]:
    """Row-normalised confusion: gold task (rows) -> route (columns, ``shared`` last)."""

    counts = run.routing_counts(stage)
    if not counts:
        raise ValueError(f"{run.path}: no routing information (results.json routing or evaluation records with routes)")
    rows = [t for t in run.tasks if t in counts]
    routes = [t for t in run.tasks if any(t in c for c in counts.values())]
    routes += sorted({r for c in counts.values() for r in c} - set(routes) - {"shared"}) + (
        ["shared"] if any("shared" in c for c in counts.values()) else []
    )
    matrix = []
    for task in rows:
        total = sum(counts[task].values()) or 1
        matrix.append([counts[task].get(route, 0) / total for route in routes])
    return rows, routes, matrix


def routing(runs: list[Run], out: Path) -> list[Path]:
    plt = _plt()
    written = []
    for run in runs:
        try:
            rows, cols, matrix = routing_matrix(run)
        except ValueError:
            continue
        fig, ax = plt.subplots(figsize=(0.45 * len(cols) + 2.5, 0.45 * len(rows) + 2))
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=6)
        ax.set_yticks(range(len(rows)), rows, fontsize=6)
        ax.set_xlabel("routed to")
        ax.set_ylabel("gold task")
        ax.set_title(f"{run.method} seed {run.seed}: routing", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.04)
        written += _save(fig, out / f"routing_{_slug(run.method)}_{_slug(run.backbone)}_{run.seed}")
    return written


def fwt_matrix(run: Run) -> tuple[list[str], list[str], list[list[float]]]:
    delta = run.forward_transfer()
    if not delta:
        raise ValueError(f"{run.path}: no forward-transfer matrices (eval.forward_transfer)")
    stages = sorted(delta)
    return stages, run.tasks, [[delta[s].get(t, float("nan")) for t in run.tasks] for s in stages]


def fwt(runs: list[Run], out: Path) -> list[Path]:
    """Heatmap of (score on a not-yet-learned task) - (untrained score): rows stages, columns tasks."""

    plt = _plt()
    written = []
    for run in runs:
        try:
            stages, tasks, matrix = fwt_matrix(run)
        except ValueError:
            continue
        fig, ax = plt.subplots(figsize=(0.5 * len(tasks) + 2.5, 0.4 * len(stages) + 2))
        limit = max((abs(v) for row in matrix for v in row if v == v), default=1.0) or 1.0
        image = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit)
        ax.set_xticks(range(len(tasks)), tasks, rotation=60, ha="right", fontsize=6)
        ax.set_yticks(range(len(stages)), [s.replace("after_task_", "after ") for s in stages], fontsize=6)
        value = run.fwt
        ax.set_title(
            f"{run.method} seed {run.seed}: forward transfer"
            + (f" (FWT {100 * value:.1f})" if value is not None else ""),
            fontsize=8,
        )
        fig.colorbar(image, ax=ax, fraction=0.04)
        written += _save(fig, out / f"fwt_{_slug(run.method)}_{_slug(run.backbone)}_{run.seed}")
    return written


def _by_block(runs: list[Run]) -> dict[tuple, dict[str, list[Run]]]:
    blocks: dict[tuple, dict[str, list[Run]]] = {}
    for (backbone, benchmark, method), members in sorted(group_runs(runs).items()):
        blocks.setdefault((backbone, benchmark), {})[method] = members
    return blocks
