"""``codelora.analysis`` on synthetic run directories laid out exactly as docs/RESULTS.md describes."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import pytest
import yaml

from codelora.analysis import cli, plots, tables
from codelora.analysis.loader import discover, group_runs

TASKS = ["alpha", "beta", "gamma"]
BASE = {"code-lora": 0.9, "lora": 0.6}
DROP = {"code-lora": 0.01, "lora": 0.1}  # score lost per stage after a task was learned


def score(method: str, seed: int, k: int, i: int) -> float:
    return BASE[method] + 0.01 * (seed - 41) - DROP[method] * (k - i) - 0.02 * i


def make_run(
    root: Path,
    method: str,
    seed: int,
    *,
    full: bool = True,
    fwt: bool = False,
) -> Path:
    run = root / method / f"s{seed}"
    run.mkdir(parents=True)
    n = len(TASKS)
    matrix = {
        f"after_task_{k:03d}": {
            TASKS[i - 1]: score(method, seed, k, i) for i in range(1, k + 1) if full or i == k or k == n
        }
        for k in range(1, n + 1)
    }
    final = matrix[f"after_task_{n:03d}"]
    diagonal = {TASKS[i - 1]: matrix[f"after_task_{i:03d}"][TASKS[i - 1]] for i in range(1, n + 1)}
    summary = {
        "schema_version": 1, "status": "passed", "method": method, "seed": seed, "world_size": 1,
        "tasks": [{"task": t, "train_seconds": 10.0 * (j + 1)} for j, t in enumerate(TASKS)],
        "performance_matrix": matrix, "average_performance": sum(final.values()) / n,
        "backward_transfer": sum(final[t] - diagonal[t] for t in TASKS[:-1]) / (n - 1), "diagonal_scores": diagonal,
        "evaluation_seconds": dict.fromkeys(matrix, 2.0), "peak_allocated_mib": 2048.0,
        "efficiency": {"train_seconds": 60.0, "eval_seconds": 6.0, "peak_memory_mib": 2048.0,
                       "adapter_parameters": 4e6 if method == "lora" else 2e7, "trainable_parameters": 2e6},
        "FWT": None,
    }  # fmt: skip
    if fwt:
        summary["forward_transfer_delta_matrix"] = {
            f"after_task_{k:03d}": {TASKS[j - 1]: 0.01 * (j - k) for j in range(k + 1, n + 1)} for k in range(1, n + 1)
        }
    (run / "summary.json").write_text(json.dumps(summary))
    config = {"method": method, "seed": seed, "backbone": {"name": "tiny"}, "tasks": [{"name": t, "path": t} for t in TASKS],
              "router": {"kind": "lda"}}  # fmt: skip
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    if method == "code-lora":
        routing = {
            f"after_task_{n:03d}": {
                t: {"counts": {t: 90, TASKS[(j + 1) % n]: 6, "shared": 4}, "mean_confidence": 0.9}
                for j, t in enumerate(TASKS)
            }
        }
        (run / "results.json").write_text(json.dumps({"schema_version": 1, "routing": routing, "performance": matrix}))
    return run


@pytest.fixture
def runs_dir(tmp_path):
    for method in BASE:
        for seed in (41, 42, 43):
            make_run(tmp_path, method, seed, fwt=method == "code-lora")
    return tmp_path


def test_discovery_and_grouping(runs_dir):
    runs = discover([runs_dir])
    assert len(runs) == 6
    groups = group_runs(runs)
    assert sorted(groups) == [("tiny", "alpha,beta,gamma", "code-lora"), ("tiny", "alpha,beta,gamma", "lora")]
    assert all(len(v) == 3 for v in groups.values())
    assert [r.path.name for r in discover([runs_dir / "lora" / "s41"])] == ["s41"]  # a run directory itself


def test_summary_table_numbers_and_formats(runs_dir):
    runs = discover([runs_dir])
    rows = tables.summary_rows(runs, methods=["lora", "code-lora"])
    assert [r["method"] for r in rows] == ["lora", "code-lora"]
    expected = [100 * r.ap for r in runs if r.method == "code-lora"]
    code = rows[1]
    assert code["AP"] == pytest.approx(statistics.mean(expected)) and code["AP_std"] == pytest.approx(
        statistics.stdev(expected)
    )
    assert code["seeds"] == 3
    md = tables.render(rows, "md")
    assert "| code-lora | 3 | **" in md and "| lora | 3 | " in md and "±" in md  # best per column in bold
    latex = tables.render(rows, "latex")
    assert "\\toprule" in latex and "\\bottomrule" in latex and "\\textbf{" in latex and "$\\pm$" in latex
    csv_text = tables.render(rows, "csv")
    assert csv_text.splitlines()[0].startswith("backbone,benchmark,method,seeds,AP")


def test_compare_deltas_and_efficiency(runs_dir):
    runs = discover([runs_dir])
    rows = tables.compare_rows(runs, "code-lora")
    by = {r["method"]: r for r in rows}
    assert by["code-lora"]["dAP"] == 0 and by["lora"]["dAP"] == pytest.approx(by["lora"]["AP"] - by["code-lora"]["AP"])
    assert "ΔAP vs code-lora" in tables.render_compare(rows, "code-lora")
    assert "\\Delta$AP" in tables.render_compare(rows, "code-lora", "latex")
    eff = {r["method"]: r for r in tables.efficiency_rows(runs)}
    assert eff["lora"]["adapter_parameters"] == 4e6 and eff["code-lora"]["train_seconds"] == 60.0
    assert "Peak mem" in tables.render_efficiency(list(eff.values()))


def test_curves_and_forgetting_numbers(runs_dir):
    run = next(r for r in discover([runs_dir]) if r.method == "lora" and r.seed == 41)
    assert run.matrix_kind == "full"
    curve = run.curve()
    assert curve[0] == pytest.approx(score("lora", 41, 1, 1))
    assert curve[2] == pytest.approx(sum(score("lora", 41, 3, i) for i in (1, 2, 3)) / 3)
    numbers = plots.forgetting_numbers(run)
    assert numbers["alpha"]["after"] == pytest.approx(score("lora", 41, 1, 1))
    assert numbers["alpha"]["final"] == pytest.approx(score("lora", 41, 3, 1))
    assert numbers["alpha"]["forgetting"] == pytest.approx(numbers["alpha"]["peak"] - numbers["alpha"]["final"])
    assert numbers["gamma"]["forgetting"] == 0


def test_diagonal_final_runs_have_no_curve(tmp_path):
    run = discover([make_run(tmp_path, "lora", 41, full=False)])[0]
    assert run.matrix_kind == "diagonal_final"
    with pytest.raises(ValueError, match=r"eval\.matrix=full"):
        run.curve()
    assert set(run.peak()) == set(TASKS)  # the diagonal stands in for the peak


def test_routing_matrix_rows_are_normalised(runs_dir):
    run = next(r for r in discover([runs_dir]) if r.method == "code-lora")
    rows, cols, matrix = plots.routing_matrix(run)
    assert rows == TASKS and cols[-1] == "shared" and set(cols) == set(TASKS) | {"shared"}
    assert all(math.isclose(sum(row), 1.0) for row in matrix)
    assert matrix[0][cols.index("alpha")] == pytest.approx(0.9)
    lora = next(r for r in discover([runs_dir]) if r.method == "lora")
    with pytest.raises(ValueError, match="no routing"):
        plots.routing_matrix(lora)


def test_routing_counts_fall_back_to_evaluation_records(tmp_path):
    run_dir = make_run(tmp_path, "lora", 41)
    stage = run_dir / "evaluations" / "after_task_003"
    stage.mkdir(parents=True)
    (stage / "alpha.json").write_text(json.dumps([{"route": "alpha"}, {"route": "alpha"}, {"route": "shared"}]))
    run = discover([run_dir])[0]
    assert run.routing_counts() == {"alpha": {"alpha": 2, "shared": 1}}


def test_forward_transfer_matrix_and_fwt(runs_dir):
    run = next(r for r in discover([runs_dir]) if r.method == "code-lora")
    stages, tasks, matrix = plots.fwt_matrix(run)
    assert tasks == TASKS and len(stages) == 3 and matrix[0][1] == pytest.approx(0.01) and math.isnan(matrix[2][0])
    # FWT: gain of task i over the untrained model right before it is learned (stage i-1), i >= 2
    assert run.fwt == pytest.approx((0.01 * (2 - 1) + 0.01 * (3 - 2)) / 2)


def has_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not has_matplotlib(), reason="figures need matplotlib")
def test_figures_are_written(runs_dir, tmp_path):
    runs = discover([runs_dir])
    out = tmp_path / "figs"
    written = plots.curves(runs, out) + plots.forgetting(runs, out) + plots.routing(runs, out) + plots.fwt(runs, out)
    assert written and all(p.is_file() and p.stat().st_size > 0 for p in written)
    assert {p.suffix for p in written} == {".png", ".pdf"}
    assert any(p.name.startswith("routing_code-lora") for p in written) and not any(
        p.name.startswith("routing_lora") for p in written
    )


@pytest.mark.skipif(not has_matplotlib(), reason="figures need matplotlib")
def test_cli_all_writes_tables_and_figures(runs_dir, tmp_path, capsys):
    out = tmp_path / "all"
    cli.main(["all", str(runs_dir), "--out", str(out)])
    names = {p.name for p in out.iterdir()}
    assert {"summary.md", "compare.md", "efficiency.md"} <= names
    assert any(n.startswith("curves_") and n.endswith(".pdf") for n in names)
    assert "AP (%)" in (out / "summary.md").read_text()


def test_cli_tables_without_matplotlib_dependencies(runs_dir, tmp_path):
    out = tmp_path / "t"
    cli.main(["table", str(runs_dir), "--out", str(out), "--format", "latex", "--methods", "lora", "code-lora"])
    assert "\\begin{tabular}" in (out / "summary.tex").read_text()
    with pytest.raises(SystemExit, match="no finished run"):
        cli.main(["table", str(tmp_path / "nothing"), "--out", str(out)])


def test_plot_commands_explain_a_missing_matplotlib(runs_dir, tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(SystemExit, match="pip install"):
        cli.main(["curves", str(runs_dir), "--out", str(tmp_path / "x")])
