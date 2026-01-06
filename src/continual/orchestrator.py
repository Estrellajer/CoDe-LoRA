"""Continual-learning orchestrator.

Parses a YAML/JSON config, builds per-task RunPaths, and delegates to
ModelFactory implementations. This is a lightweight wiring layer; concrete
models implement training/eval logic.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    yaml = None

from .base import EvalPlan, RunPaths, TaskSpec
from .factory import ModelFactory
from .utils import append_jsonl, gpu_snapshot

# Import models to register them with the factory
from . import models  # noqa: F401


def _get_rank() -> int:
    """Best-effort distributed rank detection for Deepspeed/Torch/SLURM."""
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        val = os.environ.get(key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                return 0
    return 0


def _is_main_process() -> bool:
    return _get_rank() == 0


def _configure_runtime_logging() -> None:
    """Reduce noisy logs (datasets progress bars / warnings) in multi-rank runs."""
    try:
        import datasets

        # Avoid repeated "Using custom data configuration ..." and tqdm bars
        datasets.utils.logging.disable_progress_bar()
        datasets.utils.logging.set_verbosity_error()
    except Exception:
        pass


def _load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yml", ".yaml")):
            if yaml is None:
                raise ImportError("PyYAML is required for YAML configs")
            return yaml.safe_load(f)
        return json.load(f)


def _build_task_specs(cfg: Dict[str, Any]) -> List[TaskSpec]:
    tasks_cfg = cfg.get("tasks", [])
    specs: List[TaskSpec] = []
    for task in tasks_cfg:
        # Allow per-task overrides without forcing everything into `extra`.
        # Preferred schema: task.extra.train_overrides
        # Also supported: task.train_overrides (shorthand)
        extra = dict(task.get("extra", {}) or {})
        if "train_overrides" in task and "train_overrides" not in extra:
            extra["train_overrides"] = task.get("train_overrides")

        specs.append(
            TaskSpec(
                name=task["name"],
                dataset_path=task["dataset_path"],
                train_split=task.get("train_split", "train"),
                eval_split=task.get("eval_split", "validation"),
                test_split=task.get("test_split", "test"),
                metrics=task.get("metrics", []),
                num_epochs=task.get("num_epochs"),
                max_steps=task.get("max_steps"),
                extra=extra,
            )
        )
    return specs


def _build_eval_plan(cfg: Dict[str, Any]) -> EvalPlan:
    eval_cfg = cfg.get("eval", {})
    return EvalPlan(
        strategy=eval_cfg.get("strategy", "learned"),
        custom_tasks=eval_cfg.get("custom_tasks"),
        metrics_map=eval_cfg.get("metrics_map", {}),
    )


def _eval_targets(eval_plan: EvalPlan, tasks_so_far: List[str], all_tasks: List[str]) -> List[str]:
    if eval_plan.strategy == "learned":
        return list(tasks_so_far)
    if eval_plan.strategy == "all":
        return list(all_tasks)
    if eval_plan.strategy == "custom" and eval_plan.custom_tasks:
        return list(eval_plan.custom_tasks)
    return list(tasks_so_far)


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H%M%S")

def _reset_torch_peak_memory_stats() -> None:
    """Reset per-process peak CUDA memory stats (best-effort)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return
        for d in range(int(torch.cuda.device_count())):
            try:
                torch.cuda.reset_peak_memory_stats(d)
            except Exception:
                continue
    except Exception:
        return


def _dist_barrier() -> None:
    """Synchronize distributed ranks (best-effort).

    This prevents races where rank>0 tries to reload a checkpoint directory
    before rank-0 finishes writing adapter_config.json / weights.
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        return


def _rank_str() -> str:
    return str(os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_PROCID") or "0")


def _format_gpu_brief(snapshot: Dict[str, Any], show_peak: bool = False) -> str:
    """Human-friendly one-liner for console logs (best-effort)."""
    try:
        smi = snapshot.get("nvidia_smi") or {}
        if isinstance(smi, dict) and smi.get("ok") and smi.get("gpus"):
            parts = []
            for g in smi["gpus"]:
                idx = g.get("index")
                used = g.get("mem_used_mib")
                total = g.get("mem_total_mib")
                util = g.get("util_gpu_pct")
                if idx is None:
                    continue
                if used is not None and total is not None:
                    parts.append(f"gpu{idx} {used}/{total}MiB util={util}%")
                else:
                    parts.append(f"gpu{idx} util={util}%")
            if parts:
                return "; ".join(parts)
    except Exception:
        pass

    # Fallback: torch-only stats (per-process)
    try:
        tc = snapshot.get("torch_cuda") or {}
        devs = tc.get("devices") or []
        parts = []
        for d in devs:
            idx = d.get("index")
            reserved = d.get("torch_reserved_mib")
            allocated = d.get("torch_allocated_mib")
            max_reserved = d.get("torch_max_reserved_mib")
            max_allocated = d.get("torch_max_allocated_mib")
            total = d.get("mem_total_mib")
            if idx is None:
                continue
            if show_peak and max_reserved is not None:
                parts.append(f"cuda{idx} cur={reserved}MiB peak={max_reserved}MiB total={total}MiB")
            elif reserved is not None:
                parts.append(f"cuda{idx} reserved={reserved}MiB total={total}MiB")
            else:
                parts.append(f"cuda{idx} alloc={allocated}MiB total={total}MiB")
        if parts:
            return "; ".join(parts)
    except Exception:
        pass
    return ""


def _extract_peak_memory(snapshot: Dict[str, Any]) -> Dict[str, float]:
    """Extract peak GPU memory stats from a snapshot for easy comparison."""
    result = {}
    try:
        tc = snapshot.get("torch_cuda") or {}
        devs = tc.get("devices") or []
        for d in devs:
            idx = d.get("index", 0)
            max_reserved = d.get("torch_max_reserved_mib")
            max_allocated = d.get("torch_max_allocated_mib")
            if max_reserved is not None:
                result[f"peak_reserved_mib_gpu{idx}"] = max_reserved
            if max_allocated is not None:
                result[f"peak_allocated_mib_gpu{idx}"] = max_allocated
    except Exception:
        pass
    return result


def _print_gpu_summary(artifacts, task_name: str, task_idx: int) -> None:
    """Print a standardized GPU memory summary for easy comparison across methods."""
    try:
        rs = artifacts.resource_stats or {}
        gpu = rs.get("gpu", {})
        
        train_peak = gpu.get("train_peak", {})
        
        # Find peak values across all GPUs
        peak_reserved = max(
            (v for k, v in train_peak.items() if "peak_reserved" in k and v is not None),
            default=None
        )
        peak_allocated = max(
            (v for k, v in train_peak.items() if "peak_allocated" in k and v is not None),
            default=None
        )
        
        if peak_reserved is not None or peak_allocated is not None:
            print(f"[GPU-Summary] task={task_name} idx={task_idx} ", end="")
            if peak_reserved is not None:
                print(f"peak_reserved={peak_reserved:.1f}MiB ", end="")
            if peak_allocated is not None:
                print(f"peak_allocated={peak_allocated:.1f}MiB", end="")
            print()
    except Exception:
        pass


def _record_gpu_snapshot(paths: RunPaths, stage: str, task: TaskSpec, idx: int, include_nvidia_smi: bool) -> Dict[str, Any]:
    """Record GPU usage into JSONL files under the task directory."""
    snap = gpu_snapshot(include_nvidia_smi=include_nvidia_smi)
    rec = {
        "stage": stage,
        "task": task.name,
        "idx": idx,
        "rank": _rank_str(),
        "snapshot": snap,
    }
    # One file per rank to avoid write contention and to keep per-rank peaks.
    out_path = os.path.join(paths.task_dir, f"gpu_usage_rank{_rank_str()}.jsonl")
    append_jsonl(out_path, rec)

    # Also write a single consolidated file for quick grepping (rank-0 only).
    if include_nvidia_smi:
        append_jsonl(os.path.join(paths.task_dir, "gpu_usage.jsonl"), rec)
    return snap


def run_experiment(config_path: str) -> None:
    cfg = _load_config(config_path)
    _configure_runtime_logging()

    method = cfg.get("method", "baseline")
    task_sequence = cfg.get("task_sequence", "order1")
    # YAML uses top-level `run_root` in this repo's configs; keep backward
    # compatibility with `paths.run_root` if present.
    run_root = cfg.get("run_root") or cfg.get("paths", {}).get("run_root") or "runs"
    model_key = cfg.get("model_name") or cfg.get("method")
    tasks = _build_task_specs(cfg)
    eval_plan = _build_eval_plan(cfg)
    is_main = _is_main_process()

    model = ModelFactory.create(
        model_key,
        model_cfg=cfg.get("model", {}),
        data_cfg=cfg.get("data", {}),
        train_cfg=cfg.get("train", {}),
    )
    model.initialize_or_load(cfg.get("model", {}).get("load_checkpoint"))

    run_stamp = _timestamp()
    all_task_names = [t.name for t in tasks]
    learned: List[str] = []
    # Collect GPU peak memory per task for final summary
    all_gpu_peaks: List[Dict[str, Any]] = []

    # Resolve per-task metrics from eval plan once (keeps YAML tidy).
    for t in tasks:
        t.metrics = model.metrics_for_task(t, eval_plan)

    for idx, task in enumerate(tasks, start=1):
        learned.append(task.name)
        paths = RunPaths(
            run_root=run_root,
            method=method,
            task_sequence=task_sequence,
            task_idx=idx,
            task_name=task.name,
            timestamp=run_stamp,
        )

        # Placeholder dataloaders; integrate real builders here.
        dataloaders: Dict[str, Any] = {}

        # -----------------------------
        # GPU stats (single-model runs)
        # -----------------------------
        # Reset peak stats so per-task "max_*" corresponds to this task (per process).
        _reset_torch_peak_memory_stats()
        gpu_before_train = _record_gpu_snapshot(paths, "before_train", task, idx, include_nvidia_smi=is_main)
        if is_main:
            brief = _format_gpu_brief(gpu_before_train)
            if brief:
                print(f"[GPU] before_train: {brief}")

        artifacts = model.incremental_train(task, dataloaders, paths)
        model.save_artifacts(artifacts, paths)

        gpu_after_train = _record_gpu_snapshot(paths, "after_train", task, idx, include_nvidia_smi=is_main)
        if is_main:
            # Show both current and peak memory after training
            brief = _format_gpu_brief(gpu_after_train, show_peak=True)
            if brief:
                print(f"[GPU] after_train: {brief}")

        # Store a compact snapshot into artifacts (rank-0 only) so it ends up in task_summary.json
        if is_main:
            artifacts.resource_stats = artifacts.resource_stats or {}
            artifacts.resource_stats.setdefault("gpu", {})
            artifacts.resource_stats["gpu"]["before_train"] = gpu_before_train
            artifacts.resource_stats["gpu"]["after_train"] = gpu_after_train
            # Extract peak memory for easy comparison across methods
            artifacts.resource_stats["gpu"]["train_peak"] = _extract_peak_memory(gpu_after_train)

        # Reload the model from the newly saved checkpoint to ensure evaluation 
        # and the next task start from a clean state (especially for DeepSpeed ZeRO).
        # NOTE: In multi-rank runs, only rank-0 typically writes checkpoints.
        # Add a barrier so other ranks don't attempt to reload before files exist.
        _dist_barrier()
        if artifacts.checkpoint_path:
            if is_main:
                print(f"Reloading model from {artifacts.checkpoint_path}...")
            model.initialize_or_load(artifacts.checkpoint_path)
            _dist_barrier()
            gpu_after_reload = _record_gpu_snapshot(paths, "after_reload", task, idx, include_nvidia_smi=is_main)
            if is_main:
                brief = _format_gpu_brief(gpu_after_reload)
                if brief:
                    print(f"[GPU] after_reload: {brief}")
                artifacts.resource_stats["gpu"]["after_reload"] = gpu_after_reload

        targets = _eval_targets(eval_plan, learned, all_task_names)
        
        # Perform evaluation on all target tasks
        gpu_before_eval = _record_gpu_snapshot(paths, "before_eval", task, idx, include_nvidia_smi=is_main)
        if is_main:
            brief = _format_gpu_brief(gpu_before_eval)
            if brief:
                print(f"[GPU] before_eval: {brief}")
            artifacts.resource_stats["gpu"]["before_eval"] = gpu_before_eval

        all_eval_metrics = {}
        for target_name in targets:
            # Find the TaskSpec for the target
            target_task = next(t for t in tasks if t.name == target_name)
            if is_main:
                print(f"Evaluating on {target_name}...")
            eval_metrics = model.evaluate(target_task, paths, split="test")
            all_eval_metrics.update(eval_metrics)

        gpu_after_eval = _record_gpu_snapshot(paths, "after_eval", task, idx, include_nvidia_smi=is_main)
        if is_main:
            brief = _format_gpu_brief(gpu_after_eval)
            if brief:
                print(f"[GPU] after_eval: {brief}")
            artifacts.resource_stats["gpu"]["after_eval"] = gpu_after_eval

        # Optional: log summary per task
        summary = {
            "task": task.name,
            "idx": idx,
            "run_dir": paths.task_dir,
            "artifacts": asdict(artifacts),
            "eval_targets": targets,
            "eval_metrics": all_eval_metrics,
        }
        
        # Save summary to a file in the task directory
        if is_main:
            summary_path = os.path.join(paths.task_dir, "task_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            # Keep console output compact: only show eval metrics + runtimes.
            compact = {k: v for k, v in all_eval_metrics.items() if k.endswith(("runtime", "exact_match", "rouge1", "rougeL", "loss"))}
            print(f"[Summary] task={task.name} idx={idx} run_dir={paths.task_dir}")
            print(json.dumps(compact, indent=2))

            # Print GPU peak memory summary for method comparison
            _print_gpu_summary(artifacts, task.name, idx)
            
            # Collect for final summary
            train_peak = artifacts.resource_stats.get("gpu", {}).get("train_peak", {})
            all_gpu_peaks.append({
                "task": task.name,
                "idx": idx,
                "train_peak": train_peak,
            })

    # Print final GPU memory comparison summary
    if is_main and all_gpu_peaks:
        _print_final_gpu_summary(method, all_gpu_peaks, run_root, task_sequence, run_stamp)


def _print_final_gpu_summary(
    method: str,
    all_gpu_peaks: List[Dict[str, Any]],
    run_root: str,
    task_sequence: str,
    run_stamp: str,
) -> None:
    """Print a final summary of GPU memory usage across all tasks for method comparison."""
    print("\n" + "=" * 70)
    print(f"[GPU-Final] method={method} sequence={task_sequence}")
    print("=" * 70)
    
    max_peak_reserved = 0.0
    max_peak_allocated = 0.0
    total_peak_reserved = 0.0
    
    for entry in all_gpu_peaks:
        task = entry.get("task", "?")
        idx = entry.get("idx", 0)
        train_peak = entry.get("train_peak", {})
        
        # Find max peak across all GPUs for this task
        peak_reserved = max(
            (v for k, v in train_peak.items() if "peak_reserved" in k and v is not None),
            default=0.0
        )
        peak_allocated = max(
            (v for k, v in train_peak.items() if "peak_allocated" in k and v is not None),
            default=0.0
        )
        
        print(f"  Task {idx}: {task:20s} peak_reserved={peak_reserved:,.1f}MiB  peak_allocated={peak_allocated:,.1f}MiB")
        
        max_peak_reserved = max(max_peak_reserved, peak_reserved)
        max_peak_allocated = max(max_peak_allocated, peak_allocated)
        total_peak_reserved += peak_reserved
    
    print("-" * 70)
    print(f"  MAX across all tasks:     peak_reserved={max_peak_reserved:,.1f}MiB  peak_allocated={max_peak_allocated:,.1f}MiB")
    print(f"  AVG across all tasks:     peak_reserved={total_peak_reserved/len(all_gpu_peaks):,.1f}MiB")
    print("=" * 70)
    
    # Also save to a summary file for easy comparison
    summary_path = os.path.join(run_root, method, task_sequence, run_stamp, "gpu_summary.json")
    try:
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        summary_data = {
            "method": method,
            "task_sequence": task_sequence,
            "timestamp": run_stamp,
            "max_peak_reserved_mib": max_peak_reserved,
            "max_peak_allocated_mib": max_peak_allocated,
            "avg_peak_reserved_mib": total_peak_reserved / len(all_gpu_peaks) if all_gpu_peaks else 0,
            "per_task": all_gpu_peaks,
        }
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)
        print(f"[GPU-Final] Saved to: {summary_path}\n")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Continual LoRA orchestrator")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
