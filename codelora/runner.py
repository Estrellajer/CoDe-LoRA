"""The continual-learning run: learn tasks in sequence, score the performance matrix, write results."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import peft
import torch
import transformers
import yaml

from . import distributed, profiling, tracking
from .config import Config, TaskCfg
from .data.collate import CausalLMCollator, Seq2SeqCollator
from .data.tasks import Example, PromptStyle, load_examples, score_prediction
from .evaluation.callbacks import active_callbacks
from .evaluation.metrics import stage_key, stage_tasks, summarize_matrix
from .evaluation.results import ResultsBuilder, forward_transfer_score
from .methods import METHODS, Context, ContinualMethod, build_method
from .models.backbone import load_backbone
from .training.rng import seed_everything


def resolve_device(name: str) -> torch.device:
    if name == "cpu" or not torch.cuda.is_available():
        if name == "cuda":
            raise RuntimeError("device=cuda but CUDA is not available")
        return torch.device("cpu")
    return torch.device("cuda", distributed.get_context().local_rank)


def build_context(cfg: Config) -> Context:
    device = resolve_device(cfg.device)
    model, tokenizer, backbone = load_backbone(cfg.backbone.name, cfg.backbone.path, device)
    d = cfg.data
    collator = (
        Seq2SeqCollator(tokenizer, d.max_source_length, d.max_target_length, pretruncate_sources=cfg.official_protocol)
        if backbone.seq2seq
        else CausalLMCollator(tokenizer, d.max_source_length, d.max_target_length)
    )
    return Context(cfg, model, tokenizer, backbone, device, collator)


def _mean(values: list[float]) -> float:
    total = 0.0
    for value in values:
        total += value
    return total / len(values) if values else float("nan")


class Benchmark:
    """Loads (and caches) the train/eval examples of the configured task sequence."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        d = cfg.data
        self.style = (
            PromptStyle(d.prompt_style, d.add_task_name, d.add_dataset_name)
            if d.prompt_style == "nlora_official"
            else None
        )
        self._eval: dict[int, list[Example]] = {}
        self._dev: dict[str, list[Example]] = {}

    def _dir(self, task: TaskCfg) -> Path:
        return Path(self.cfg.data.root) / task.path

    def train(self, index: int) -> list[Example]:
        task = self.cfg.tasks[index - 1]
        return load_examples(
            task.name,
            self._dir(task),
            "train",
            adapter=task.adapter,
            style=self.style,
            limit=self.cfg.data.max_train_samples,
            seed=self.cfg.seed + index,
        )

    def labels(self, name: str) -> list[str] | None:
        """The task's candidate answers (``labels.json``); None for generation tasks such as TRACE."""

        task = next(t for t in self.cfg.tasks if t.name == name)
        path = self._dir(task) / "labels.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def eval(self, index: int) -> list[Example]:
        """A fixed evaluation subset per task (the seed depends on the task, never on the stage)."""

        if index not in self._eval:
            task = self.cfg.tasks[index - 1]
            self._eval[index] = load_examples(
                task.name,
                self._dir(task),
                self.cfg.eval.split,
                adapter=task.adapter,
                style=self.style,
                limit=self.cfg.data.max_eval_samples,
                seed=self.cfg.seed + 1000 + index,
            )
        return self._eval[index]

    def dev(self, name: str) -> list[Example]:
        """A fixed dev subset of task ``name`` (``fusion.dev_samples`` examples; all if None), whatever ``eval.split`` is."""

        if name not in self._dev:
            index = 1 + [t.name for t in self.cfg.tasks].index(name)
            task = self.cfg.tasks[index - 1]
            self._dev[name] = load_examples(
                task.name,
                self._dir(task),
                "dev",
                adapter=task.adapter,
                style=self.style,
                limit=self.cfg.fusion.dev_samples,
                seed=self.cfg.seed + 3000 + index,
            )
        return self._dev[name]


def evaluate_task(
    method: ContinualMethod, examples: list[Example], official: bool
) -> tuple[float, list[dict[str, Any]]]:
    predictions = method.predict(examples)
    scores = [score_prediction(e, p.text, official) for e, p in zip(examples, predictions, strict=True)]
    records = [
        {
            "index": e.index,
            "input": e.input_text,
            "target": e.target_text,
            "prediction": p.text,
            "route": p.route,
            "confidence": p.confidence,
            "score": s,
            **(p.extra or {}),
        }
        for e, p, s in zip(examples, predictions, scores, strict=True)
    ]
    return (_mean(scores) if scores else 0.0), records


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _checkpoint_payload(cfg: Config, method: ContinualMethod) -> dict[str, Any]:
    return {"method": cfg.method, "config": dataclasses.asdict(cfg), "state": method.state_dict()}


def _save_stage_checkpoint(cfg: Config, method: ContinualMethod, run_dir: Path, index: int) -> None:
    """``save_checkpoint``: all -> every stage; every:N -> every N-th stage; the last stage is written by ``run``."""

    policy = cfg.save_checkpoint
    due = policy == "all" or (policy.startswith("every:") and index % int(policy.split(":")[1]) == 0)
    if due and index < len(cfg.tasks):
        path = run_dir / "checkpoints" / f"after_task_{index:03d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_checkpoint_payload(cfg, method), path)


def run(cfg: Config) -> Path:
    """Run ``cfg`` and return the run directory (only rank 0 writes files; see docs/RESULTS.md)."""

    if cfg.method not in METHODS:
        raise KeyError(f"unknown method {cfg.method!r}; choose one of {sorted(METHODS)}")
    if not cfg.tasks:
        raise ValueError("the configuration contains no tasks")
    seed_everything(cfg.seed)
    ctx = build_context(cfg)
    profiling.configure(cfg.log.profile, ctx.device)
    clock = tracking.Clock(ctx.device)
    base_ids = {id(p) for p in ctx.base.parameters()}
    base_parameters = sum(p.numel() for p in ctx.base.parameters())
    method = build_method(ctx)
    if cfg.train.max_steps is not None and cfg.train.max_steps % method.branches:
        raise ValueError(
            f"train.max_steps counts optimizer updates; it must be divisible by {method.branches} for {cfg.method}"
        )
    method.setup()
    run_dir = Path(cfg.output_dir) / cfg.name
    main = distributed.is_main()
    config_yaml = yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False)
    recorder = tracking.install(tracking.Recorder(run_dir if main else None, cfg.log, cfg))
    run_info = {
        "schema_version": tracking.SCHEMA_VERSION,
        "name": cfg.name,
        "method": cfg.method,
        "seed": cfg.seed,
        "world_size": distributed.get_context().world_size,
        "config_sha256": tracking.config_digest(config_yaml),
        "environment": tracking.environment(),
        "parameters": {"base": base_parameters, "adapter_after_task": {}},
        "started_at": time.time(),
    }
    if main:
        (run_dir / "config.yaml").write_text(config_yaml, encoding="utf-8")
        _write_json(run_dir / "run.json", run_info)
    recorder.event("run_start", name=cfg.name, method=cfg.method, seed=cfg.seed, world_size=run_info["world_size"])

    bench = ctx.benchmark = Benchmark(cfg)
    names = [t.name for t in cfg.tasks]
    performance: dict[str, dict[str, float]] = {}
    task_stats: list[dict[str, Any]] = []
    per_task: list[dict[str, Any]] = []  # results.json: efficiency of every task (training, evaluation, phases)
    eval_seconds: dict[str, float] = {}
    results = ResultsBuilder(cfg.log.per_class)
    callbacks = active_callbacks(cfg)
    for callback in callbacks:
        callback.on_start(method, bench)
    peak_memory = trainable_max = 0
    train_total = eval_total = 0.0
    run_started = clock()
    for index, task in enumerate(cfg.tasks, start=1):
        examples = bench.train(index)
        recorder.index, recorder.trainable_parameters = index, 0
        recorder.reset_window()
        recorder.event("task_start", index=index, task=task.name, n_train=len(examples))
        tracking.peak_memory_mib(ctx.device, reset=True)
        started = clock()
        stats = method.learn_task(index, task, examples)
        stats = {k: _mean(v) if isinstance(v, list) else v for k, v in stats.items()}
        train_seconds = clock() - started
        train_phases = profiling.take()
        adapter_parameters = tracking.adapter_parameters(method.model, base_ids)
        task_peak = tracking.peak_memory_mib(ctx.device)
        peak_memory, trainable_max = max(peak_memory, task_peak), max(trainable_max, recorder.trainable_parameters)
        train_total += train_seconds
        task_stats.append(
            {
                "task": task.name,
                "train_seconds": train_seconds,
                "trainable_parameters": recorder.trainable_parameters,
                "adapter_parameters": adapter_parameters,
                "peak_memory_mib": task_peak,
                **stats,
            }
        )
        run_info["parameters"]["adapter_after_task"][task.name] = adapter_parameters
        recorder.event(
            "task_end",
            index=index,
            task=task.name,
            steps=stats.get("steps"),
            train_seconds=train_seconds,
            trainable_parameters=recorder.trainable_parameters,
            adapter_parameters=adapter_parameters,
            peak_memory_mib=task_peak,
        )

        key, stage_started = stage_key(index), clock()
        ctx.stage = index
        performance[key] = {}
        tracking.peak_memory_mib(ctx.device, reset=True)
        eval_task_seconds: dict[str, float] = {}
        for learned in stage_tasks(cfg.eval.matrix, index, len(cfg.tasks)):
            name = cfg.tasks[learned - 1].name
            started = clock()
            score, records = evaluate_task(method, bench.eval(learned), cfg.official_protocol)
            performance[key][name] = score
            eval_task_seconds[name] = clock() - started
            recorder.event("eval", stage=key, task=name, score=score, n=len(records), seconds=eval_task_seconds[name])
            if main:
                results.add(key, name, records)
                _write_json(run_dir / "evaluations" / key / f"{name}.json", records)
        eval_seconds[key] = clock() - stage_started
        eval_total += eval_seconds[key]
        eval_peak = tracking.peak_memory_mib(ctx.device)
        peak_memory = max(peak_memory, eval_peak)
        per_task.append(
            {
                "task": task.name,
                "train_seconds": train_seconds,
                "eval_seconds": eval_seconds[key],
                "eval_task_seconds": eval_task_seconds,
                "train_peak_memory_mib": task_peak,
                "eval_peak_memory_mib": eval_peak,
                "adapter_parameters": adapter_parameters,
                "trainable_parameters": recorder.trainable_parameters,
                "train_phases": train_phases,
                "eval_phases": profiling.take(),
            }
        )
        recorder.event("stage_end", stage=key, eval_seconds=eval_seconds[key])
        for callback in callbacks:
            callback.after_stage(method, bench, index)
        if main:
            _save_stage_checkpoint(cfg, method, run_dir, index)

    metrics = summarize_matrix(performance, names)
    summary = {
        "schema_version": tracking.SCHEMA_VERSION,
        "status": "passed",
        "method": cfg.method,
        "seed": cfg.seed,
        "world_size": distributed.get_context().world_size,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "tasks": task_stats,
        "performance_matrix": performance,
        "average_performance": metrics.average_performance,
        "backward_transfer": metrics.backward_transfer,
        "diagonal_average_performance": metrics.diagonal_average,
        "diagonal_scores": metrics.diagonal_scores,
        "evaluation_seconds": eval_seconds,
        "peak_allocated_mib": peak_memory,
    }
    for callback in callbacks:
        summary.update(callback.summary())
    fwt = (
        forward_transfer_score(summary["forward_transfer_delta_matrix"], names)
        if "forward_transfer_delta_matrix" in summary
        else None
    )
    summary["FWT"] = fwt
    total_seconds = clock() - run_started
    summary["efficiency"] = {
        "train_seconds": train_total,
        "eval_seconds": eval_total,
        "total_seconds": total_seconds,
        "peak_memory_mib": peak_memory,
        "base_parameters": base_parameters,
        "adapter_parameters": adapter_parameters,
        "adapter_fraction": adapter_parameters / base_parameters,
        "trainable_parameters": trainable_max,
    }
    recorder.event(
        "run_end",
        AP=metrics.average_performance,
        BWT=metrics.backward_transfer,
        FWT=fwt,
        total_seconds=total_seconds,
    )
    if main:
        run_info["finished_at"] = time.time()
        _write_json(run_dir / "run.json", run_info)
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "results.json",
            results.build(
                method=cfg.method,
                seed=cfg.seed,
                tasks=names,
                eval_split=cfg.eval.split,
                matrix_kind=cfg.eval.matrix,
                label_sets={n: bench.labels(n) for n in names} if cfg.fusion.enabled else {},
                performance=performance,
                metrics={
                    "AP": metrics.average_performance,
                    "BWT": metrics.backward_transfer,
                    "FWT": fwt,
                    "diagonal_average": metrics.diagonal_average,
                },
                efficiency={**summary["efficiency"], "per_task": per_task},
            ),
        )
        if cfg.save_checkpoint != "none":
            torch.save(_checkpoint_payload(cfg, method), run_dir / "checkpoint.pt")
        print(
            json.dumps(
                {
                    "status": "passed",
                    "run_dir": str(run_dir),
                    "AP": metrics.average_performance,
                    "BWT": metrics.backward_transfer,
                }
            )
        )
    recorder.close()
    tracking.install(tracking.Recorder())
    distributed.barrier()
    return run_dir
