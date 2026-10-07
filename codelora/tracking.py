"""Run recording: structured events (``log.jsonl``), the readable ``train.log``, run metadata and optional dashboards.

The recorder is process-global (like ``distributed``): the training loops report each optimizer step to
``recorder()``; ``runner.run`` installs a recorder that writes into the run directory on rank 0.  Without an installed
recorder a null one is active: it only prints the JSON ``train_step`` line to standard output.  Recording reads values the
training already computed, so it cannot change any number.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import torch

from . import profiling
from .config import Config, LogCfg

SCHEMA_VERSION = 1


class Sink(Protocol):
    def scalars(self, values: dict[str, float], step: int) -> None: ...

    def close(self) -> None: ...


class TensorBoardSink:
    def __init__(self, directory: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter  # needs the tensorboard package

        self.writer = SummaryWriter(str(directory))

    def scalars(self, values: dict[str, float], step: int) -> None:
        for tag, value in values.items():
            self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        self.writer.close()


class WandbSink:
    def __init__(self, project: str, name: str, config: dict) -> None:
        import wandb  # needs the wandb package

        self.run = wandb.init(project=project, name=name, config=config)

    def scalars(self, values: dict[str, float], step: int) -> None:
        self.run.log(values, step=step)

    def close(self) -> None:
        self.run.finish()


class Recorder:
    """Writes the run's event log; ``run_dir=None`` records nothing (every non-zero rank, and bare ``train_task`` calls)."""

    def __init__(
        self, run_dir: Path | None = None, cfg: LogCfg | None = None, run_config: Config | None = None
    ) -> None:
        self.dir, self.cfg = run_dir, cfg or LogCfg()
        self.trainable_parameters = 0  # set by the training loop for the current task
        self.index = 0  # the task being trained (set by the runner)
        self.step = 0  # cumulative optimizer steps over tasks (x axis of the dashboards)
        self._window: list[float] = []
        self._sinks: list[Sink] = []
        self._events = self._text = None
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            self._events = (run_dir / "log.jsonl").open("a", encoding="utf-8")
            self._text = (run_dir / "train.log").open("a", encoding="utf-8")
            if self.cfg.tensorboard:
                self._sinks.append(TensorBoardSink(run_dir / "tb"))
            if self.cfg.wandb:
                self._sinks.append(WandbSink(self.cfg.wandb, run_config.name if run_config else "run", {}))

    # ------------------------------------------------------------------ events

    def event(self, event: str, /, **fields: Any) -> None:
        if self._events is None:
            return
        record = {"t": round(time.time(), 3), "event": event, **fields}
        self._events.write(json.dumps(record) + "\n")
        self._events.flush()
        self._text.write(
            f"{time.strftime('%H:%M:%S')} {event} " + " ".join(f"{k}={_short(v)}" for k, v in fields.items()) + "\n"
        )
        self._text.flush()

    def train_step(self, task: str, step: int, branches: list[dict[str, float]], lr: float) -> None:
        """One optimizer step: JSON line on stdout, windowed event in log.jsonl / train.log, dashboard scalars."""

        self.step += 1
        if step % max(self.cfg.stdout_interval, 1) == 0:
            print(json.dumps({"event": "train_step", "task": task, "step": step, "branches": branches}), flush=True)
        if self._events is None:
            return
        self._window.append(sum(b["loss"] for b in branches) / len(branches))
        if step % max(self.cfg.interval, 1) == 0:
            loss_mean = sum(self._window) / len(self._window)
            self._window.clear()
            self.event("train", index=self.index, task=task, step=step, branches=branches, lr=lr, loss_mean=loss_mean)
            if self._sinks:
                scalars = {"train/loss_mean": loss_mean, "train/lr": lr}
                for b in branches:
                    scalars[f"train/{b['adapter']}/loss"] = b["loss"]
                    scalars[f"train/{b['adapter']}/penalty"] = b["penalty"]
                    scalars[f"train/{b['adapter']}/grad_norm"] = b["grad_norm"]
                for sink in self._sinks:
                    sink.scalars(scalars, self.step)

    def scalars(self, values: dict[str, float]) -> None:
        for sink in self._sinks:
            sink.scalars(values, self.step)

    def reset_window(self) -> None:
        self._window.clear()

    def close(self) -> None:
        for sink in self._sinks:
            sink.close()
        for handle in (self._events, self._text):
            if handle is not None:
                handle.close()


def _short(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.5g}"
    text = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
    return text if len(text) <= 120 else text[:117] + "..."


_ACTIVE = Recorder()


def recorder() -> Recorder:
    return _ACTIVE


def install(rec: Recorder) -> Recorder:
    global _ACTIVE
    _ACTIVE = rec
    return rec


# ------------------------------------------------------------------ run metadata


def environment() -> dict[str, Any]:
    git = shutil.which("git")
    done = (
        subprocess.run([git, "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True)
        if git
        else None
    )
    commit = done.stdout.strip() if done is not None and done.returncode == 0 else None
    import peft
    import transformers

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "git_commit": commit,
        "argv": sys.argv,
    }


def adapter_parameters(model: torch.nn.Module, base_ids: set[int]) -> int:
    """Parameters (and growing-rank history buffers) that are not part of the frozen base model."""

    count = sum(p.numel() for p in model.parameters() if id(p) not in base_ids)
    return count + sum(b.numel() for name, b in model.named_buffers() if name.endswith("history_weight"))


def config_digest(config_yaml: str) -> str:
    return hashlib.sha256(config_yaml.encode()).hexdigest()


class Clock:
    """Wall-clock seconds that include all work queued on ``device`` (the run's timings are comparable across methods)."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def __call__(self) -> float:
        profiling.synchronize(self.device)
        return time.perf_counter()


def peak_memory_mib(device: torch.device, reset: bool = False) -> float:
    if device.type != "cuda":
        return 0.0
    peak = torch.cuda.max_memory_allocated(device) / 2**20
    if reset:
        torch.cuda.reset_peak_memory_stats(device)
    return peak
