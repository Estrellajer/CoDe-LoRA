"""The continual-learning method interface and registry."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..config import Config, TaskCfg
from ..data.tasks import Example
from ..evaluation.generate import GenerationSettings
from ..models.backbone import Backbone


@dataclass
class Prediction:
    text: str
    route: str | None = None  # the task / adapter a router picked
    confidence: float | None = None
    extra: dict[str, Any] | None = None  # additional per-example record fields (e.g. branch log-probs)


@dataclass
class Context:
    """Everything a method needs: the frozen base model, tokenizer and run settings."""

    cfg: Config
    base: nn.Module
    tokenizer: Any
    backbone: Backbone
    device: torch.device
    collator: Callable
    benchmark: Any = None  # the run's ``Benchmark`` (set by the runner): dev split and candidate labels of a task
    stage: int | None = None  # the stage being evaluated (set by the runner); None: the final model, outside a run

    @property
    def generation(self) -> GenerationSettings:
        cfg = self.cfg
        return GenerationSettings(
            max_source_length=cfg.data.max_source_length,
            max_target_length=cfg.data.max_target_length,
            batch_size=cfg.eval.batch_size,
            official=cfg.official_protocol,
            sort_by_length=cfg.eval.sort_by_length,
        )


class ContinualMethod:
    """One continual-learning method: learn tasks in sequence, predict task-agnostically."""

    name: str = ""
    branches: int = 1  # optimizer updates per training iteration (one per trained branch)

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.cfg = ctx.cfg

    def setup(self) -> None:
        """Build the adapter structure on the loaded base model."""

        raise NotImplementedError

    def learn_task(self, index: int, task: TaskCfg, examples: Sequence[Example]) -> dict[str, Any]:
        """Train on task ``index`` (1-based) and update the deployed state; returns run statistics."""

        raise NotImplementedError

    def predict(self, examples: Sequence[Example]) -> list[Prediction]:
        """Task-agnostic inference over the current deployed state."""

        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state: dict[str, Any]) -> None:
        raise NotImplementedError


METHODS: dict[str, type[ContinualMethod]] = {}


def register(name: str) -> Callable[[type[ContinualMethod]], type[ContinualMethod]]:
    """Class decorator: make the method selectable as ``method: <name>`` in a config."""

    def decorator(cls: type[ContinualMethod]) -> type[ContinualMethod]:
        cls.name = name
        METHODS[name] = cls
        return cls

    return decorator


def build_method(ctx: Context) -> ContinualMethod:
    if ctx.cfg.method not in METHODS:
        raise KeyError(f"unknown method {ctx.cfg.method!r}; choose one of {sorted(METHODS)}")
    return METHODS[ctx.cfg.method](ctx)
