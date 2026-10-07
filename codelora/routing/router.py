"""The task-router interface and registry.

A router learns one set of statistics per task from a few support embeddings (the frozen base model's pooled hidden
state of training prompts) and maps the embedding of an unseen example to ``(task, confidence)``.  Register new
routers with ``@register_router("name")`` and select them with ``router.kind: name``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from ..config import RouterCfg

ROUTERS: dict[str, type[Router]] = {}


class Router:
    seed_offset: int = 0  # offsets the seed of the support draw, so that different routers draw independent supports

    def __init__(self, cfg: RouterCfg) -> None:
        self.cfg = cfg

    @property
    def support_samples(self) -> int:
        """How many training examples per task the router wants to see."""

        raise NotImplementedError

    def add_task(self, task: str, support: torch.Tensor) -> None:
        """Record the statistics of ``task`` from its ``[n, hidden]`` support embeddings."""

        raise NotImplementedError

    def route(self, embedding: torch.Tensor) -> tuple[str, float]:
        """``(best task, confidence)`` for one ``[hidden]`` embedding."""

        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state: dict[str, Any]) -> None:
        raise NotImplementedError


def register_router(name: str) -> Callable[[type[Router]], type[Router]]:
    def decorator(cls: type[Router]) -> type[Router]:
        ROUTERS[name] = cls
        return cls

    return decorator


def build_router(cfg: RouterCfg) -> Router:
    from . import lda, prototype  # noqa: F401  (register the built-in routers)

    if cfg.kind not in ROUTERS:
        raise KeyError(f"unknown router {cfg.kind!r}; choose one of {sorted(ROUTERS)}")
    return ROUTERS[cfg.kind](cfg)
