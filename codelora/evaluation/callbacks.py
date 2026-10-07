"""Optional run-time extensions of the continual loop (off by default).

A callback sees the run at two points -- before the first task and after every stage's evaluation -- and contributes
entries to ``summary.json``.  Callbacks register with ``@register_callback`` and declare when a config enables them.
"""

from __future__ import annotations

from typing import Any

from ..config import Config

CALLBACKS: list[type[Callback]] = []


class Callback:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    @staticmethod
    def enabled(cfg: Config) -> bool:
        raise NotImplementedError

    def on_start(self, method: Any, bench: Any) -> None:
        """Before the first task (the model is untrained)."""

    def after_stage(self, method: Any, bench: Any, index: int) -> None:
        """After task ``index`` has been learned and the stage's matrix row has been scored."""

    def summary(self) -> dict[str, Any]:
        """Entries merged into ``summary.json``."""

        return {}


def register_callback(cls: type[Callback]) -> type[Callback]:
    CALLBACKS.append(cls)
    return cls


def active_callbacks(cfg: Config) -> list[Callback]:
    from . import forward_transfer  # noqa: F401  (registers the optional callbacks)

    return [cls(cfg) for cls in CALLBACKS if cls.enabled(cfg)]
