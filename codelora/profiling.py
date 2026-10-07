"""Wall-clock phase timers (``log.profile``): where the seconds of a task go.

``phase(name)`` accumulates the elapsed time of a block under ``name``.  While profiling is off (the default) it is a
shared no-op context.  While it is on, every phase boundary synchronises the device so that asynchronous CUDA work is
charged to the phase that launched it; this is for diagnosing, not for timing a run (the run totals recorded in
``results.json`` never synchronise inside the training loop).  Phases must not be nested.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TypeVar

import torch

T = TypeVar("T")

_ENABLED = False
_DEVICE: torch.device | None = None
_SECONDS: dict[str, float] = defaultdict(float)
_CALLS: dict[str, int] = defaultdict(int)
_IDLE = nullcontext()


def configure(enabled: bool, device: torch.device | None = None) -> None:
    global _ENABLED, _DEVICE
    _ENABLED, _DEVICE = enabled, device
    _SECONDS.clear()
    _CALLS.clear()


def synchronize(device: torch.device | None) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def _timed(name: str) -> Iterator[None]:
    synchronize(_DEVICE)
    started = time.perf_counter()
    try:
        yield
    finally:
        synchronize(_DEVICE)
        _SECONDS[name] += time.perf_counter() - started
        _CALLS[name] += 1


def phase(name: str) -> AbstractContextManager[None]:
    return _timed(name) if _ENABLED else _IDLE


def timed_iter(name: str, items: Iterable[T]) -> Iterator[T]:
    """Iterate ``items`` charging the time spent producing each element to ``name`` (data loading)."""

    iterator = iter(items)
    while True:
        try:
            with phase(name):
                item = next(iterator)
        except StopIteration:
            return
        yield item


def take() -> dict[str, dict[str, float]]:
    """The phases recorded since the last call (``{name: {seconds, calls}}``); resets the timers."""

    taken = {name: {"seconds": _SECONDS[name], "calls": _CALLS[name]} for name in sorted(_SECONDS)}
    _SECONDS.clear()
    _CALLS.clear()
    return taken
