"""Seeding and isolated random streams."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator
from contextlib import contextmanager

import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _derived_seed(base_seed: int, index: int, role: str) -> int:
    payload = f"codelora-adapter-init-v1\0{int(base_seed)}\0{index}\0{role}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


@contextmanager
def isolated_rng(base_seed: int, index: int, role: str) -> Iterator[None]:
    """Run a block on its own random stream seeded by ``(base_seed, index, role)``.

    ``fork_rng`` restores the CPU and every CUDA generator on exit, so adapter creation
    and per-branch dropout never shift the surrounding randomness.  Keying by role makes
    corresponding experts of CoDe and De-LoRA start (and drop out) identically.
    """

    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        seed = _derived_seed(base_seed, index, role)
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        yield
