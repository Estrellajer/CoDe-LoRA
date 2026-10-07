"""AdamW as configured by ``TrainCfg``."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from ..config import TrainCfg


def build_adamw(parameters: Iterable[torch.nn.Parameter], cfg: TrainCfg) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        list(parameters), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps
    )
