"""Cosine prototype router: a task prototype is the normalised mean of normalised support embeddings."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .router import Router, register_router

EPS = 1e-12


def build_prototype(support: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(support, p=2, dim=-1, eps=EPS)
    return F.normalize(normalized.mean(dim=0), p=2, dim=0, eps=EPS)


@register_router("cosine")
class CosineRouter(Router):
    """Top task by cosine similarity to its prototype; confidence is that cosine."""

    seed_offset = 2000

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.prototypes: dict[str, torch.Tensor] = {}

    @property
    def support_samples(self) -> int:
        return self.cfg.prototype_samples

    def add_task(self, task: str, support: torch.Tensor) -> None:
        self.prototypes[task] = build_prototype(support)

    def route(self, embedding: torch.Tensor) -> tuple[str, float]:
        query = F.normalize(embedding, p=2, dim=0, eps=EPS)
        # normalised dot products can overshoot the cosine range by a few ulps
        scores = {t: max(-1.0, min(1.0, float(torch.dot(query, p).item()))) for t, p in self.prototypes.items()}
        return max(scores.items(), key=lambda pair: pair[1])

    def state_dict(self) -> dict[str, Any]:
        return {"prototypes": self.prototypes}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.prototypes = state["prototypes"]
