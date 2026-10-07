"""Shared-covariance (shrinkage) linear discriminant router.

Only sufficient statistics are kept: each learned task contributes its support mean and
centred scatter, and the shared covariance is the pooled scatter of all learned tasks.
Statistics and the solve use float64.  Confidence is the top task's posterior under equal priors.
"""

from __future__ import annotations

from typing import Any

import torch

from .router import Router, register_router


@register_router("lda")
class LDARouter(Router):
    seed_offset = 3000

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.means: dict[str, torch.Tensor] = {}
        self.pooled_scatter: torch.Tensor | None = None
        self.count = 0
        self._solved: tuple[list[str], torch.Tensor, torch.Tensor] | None = None

    @property
    def support_samples(self) -> int:
        return self.cfg.lda_samples

    def add_task(self, task: str, support: torch.Tensor) -> None:
        features = support.detach().double().cpu()
        mean = features.mean(dim=0)
        centred = features - mean
        scatter = centred.T @ centred
        self.means[task] = mean
        self.pooled_scatter = scatter if self.pooled_scatter is None else self.pooled_scatter + scatter
        self.count += int(features.shape[0])
        self._solved = None

    def _solve(self) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        if self._solved is None:
            tasks = list(self.means)
            means = torch.stack([self.means[t] for t in tasks]).double()
            dof = max(self.count - len(tasks), 1)
            cov = self.pooled_scatter.double() / dof
            shrinkage = self.cfg.lda_shrinkage
            dim = cov.shape[0]
            cov = (1.0 - shrinkage) * cov + shrinkage * torch.trace(cov) / dim * torch.eye(dim, dtype=cov.dtype)
            weights = torch.linalg.solve(cov, means.T).T
            self._solved = (tasks, weights, -0.5 * (weights * means).sum(dim=1))
        return self._solved

    def route(self, embedding: torch.Tensor) -> tuple[str, float]:
        tasks, weights, bias = self._solve()
        posterior = torch.softmax(weights @ embedding.detach().double().cpu() + bias, dim=0)
        scores = {task: float(p) for task, p in zip(tasks, posterior, strict=True)}
        return max(scores.items(), key=lambda pair: pair[1])

    def state_dict(self) -> dict[str, Any]:
        return {"means": self.means, "pooled_scatter": self.pooled_scatter, "count": self.count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.means, self.pooled_scatter, self.count = state["means"], state["pooled_scatter"], state["count"]
        self._solved = None
