"""PMI-calibrated fusion of the Co branch and the routed expert on candidate-label scores.

For tasks with a closed label set (classification) CoDe-LoRA can score every candidate answer ``c`` under both branches
instead of generating: ``log p_Co(c | x)`` under the consolidated shared branch and ``log p_De(c | x)`` under the routed
expert (``evaluation/branch_scores.py``).  Each branch is normalised over the candidates and calibrated by pointwise
mutual information against its own prior ``q_b(c)``, the branch's mean predicted label probability on the task's dev split,

    PMI_b(c | x) = log_softmax_c( log p_b(c | x) - log(q_b(c) + 1e-9) ),

and the two are blended with a task-wise weight alpha,

    score(c | x) = alpha * PMI_Co(c | x) + (1 - alpha) * PMI_De(c | x),

with alpha taken from a small grid and chosen by dev accuracy (ties go to the smaller alpha).  Priors and alpha are fitted
on dev only and applied unchanged to test.  Open generation (TRACE) has no candidate set and is not touched.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

PRIOR_EPS = 1e-9


def normalize(logprobs: torch.Tensor) -> torch.Tensor:
    """Log-probabilities of the candidates, renormalised over the candidate set (``[n, labels]``, float64)."""

    return torch.log_softmax(logprobs.double(), dim=-1)


def prior(logprobs: torch.Tensor) -> torch.Tensor:
    """A branch's mean predicted label distribution (plus ``PRIOR_EPS``) over examples."""

    return normalize(logprobs).exp().mean(dim=0) + PRIOR_EPS


def pmi(logprobs: torch.Tensor, label_prior: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(normalize(logprobs) - label_prior.log(), dim=-1)


@dataclass(frozen=True)
class PMIFusion:
    """The dev-fitted rule of one task: the two branch priors and the Co weight."""

    co_prior: torch.Tensor
    expert_prior: torch.Tensor
    alpha: float
    dev_accuracy: float

    def scores(self, co: torch.Tensor, expert: torch.Tensor) -> torch.Tensor:
        return self.alpha * pmi(co, self.co_prior) + (1.0 - self.alpha) * pmi(expert, self.expert_prior)

    def predict(self, co: torch.Tensor, expert: torch.Tensor) -> list[int]:
        """Index of the best candidate per example."""

        return self.scores(co, expert).argmax(dim=-1).tolist()


def fit(co: torch.Tensor, expert: torch.Tensor, gold: Sequence[int], alphas: Sequence[float]) -> PMIFusion:
    """Fit priors and alpha on the dev scores ``co`` / ``expert`` (``[n, labels]`` log-probabilities) and gold indices."""

    co_prior, expert_prior = prior(co), prior(expert)
    target = torch.as_tensor(gold)

    def hits(alpha: float) -> int:
        fused = alpha * pmi(co, co_prior) + (1.0 - alpha) * pmi(expert, expert_prior)
        return int((fused.argmax(dim=-1) == target).sum())

    alpha = max(alphas, key=lambda a: (hits(a), -a))
    return PMIFusion(co_prior, expert_prior, float(alpha), hits(alpha) / len(target))
