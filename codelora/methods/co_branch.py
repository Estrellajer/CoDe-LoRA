"""The consolidated shared (Co) branch of CoDe-LoRA: a fixed-rank ``shared`` adapter that absorbs every task's expert.

After task ``t`` the expert's update ``E_t`` is folded into ``shared`` by an exact rank-``lora.r`` SVD retraction,
``Co_t = rank_r(c_t Co_{t-1} + s_t E_t^perp)``, computed on the low-rank factors (QR + SVD of the small core), so the
branch never grows.  ``E_t^perp`` is ``E_t`` with its component along the column space of ``Co_{t-1}`` removed
(``colora.projection: null_space``; ``none`` folds ``E_t`` as is), ``(c_t, s_t)`` come from ``colora.scaling`` (``sqrt``:
``sqrt((t-1)/t)``, ``1/sqrt(t)``; ``additive``: 1, 1).  The fold is a deterministic function of the experts, which is
what lets a checkpoint store only the experts.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from ..config import Config
from ..models import adapters
from .consolidation import factorize_low_rank_sum, null_space_project, retract_coefficients


class CoBranch:
    def __init__(self, cfg: Config, model: nn.Module) -> None:
        self.cfg, self.model = cfg, model

    def fold(self, index: int, expert: str) -> dict[str, Any]:
        """Write ``rank_r(c_t shared + s_t expert_perp)`` into ``shared`` and freeze it; ``expert`` is left in place.

        For the first task ``shared`` is still zero (and ``c_1 = 0`` under ``sqrt``), so this is the (balanced)
        re-factorisation of the first expert.
        """

        model, colora = self.model, self.cfg.colora
        c_t, s_t = retract_coefficients(colora.scaling, index)
        shared, update = adapters.effective_factors(model, "shared"), adapters.effective_factors(model, expert)
        if colora.projection == "null_space":
            update = {name: null_space_project(shared[name], update[name]) for name in update}
        deployed = {
            name: factorize_low_rank_sum(
                [(c_t * shared[name][0], shared[name][1]), (s_t * update[name][0], update[name][1])], self.cfg.lora.r
            )
            for name in sorted(shared)
        }
        adapters.write_factors(model, "shared", deployed)
        model.set_adapter("shared")
        adapters.freeze(model, "shared")
        norm_sq = sum(float(((b.T @ b) * (a @ a.T)).sum()) for b, a in deployed.values())
        return {"shared_fro_norm": norm_sq**0.5}  # ||W_Co - W0||_F
