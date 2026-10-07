"""Shared-branch-only methods: O-LoRA, N-LoRA and Co-LoRA.

One growing-rank ``shared`` adapter (``models/growing.py``) is trained on every task, following the official O-LoRA /
N-LoRA protocol: the process is re-seeded with ``seed`` before every new block, every task trains on the same data
order and dropout draws from the global random stream.  The methods differ in the penalty on the block trained at the
current task and in the task-boundary consolidation:

``o-lora``   orthogonality to the history, then commit the block (the rank grows by ``lora.r``);
``n-lora``   L1 sparsity of the block, then commit;
``co-lora``  the N-LoRA penalty, then replace history + block by their rank-``lora.r`` SVD (the rank stays fixed).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from .. import distributed, profiling
from ..config import TaskCfg
from ..data.tasks import Example
from ..evaluation.generate import generate_routed
from ..models import adapters, growing
from ..training.loop import Branch, train_task
from ..training.rng import seed_everything
from .base import ContinualMethod, Prediction, register
from .consolidation import svd_truncate
from .regularizers import nlora_penalty, olora_penalty


class GrowingBranch:
    """The growing-rank ``shared`` adapter: what is trained at a task and how the task is consolidated."""

    def __init__(self, method: SharedBranchMethod) -> None:
        self.cfg, self.model = method.cfg, method.model
        growing.install_growing_rank(self.model, "shared")

    def penalty(self) -> Callable[[], torch.Tensor]:
        raise NotImplementedError

    def begin_task(self, index: int) -> Branch:
        if index > 1:
            seed_everything(self.cfg.seed)
            growing.begin_task(self.model, "shared", self.cfg.lora.r)
        return Branch(["shared"], "shared", penalty=self.penalty())

    def end_task(self) -> dict[str, Any]:
        raise NotImplementedError

    def commit(self) -> dict[str, Any]:
        """Freeze the current block into the history (the rank grows by ``lora.r``)."""

        return {"cumulative_rank": growing.commit_task(self.model, "shared")}

    def state_dict(self) -> dict[str, Any]:
        return {"history": growing.export_state(self.model, "shared")}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        history = {name: (blocks["B"], blocks["A"]) for name, blocks in state["history"].items()}
        growing.replace_history(self.model, "shared", history)


class OLoraBranch(GrowingBranch):
    def penalty(self) -> Callable[[], torch.Tensor]:
        c = self.cfg.olora
        return lambda: olora_penalty(
            growing.current_factors(self.model, "shared"),
            {name: a for name, (_, a) in growing.history_factors(self.model, "shared").items()},
            c.lambda_orth,
            c.lambda_l2,
        )

    end_task = GrowingBranch.commit


class NLoraBranch(GrowingBranch):
    def penalty(self) -> Callable[[], torch.Tensor]:
        c = self.cfg.nlora
        return lambda: nlora_penalty(growing.current_factors(self.model, "shared").values(), c.lambda_l1, c.reduction)

    end_task = GrowingBranch.commit


class CoLoraBranch(NLoraBranch):
    """N-LoRA training; the boundary replaces history + current block by the optimal rank-``lora.r`` SVD of their sum."""

    def penalty(self) -> Callable[[], torch.Tensor]:
        c = self.cfg.colora
        return lambda: nlora_penalty(growing.current_factors(self.model, "shared").values(), c.lambda_l1, c.reduction)

    def end_task(self) -> dict[str, Any]:
        history = growing.history_factors(self.model, "shared")
        current = growing.current_factors(self.model, "shared")
        merged = {}
        for name in sorted(current):
            history_b, history_a = history[name]
            current_b, current_a = current[name]
            dense = history_b.detach() @ history_a.detach() + current_b.detach() @ current_a.detach()
            merged[name] = svd_truncate(dense, self.cfg.lora.r)
        growing.replace_history(self.model, "shared", merged)
        return {"cumulative_rank": self.cfg.lora.r}


class SharedBranchMethod(ContinualMethod):
    branch_class: type[GrowingBranch]

    def setup(self) -> None:
        ctx = self.ctx
        lora_config = adapters.lora_config(ctx.base, ctx.backbone, self.cfg.lora)
        self.model = adapters.build_peft_model(ctx.base, lora_config, seed=None).to(ctx.device)
        self.shared = self.branch_class(self)

    def learn_task(self, index: int, task: TaskCfg, examples: Sequence[Example]) -> dict[str, Any]:
        branch = self.shared.begin_task(index)
        distributed.sync_adapters(self.model)
        result = train_task(
            self.model,
            [branch],
            examples,
            self.ctx.collator,
            self.cfg.train,
            seed=self.cfg.seed,
            label=task.name,
        )
        with profiling.phase("consolidation"):
            stats = self.shared.end_task()
            distributed.sync_adapters(self.model)
        return {"steps": result.steps, "loss": result.loss[0], "penalty": result.penalty[0], **stats}

    def predict(self, examples: Sequence[Example]) -> list[Prediction]:
        ctx = self.ctx
        texts = [e.input_text for e in examples]
        with profiling.phase("generate"):
            outputs = generate_routed(
                self.model, ctx.tokenizer, ctx.backbone, texts, ["shared"] * len(texts), ctx.generation
            )
        return [Prediction(text) for text in outputs]

    def state_dict(self) -> dict[str, Any]:
        return self.shared.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.shared.load_state_dict(state)


@register("o-lora")
class OLoRA(SharedBranchMethod):
    branch_class = OLoraBranch


@register("n-lora")
class NLoRA(SharedBranchMethod):
    branch_class = NLoraBranch


@register("co-lora")
class CoLoRA(SharedBranchMethod):
    branch_class = CoLoraBranch
