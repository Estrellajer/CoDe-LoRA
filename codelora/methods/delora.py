"""De-LoRA: an isolated expert per task, selected by the task router at inference."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .. import distributed
from ..config import TaskCfg
from ..data.tasks import Example
from ..evaluation.generate import generate_routed
from ..models import adapters
from ..training.loop import Branch, train_task
from .base import ContinualMethod, Prediction, register
from .de_branch import DeBranch


@register("de-lora")
class DeLoRA(ContinualMethod):
    def setup(self) -> None:
        ctx = self.ctx
        self.lora_config = adapters.lora_config(ctx.base, ctx.backbone, self.cfg.lora)
        self.model = adapters.build_peft_model(ctx.base, self.lora_config, seed=self.cfg.seed).to(ctx.device)
        self.de = DeBranch(ctx, self.model, self.lora_config)

    def learn_task(self, index: int, task: TaskCfg, examples: Sequence[Example]) -> dict[str, Any]:
        expert = self.de.new_expert(index, task.name)
        distributed.sync_adapters(self.model)
        result = train_task(
            self.model,
            [Branch([expert], expert, rng_role="expert")],
            examples,
            self.ctx.collator,
            self.cfg.train,
            seed=self.cfg.seed + index,
            label=task.name,
        )
        distributed.sync_adapters(self.model)
        self.de.register(index, task.name, examples)
        adapters.freeze(self.model, expert)
        return {"steps": result.steps, "loss": result.loss[0], "penalty": result.penalty[0]}

    def predict(self, examples: Sequence[Example]) -> list[Prediction]:
        ctx = self.ctx
        routes = self.de.route(examples)
        names = [self.de.experts[task] for task, _ in routes]
        outputs = generate_routed(
            self.model, ctx.tokenizer, ctx.backbone, [e.input_text for e in examples], names, ctx.generation
        )
        return [Prediction(text, task, confidence) for text, (task, confidence) in zip(outputs, routes, strict=True)]

    def state_dict(self) -> dict[str, Any]:
        return self.de.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.de.load_state_dict(state)
