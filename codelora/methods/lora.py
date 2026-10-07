"""Sequential LoRA: one persistent adapter trained on every task in turn (the forgetting baseline)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from peft import get_peft_model_state_dict, set_peft_model_state_dict

from .. import distributed
from ..config import TaskCfg
from ..data.tasks import Example
from ..evaluation.generate import generate_routed
from ..models import adapters
from ..training.loop import Branch, train_task
from .base import ContinualMethod, Prediction, register


@register("lora")
class SequentialLoRA(ContinualMethod):
    def setup(self) -> None:
        ctx = self.ctx
        self.lora_config = adapters.lora_config(ctx.base, ctx.backbone, self.cfg.lora)
        self.model = adapters.build_peft_model(ctx.base, self.lora_config, seed=self.cfg.seed).to(ctx.device)

    def branch(self) -> Branch:
        return Branch(["shared"], "shared", rng_role="shared")

    def learn_task(self, index: int, task: TaskCfg, examples: Sequence[Example]) -> dict[str, Any]:
        distributed.sync_adapters(self.model)
        result = train_task(
            self.model,
            [self.branch()],
            examples,
            self.ctx.collator,
            self.cfg.train,
            seed=self.cfg.seed + index,
            label=task.name,
        )
        distributed.sync_adapters(self.model)
        return {"steps": result.steps, "loss": result.loss[0], "penalty": result.penalty[0]}

    def predict(self, examples: Sequence[Example]) -> list[Prediction]:
        ctx = self.ctx
        texts = [e.input_text for e in examples]
        outputs = generate_routed(
            self.model, ctx.tokenizer, ctx.backbone, texts, ["shared"] * len(texts), ctx.generation
        )
        return [Prediction(text) for text in outputs]

    def state_dict(self) -> dict[str, Any]:
        return {"shared": get_peft_model_state_dict(self.model, adapter_name="shared")}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        set_peft_model_state_dict(self.model, state["shared"], adapter_name="shared")
