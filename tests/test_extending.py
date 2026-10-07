"""The method written out in docs/EXTENDING.md trains, predicts and round-trips through its state."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from codelora import distributed
from codelora.config import TaskCfg
from codelora.data.tasks import Example
from codelora.evaluation.generate import generate_routed
from codelora.methods import METHODS, ContinualMethod, Prediction, register
from codelora.models import adapters
from codelora.training.loop import Branch, train_task
from tests.test_codelora import learn, make_method


class AnchoredLoRA(ContinualMethod):
    """Sequential LoRA whose update is pulled towards the update it had after the previous task."""

    weight = 1.0

    def setup(self) -> None:
        ctx = self.ctx
        self.lora_config = adapters.lora_config(ctx.base, ctx.backbone, self.cfg.lora)
        self.model = adapters.build_peft_model(ctx.base, self.lora_config, seed=self.cfg.seed).to(ctx.device)
        self.anchor: dict[str, torch.Tensor] | None = None

    def penalty(self):
        def penalty() -> torch.Tensor:
            total = 0.0
            for name, layer in adapters.lora_layers(self.model, "shared"):
                update = float(layer.scaling["shared"]) * layer.lora_B["shared"].weight @ layer.lora_A["shared"].weight
                total = total + self.weight * (update - self.anchor[name]).pow(2).sum()
            return total

        return penalty

    def learn_task(self, index: int, task: TaskCfg, examples: Sequence[Example]) -> dict[str, Any]:
        distributed.sync_adapters(self.model)
        branch = Branch(
            ["shared"], "shared", penalty=None if self.anchor is None else self.penalty(), rng_role="shared"
        )
        result = train_task(
            self.model,
            [branch],
            examples,
            self.ctx.collator,
            self.cfg.train,
            seed=self.cfg.seed + index,
            label=task.name,
        )
        distributed.sync_adapters(self.model)
        self.anchor = {n: b @ a for n, (b, a) in adapters.effective_factors(self.model, "shared").items()}
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


@pytest.fixture
def registered():
    register("anchored-lora")(
        AnchoredLoRA
    )  # in the docs a decorator; here undone so other tests see the built-ins only
    yield
    del METHODS["anchored-lora"]


def test_a_registered_method_trains_penalises_and_reloads(lab, registered):
    method = make_method(lab, method="anchored-lora")
    assert isinstance(method, AnchoredLoRA)
    stats = learn(method, 2)
    assert max(stats[0]["penalty"]) == 0.0 and max(stats[1]["penalty"]) > 0.0  # the anchor exists from task 2 on

    restored = make_method(lab, method="anchored-lora")
    restored.load_state_dict(method.state_dict())
    examples = method.ctx.benchmark.eval(2)
    assert [p.text for p in restored.predict(examples)] == [p.text for p in method.predict(examples)]
