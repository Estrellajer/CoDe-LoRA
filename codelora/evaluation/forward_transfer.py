"""Optional forward-transfer evaluation (``eval.forward_transfer: true``).

Before training, the untrained model is scored on every task's fixed eval subset (the baseline); after each stage the
tasks that have not been learned yet are scored with the deployed state.  ``forward_transfer_delta_matrix`` is the gain
over the baseline.  Unseen data never reaches training, routing statistics or calibration.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from .. import distributed
from ..config import Config
from ..data.tasks import Example, score_prediction
from .callbacks import Callback, register_callback
from .generate import generate_batch
from .metrics import mean, stage_key


def score_untrained(method: Any, examples: list[Example]) -> float:
    """Score of the base model (adapters disabled) on ``examples``, batched like every other evaluation."""

    ctx, model = method.ctx, method.model
    settings = ctx.generation

    def generate(shard: list[Example]) -> list[str]:
        order = (
            sorted(range(len(shard)), key=lambda i: len(shard[i].input_text))
            if settings.sort_by_length
            else range(len(shard))
        )
        order = list(order)
        outputs = [""] * len(shard)
        for start in range(0, len(order), settings.batch_size):
            chunk = order[start : start + settings.batch_size]
            decoded = generate_batch(model, ctx.tokenizer, ctx.backbone, [shard[i].input_text for i in chunk], settings)
            for i, text in zip(chunk, decoded, strict=True):
                outputs[i] = text
        return outputs

    disabled = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
    with disabled:
        predictions = distributed.sharded_map(generate, examples)
    return mean([score_prediction(e, p, ctx.cfg.official_protocol) for e, p in zip(examples, predictions, strict=True)])


@register_callback
class ForwardTransfer(Callback):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.baseline: dict[str, float] = {}
        self.matrix: dict[str, dict[str, float]] = {}
        self.delta: dict[str, dict[str, float]] = {}

    @staticmethod
    def enabled(cfg: Config) -> bool:
        return cfg.eval.forward_transfer

    def on_start(self, method: Any, bench: Any) -> None:
        for index, task in enumerate(self.cfg.tasks, start=1):
            self.baseline[task.name] = score_untrained(method, bench.eval(index))

    def after_stage(self, method: Any, bench: Any, index: int) -> None:
        key = stage_key(index)
        self.matrix[key], self.delta[key] = {}, {}
        for unseen, task in enumerate(self.cfg.tasks[index:], start=index + 1):
            examples = bench.eval(unseen)
            predictions = method.predict(examples)
            scores = [
                score_prediction(e, p.text, self.cfg.official_protocol)
                for e, p in zip(examples, predictions, strict=True)
            ]
            self.matrix[key][task.name] = mean(scores)
            self.delta[key][task.name] = self.matrix[key][task.name] - self.baseline[task.name]

    def summary(self) -> dict[str, Any]:
        return {
            "forward_transfer_baseline_matrix": self.baseline,
            "forward_transfer_matrix": self.matrix,
            "forward_transfer_delta_matrix": self.delta,
        }
