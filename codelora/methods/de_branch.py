"""The De (decoupling) branch: one frozen PEFT expert per task plus a task router.

Used alone by De-LoRA (always route to the best expert) and by CoDe-LoRA (route to
the best expert only when the router is confident, else fall back to the shared branch).
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from typing import Any

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from ..config import Config
from ..data.tasks import Example
from ..models import adapters
from ..routing.embed import base_embeddings
from ..routing.router import build_router
from .base import Context


def safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip()).strip("-")
    if not clean:
        raise ValueError(f"cannot derive an adapter name from {value!r}")
    return clean


def expert_adapter_name(index: int, task_name: str) -> str:
    return f"expert_{index:03d}_{safe_name(task_name)}"


def select_support(examples: Sequence[Example], count: int, seed: int) -> list[Example]:
    """Seeded support subset without a prefix bias, in original order."""

    chosen = sorted(random.Random(seed).sample(range(len(examples)), min(count, len(examples))))
    return [examples[i] for i in chosen]


def router_texts(examples: Sequence[Example], embedding_input: str) -> list[str]:
    if embedding_input == "raw_source":  # the example's own text: routing cannot read the prompt template
        return [e.source if e.source is not None else e.input_text for e in examples]
    return [e.input_text for e in examples]


class DeBranch:
    """Experts (``task -> adapter``) and the router statistics that select among them."""

    def __init__(self, ctx: Context, model: torch.nn.Module, lora_config) -> None:
        self.ctx, self.model, self.lora_config = ctx, model, lora_config
        self.cfg: Config = ctx.cfg
        self.experts: dict[str, str] = {}
        self.router = build_router(self.cfg.router)

    def new_expert(self, index: int, task_name: str) -> str:
        name = expert_adapter_name(index, task_name)
        adapters.add_adapter(self.model, name, self.lora_config, self.cfg.seed, index, "expert")
        self.experts[task_name] = name
        return name

    def _embed(self, examples: Sequence[Example]) -> torch.Tensor:
        return base_embeddings(
            self.model,
            self.ctx.tokenizer,
            self.ctx.backbone,
            router_texts(examples, self.cfg.router.embedding_input),
            batch_size=self.cfg.eval.embedding_batch_size,
            max_length=self.cfg.data.max_source_length,
        )

    def register(self, index: int, task_name: str, examples: Sequence[Example]) -> None:
        """Record the task's routing statistics from its (frozen-base) training support set."""

        seed = self.cfg.seed + self.router.seed_offset + index
        self.router.add_task(task_name, self._embed(select_support(examples, self.router.support_samples, seed)))

    def route(self, examples: Sequence[Example]) -> list[tuple[str, float]]:
        return [self.router.route(embedding) for embedding in self._embed(examples)]

    def state_dict(self) -> dict[str, Any]:
        return {
            "experts": dict(self.experts),
            "adapters": {
                name: get_peft_model_state_dict(self.model, adapter_name=name) for name in self.experts.values()
            },
            "router": self.router.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for index, (task, name) in enumerate(state["experts"].items(), start=1):
            adapters.add_adapter(self.model, name, self.lora_config, self.cfg.seed, index, "expert")
            set_peft_model_state_dict(self.model, state["adapters"][name], adapter_name=name)
            adapters.freeze(self.model, name)
            self.experts[task] = name
        self.router.load_state_dict(state["router"])
