"""Backbone loading: T5-large, Qwen3-0.6B, Llama-2-7B and Qwen3.5-4B (all bf16)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class Backbone:
    name: str
    seq2seq: bool
    default_targets: tuple[str, ...]  # ("auto",): discover text attention projections
    task_type: str

    @property
    def causal(self) -> bool:
        return not self.seq2seq


BACKBONES = {
    "t5-large": Backbone("t5-large", True, ("q", "v"), "SEQ_2_SEQ_LM"),
    "qwen3-0.6b": Backbone("qwen3-0.6b", False, ("q_proj", "v_proj"), "CAUSAL_LM"),
    "llama2-7b": Backbone("llama2-7b", False, ("q_proj", "v_proj"), "CAUSAL_LM"),
    "qwen3.5-4b": Backbone("qwen3.5-4b", False, ("auto",), "CAUSAL_LM"),
}

# Qwen3.5 interleaves full and linear attention, so q/v-only LoRA would skip much of the text
# backbone.  Exact qualified names also avoid adapting the visual tower.
_QWEN35_SUFFIXES = {"q_proj", "v_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"}


def lora_targets(model: nn.Module, backbone: Backbone, requested: list[str] | None) -> list[str]:
    targets = list(requested or backbone.default_targets)
    if targets != ["auto"]:
        return targets
    found = []
    for name, module in model.named_modules():
        lowered = name.lower()
        if not isinstance(module, nn.Linear) or "vision" in lowered or "visual" in lowered:
            continue
        if name.rsplit(".", 1)[-1] in _QWEN35_SUFFIXES and "attn" in lowered:
            found.append(name)
    return found


def _eos_stops_generation(model: nn.Module, tokenizer: Any) -> None:
    """Make ``generate`` stop on the EOS token that the collator appends to every target.

    Qwen3.5 ships no generation_config.json and lists <|endoftext|> as EOS while the tokenizer (and the
    training labels) end answers with <|im_end|>.
    """

    configured = model.generation_config.eos_token_id
    ids = [configured] if isinstance(configured, int) else list(configured or [])
    if tokenizer.eos_token_id is not None and int(tokenizer.eos_token_id) not in ids:
        ids.append(int(tokenizer.eos_token_id))
    model.generation_config.eos_token_id = ids if len(ids) > 1 else (ids[0] if ids else None)


def load_backbone(name: str, path: str, device: torch.device) -> tuple[nn.Module, Any, Backbone]:
    """Load the frozen base model and tokenizer from a local directory."""

    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoModelForSeq2SeqLM,
        AutoProcessor,
        AutoTokenizer,
    )

    backbone = BACKBONES[name]
    if name == "qwen3.5-4b":
        tokenizer = AutoProcessor.from_pretrained(path).tokenizer
        model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16)
    elif backbone.seq2seq:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path, dtype=torch.bfloat16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    if backbone.causal:
        tokenizer.truncation_side = "left"  # over-long prompts keep the answer cue
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        _eos_stops_generation(model, tokenizer)
    return model.to(device), tokenizer, backbone
