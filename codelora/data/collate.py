"""Tokenizer collation for T5-style and decoder-only models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .tasks import Example, causal_prompt


def pretruncate(tokenizer: Any, texts: Sequence[str], max_length: int) -> list[str]:
    """Upstream O-LoRA/N-LoRA pre-truncation: tokenize, cut, decode back to text."""

    prepared = list(texts)
    for index, text in enumerate(prepared):
        ids = tokenizer(text)["input_ids"]
        if len(ids) > max_length:
            prepared[index] = tokenizer.decode(ids[:max_length], skip_special_tokens=True)
    return prepared


@dataclass
class Seq2SeqCollator:
    """Encoder inputs and ``-100``-masked labels for T5-like models."""

    tokenizer: Any
    max_source_length: int = 512
    max_target_length: int = 50
    pretruncate_sources: bool = False

    def __call__(self, batch: Sequence[Example]) -> dict[str, torch.Tensor]:
        sources = [e.input_text for e in batch]
        if self.pretruncate_sources:
            sources = pretruncate(self.tokenizer, sources, self.max_source_length)
        inputs = self.tokenizer(
            sources, max_length=self.max_source_length, truncation=True, padding=True, return_tensors="pt"
        )
        targets = self.tokenizer(
            text_target=[e.target_text for e in batch],
            max_length=self.max_target_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        labels = targets["input_ids"].masked_fill(targets["input_ids"].eq(self.tokenizer.pad_token_id), -100)
        return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": labels}


@dataclass
class CausalLMCollator:
    """``prompt + target + EOS`` with prompt tokens masked out of the loss.

    Prompt and target are tokenized separately so that a long prompt can never
    truncate the answer away; over-long prompts keep their tail (the tokenizer's
    ``truncation_side`` is ``left``) so the ``Answer:`` cue survives.
    """

    tokenizer: Any
    max_source_length: int = 512
    max_target_length: int = 50

    def __call__(self, batch: Sequence[Example]) -> dict[str, torch.Tensor]:
        tok = self.tokenizer
        prompts = tok(
            [causal_prompt(e.input_text) for e in batch],
            max_length=self.max_source_length,
            truncation=True,
            padding=False,
        )["input_ids"]
        targets = [
            ids[: self.max_target_length]
            for ids in tok([e.target_text for e in batch], padding=False, add_special_tokens=False)["input_ids"]
        ]
        rows, label_rows = [], []
        for prompt, target in zip(prompts, targets, strict=True):
            if not target:
                raise ValueError("tokenizer produced an empty causal target")
            if tok.eos_token_id is not None and len(target) < self.max_target_length and target[-1] != tok.eos_token_id:
                target = [*target, int(tok.eos_token_id)]
            rows.append([*prompt, *target])
            label_rows.append([-100] * len(prompt) + target)
        width = max(map(len, rows))
        pad = int(tok.pad_token_id)
        left = tok.padding_side == "left"
        ids, labels, mask = [], [], []
        for row, label in zip(rows, label_rows, strict=True):
            gap = width - len(row)
            ids.append([pad] * gap + row if left else row + [pad] * gap)
            labels.append([-100] * gap + label if left else label + [-100] * gap)
            mask.append([0] * gap + [1] * len(row) if left else [1] * len(row) + [0] * gap)
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(mask),
            "labels": torch.tensor(labels),
        }
