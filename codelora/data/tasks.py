"""Benchmark loading, the upstream O-LoRA/N-LoRA prompt and answer scoring.

A task is a directory with ``train.json``/``dev.json``/``test.json`` (JSON lists of
``{"sentence", "label"}`` records) and a ``labels.json`` option list; TRACE tasks
use their own record schema (``adapter: trace``).
"""

from __future__ import annotations

import json
import random
import string
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .trace import score_trace, trace_metric, trace_prompt, trace_source, trace_target

# Verbatim from upstream O-LoRA/N-LoRA ``configs/instruction_config.json``.
# COPA has no instruction there: its loader uses only ``{0}\nAnswer:``.
OFFICIAL_INSTRUCTIONS: dict[str, str | None] = {
    "NLI": 'What is the logical relationship between the "sentence 1" and the "sentence 2"? Choose one from the option.\n',
    "QQP": 'Whether the "first sentence" and the "second sentence" have the same meaning? Choose one from the option.\n',
    "SC": "What is the sentiment of the following paragraph? Choose one from the option.\n",
    "TC": "What is the topic of the following paragraph? Choose one from the option.\n",
    "BoolQA": "According to the following passage, is the question true or false? Choose one from the option.\n",
    "MultiRC": "According to the following passage and question, is the candidate answer true or false? Choose one from the option.\n",
    "WiC": "Given a word and two sentences, whether the word is used with the same sense in both sentence? Choose one from the option.\n",
    "COPA": None,
}

CAUSAL_ANSWER_SEPARATOR = "\nAnswer: "


@dataclass(frozen=True)
class Example:
    """One benchmark example; ``source`` is the raw text before any prompt template."""

    input_text: str
    target_text: str
    task: str
    metric: str = "exact_match"
    source: str | None = None
    index: int | None = None  # position in the original split file (before sub-sampling and test exclusions)


@dataclass(frozen=True)
class PromptStyle:
    """``nlora_official`` wraps the source in the upstream instruction + option list."""

    name: str = "nlora_official"
    add_task_name: bool = True
    add_dataset_name: bool = True


def causal_prompt(input_text: str) -> str:
    """Decoder-only prompt used for both training and generation."""

    if input_text.rstrip().endswith(CAUSAL_ANSWER_SEPARATOR.strip()):
        return input_text.rstrip() + " "
    return input_text + CAUSAL_ANSWER_SEPARATOR


def normalize_answer(value: str, remove_punctuation: bool = False) -> str:
    normalized = value.lower()
    if remove_punctuation:
        normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    return " ".join(normalized.split())


def score_prediction(example: Example, prediction: str, official: bool) -> float:
    """Score one prediction.  ``official`` makes generic exact match ignore punctuation."""

    if example.metric == "exact_match":
        if official:
            return float(normalize_answer(prediction, True) == normalize_answer(example.target_text, True))
        return score_trace(prediction, example.target_text, "exact_match")
    return score_trace(prediction, example.target_text, example.metric, source=example.source)


def _load_records(path: Path) -> list[tuple[int, dict]]:
    """``(original index, record)`` pairs, without the records listed in ``test_exclusions.json``."""

    records = list(enumerate(json.loads(path.read_text(encoding="utf-8"))))
    if path.name != "test.json":
        return records
    manifest_path = path.with_name("test_exclusions.json")
    if not manifest_path.is_file():
        return records
    excluded = {
        int(entry["index"]) for entry in json.loads(manifest_path.read_text(encoding="utf-8"))["excluded_records"]
    }
    return [(index, record) for index, record in records if index not in excluded]


def _apply_prompt(examples: Sequence[Example], task_dir: Path, style: PromptStyle) -> list[Example]:
    labels = json.loads((task_dir.joinpath("labels.json")).read_text(encoding="utf-8"))
    family = next(part for part in task_dir.parts if part in OFFICIAL_INSTRUCTIONS)
    prefix = ""
    if style.add_task_name:
        prefix += f"Task:{family}\n"
    if style.add_dataset_name:
        prefix += f"Dataset:{task_dir.name}\n"
    instruction = OFFICIAL_INSTRUCTIONS[family]
    template = (
        prefix + "{0}\nAnswer:"
        if instruction is None
        else prefix + instruction + "Option: " + ", ".join(labels) + " \n{0}\nAnswer:"
    )
    return [
        Example(
            template.format(e.source if e.source is not None else e.input_text),
            e.target_text,
            e.task,
            e.metric,
            e.source,
            e.index,
        )
        for e in examples
    ]


def load_examples(
    name: str,
    task_dir: Path,
    split: str,
    *,
    adapter: str = "generic",
    style: PromptStyle | None = None,
    limit: int | None = None,
    seed: int = 42,
) -> list[Example]:
    """Load ``split`` of a task; ``limit`` keeps a seeded subset in original order."""

    if adapter not in ("generic", "trace"):
        raise ValueError(f"task {name!r}: unknown data adapter {adapter!r}")
    records = _load_records(task_dir / f"{split}.json")
    if adapter == "trace":
        metric = trace_metric(name)
        examples = [
            Example(trace_prompt(name, r), trace_target(r), name, metric, trace_source(r), i) for i, r in records
        ]
    else:
        examples = []
        for i, record in records:
            source = str(record["sentence"])
            examples.append(Example(source.strip(), str(record["label"]).strip(), name, source=source, index=i))
    if limit is not None and limit < len(examples):
        keep = sorted(random.Random(seed).sample(range(len(examples)), limit))
        examples = [examples[i] for i in keep]
    if style is not None and style.name == "nlora_official" and adapter != "trace":
        examples = _apply_prompt(examples, task_dir, style)
    return examples
