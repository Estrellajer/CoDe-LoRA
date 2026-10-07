"""TRACE benchmark: task contracts, prompt adapter and the official metrics."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping

from fuzzywuzzy import fuzz
from rouge import Rouge
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a

TRACE_ORDER_1 = ("C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten")

TRACE_TASKS: dict[str, dict[str, str]] = {
    "C-STANCE": {"metric": "accuracy", "instruction": "Classify the stance expressed in the input."},
    "FOMC": {"metric": "accuracy", "instruction": "Answer the financial multiple-choice question."},
    "MeetingBank": {"metric": "rouge_l_f1", "instruction": "Produce the requested meeting summary."},
    "Py150": {"metric": "edit_similarity", "instruction": "Complete the code exactly as required."},
    "ScienceQA": {"metric": "scienceqa_accuracy", "instruction": "Answer the science question."},
    "NumGLUE-cm": {"metric": "accuracy", "instruction": "Solve the numerical reasoning problem."},
    "NumGLUE-ds": {"metric": "accuracy", "instruction": "Solve the numerical reasoning problem."},
    "20Minuten": {"metric": "sari", "instruction": "Simplify the requested German news text."},
}

_INPUT_FIELDS = ("input_text", "input", "prompt", "text", "question", "source")
_TARGET_FIELDS = ("target_text", "target", "output", "answer", "label", "response", "ground_truth")
_ROUGE_L = Rouge(metrics=["rouge-l"])
_SARI_TOKENIZER = Tokenizer13a()


def _field(record: Mapping[str, object], fields: tuple[str, ...], kind: str) -> str:
    for field in fields:
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(f"TRACE record has no non-empty {kind}; expected one of {fields}")


def trace_metric(task_name: str) -> str:
    return TRACE_TASKS[task_name]["metric"]


def trace_prompt(task_name: str, record: Mapping[str, object]) -> str:
    """Official TRACE exports already store the full instruction in ``prompt``; keep it verbatim."""

    raw_input = _field(record, _INPUT_FIELDS, "input field")
    if record.get("prompt") is not None and str(record["prompt"]).strip():
        return raw_input
    instruction = TRACE_TASKS[task_name]["instruction"]
    return f"Task: {task_name}\nInstruction: {instruction}\nInput:\n{raw_input}\nResponse:"


def trace_target(record: Mapping[str, object]) -> str:
    return _field(record, _TARGET_FIELDS, "target field")


def trace_source(record: Mapping[str, object]) -> str:
    """Unwrapped source text required by the SARI metric."""

    return _field(record, _INPUT_FIELDS, "input field")


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _py150_postprocess(code: str) -> str:
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    for kind, literal in re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S).findall(code):
        code = code.replace(f"<{kind}_LIT:{literal}>", literal)
    return code


def _ngrams(tokens: list[str], size: int) -> list[str]:
    return [" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)]


def _sari_ngram_score(source: list[str], prediction: list[str], reference: list[str]) -> tuple[float, float, float]:
    """Single-reference form of Hugging Face ``datasets``' SARI: (keep F1, delete precision, add F1)."""

    source_counts, prediction_counts, reference_counts = Counter(source), Counter(prediction), Counter(reference)

    predicted_keep = source_counts & prediction_counts
    correct_keep = predicted_keep & reference_counts
    reference_keep = source_counts & reference_counts
    keep_precision = keep_recall = 1.0
    if predicted_keep:
        keep_precision = sum(correct_keep[g] / predicted_keep[g] for g in correct_keep) / len(predicted_keep)
    if reference_keep:
        keep_recall = sum(correct_keep.values()) / sum(reference_keep.values())
    keep = (
        2 * keep_precision * keep_recall / (keep_precision + keep_recall)
        if keep_precision > 0 or keep_recall > 0
        else 0.0
    )

    predicted_delete = source_counts - prediction_counts
    correct_delete = predicted_delete - reference_counts
    delete_precision = 1.0
    if predicted_delete:
        delete_precision = sum(correct_delete[g] / predicted_delete[g] for g in correct_delete) / len(predicted_delete)

    predicted_add = set(prediction_counts) - set(source_counts)
    correct_add = predicted_add & set(reference_counts)
    reference_add = set(reference_counts) - set(source_counts)
    add_precision = len(correct_add) / len(predicted_add) if predicted_add else 1.0
    add_recall = len(correct_add) / len(reference_add) if reference_add else 1.0
    add = 2 * add_precision * add_recall / (add_precision + add_recall) if add_precision > 0 or add_recall > 0 else 0.0
    return keep, delete_precision, add


def sari(source: str, prediction: str, target: str) -> float:
    """Official single-reference SARI on ``[0, 1]``."""

    tokens = [_SARI_TOKENIZER(text.lower()).split(" ") for text in (source, prediction, target)]
    scores = [_sari_ngram_score(*(_ngrams(t, size) for t in tokens)) for size in range(1, 5)]
    keep, delete, add = (sum(score[i] for score in scores) / 4 for i in range(3))
    return (keep + delete + add) / 3


def score_trace(prediction: str, target: str, metric: str, source: str | None = None) -> float:
    """Score one prediction with a TRACE metric (``exact_match`` is the generic lower-cased comparison)."""

    if metric == "accuracy":
        return float(bool(prediction and target) and prediction == target)
    if metric == "exact_match":
        return float(normalize_text(prediction) == normalize_text(target))
    if metric == "scienceqa_accuracy":
        return float(prediction[:1] == target[:1] and bool(target))
    if metric == "rouge_l_f1":
        if not prediction or not target:
            return 0.0
        return float(_ROUGE_L.get_scores(target, prediction, avg=True)["rouge-l"]["f"])
    if metric == "edit_similarity":
        if not prediction or not target:
            return 0.0
        return fuzz.ratio(_py150_postprocess(prediction), _py150_postprocess(target)) / 100.0
    if metric == "sari":
        if source is None:
            raise ValueError("SARI requires the original TRACE source text")
        return sari(source, prediction, target)
    raise ValueError(f"unsupported TRACE metric: {metric}")
