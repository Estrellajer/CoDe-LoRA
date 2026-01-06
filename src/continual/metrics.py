import string
import logging
from typing import List, Dict, Any, Optional, Iterable
from rouge import rouge_scorer

logger = logging.getLogger(__name__)

# Cache RougeScorer instances
_rouge1_scorer_default = None
_rougeL_scorer_default = None

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, and extra whitespace."""
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def rouge1_score(prediction: str, ground_truth: str) -> float:
    global _rouge1_scorer_default
    if _rouge1_scorer_default is None:
        _rouge1_scorer_default = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = _rouge1_scorer_default.score(prediction=prediction, target=ground_truth)
    return scores["rouge1"].fmeasure

def rougeL_score(prediction: str, ground_truth: str) -> float:
    global _rougeL_scorer_default
    if _rougeL_scorer_default is None:
        _rougeL_scorer_default = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = _rougeL_scorer_default.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure

def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]) -> float:
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def _normalize_metric_names(metrics: Optional[Iterable[str]]) -> List[str]:
    """Normalize/expand metric aliases used in configs.

    Supported:
    - exact_match / em / acc / accuracy
    - rouge / rouge1 / rougeL (rouge expands to rouge1+rougeL)
    """
    if not metrics:
        return []
    out: List[str] = []
    for m in metrics:
        if not m:
            continue
        key = str(m).strip()
        low = key.lower()
        if low in {"exact_match", "em", "acc", "accuracy"}:
            out.append("exact_match")
        elif low in {"rouge"}:
            out.extend(["rouge1", "rougeL"])
        elif low in {"rouge1"}:
            out.append("rouge1")
        elif low in {"rougel", "rouge-l", "rouge_l"}:
            out.append("rougeL")
        else:
            # Unknown metric name: keep as-is (will be ignored if not supported)
            out.append(key)

    # De-dup while preserving order
    seen = set()
    deduped: List[str] = []
    for m in out:
        if m not in seen:
            deduped.append(m)
            seen.add(m)
    return deduped


def compute_metrics(predictions: List[str], references: List[str], metrics: Optional[List[str]] = None) -> Dict[str, float]:
    """Compute selected metrics (exact_match / rouge1 / rougeL)."""
    assert len(predictions) == len(references)
    if len(predictions) == 0:
        return {"exact_match": 0.0, "rouge1": 0.0, "rougeL": 0.0}

    wanted = _normalize_metric_names(metrics)
    if not wanted:
        wanted = ["exact_match", "rouge1", "rougeL"]
        
    exact_match_total, rouge1_total, rougeL_total = 0, 0, 0
    for pred, gold in zip(predictions, references):
        gold_list = [gold]
        if "exact_match" in wanted:
            exact_match_total += metric_max_over_ground_truths(exact_match_score, prediction=pred, ground_truths=gold_list)
        if "rouge1" in wanted:
            rouge1_total += metric_max_over_ground_truths(rouge1_score, prediction=pred, ground_truths=gold_list)
        if "rougeL" in wanted:
            rougeL_total += metric_max_over_ground_truths(rougeL_score, prediction=pred, ground_truths=gold_list)
        
    n = len(references)
    out: Dict[str, float] = {}
    if "exact_match" in wanted:
        out["exact_match"] = 100.0 * exact_match_total / n
    if "rouge1" in wanted:
        out["rouge1"] = 100.0 * rouge1_total / n
    if "rougeL" in wanted:
        out["rougeL"] = 100.0 * rougeL_total / n
    return {k: round(v, 4) for k, v in out.items()}

def compute_grouped_metrics(
    predictions: List[str],
    references: List[str],
    groups: List[str],
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute metrics grouped by a task/dataset name."""
    assert len(predictions) == len(references) == len(groups)
    if len(predictions) == 0:
        return {}

    examples_by_group = {}
    for pred, gold, group in zip(predictions, references, groups):
        if group not in examples_by_group:
            examples_by_group[group] = []
        examples_by_group[group].append((pred, gold))
    
    results = {}
    for group, group_examples in examples_by_group.items():
        task_predictions, task_references = zip(*group_examples)
        group_metrics = compute_metrics(list(task_predictions), list(task_references), metrics=metrics)
        for metric, value in group_metrics.items():
            results[f"{metric}_for_{group}"] = value
    return results

