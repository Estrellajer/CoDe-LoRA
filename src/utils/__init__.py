"""
Utility functions module.

Provides common functionality for sentence embeddings, evaluation, timing, etc.
"""

from .embeddings import (
    encode_texts,
    extract_sentence_embeddings,
    load_embeddings,
    save_embeddings,
)
from .evaluation import (
    compute_task_metrics,
    compare_results,
    evaluate_model,
    load_predictions,
    save_predictions,
)
from .timing import Timer, log_timing

__all__ = [
    # Embeddings
    "encode_texts",
    "extract_sentence_embeddings",
    "save_embeddings",
    "load_embeddings",
    # Evaluation
    "evaluate_model",
    "compute_task_metrics",
    "save_predictions",
    "load_predictions",
    "compare_results",
    # Timing
    "Timer",
    "log_timing",
]

