"""
Evaluation utility functions.

Provides model evaluation, prediction result saving/loading, result comparison, etc.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from compute_metrics import compute_metrics, compute_grouped_metrics


def compute_task_metrics(
    predictions: List[str],
    references: List[str],
    groups: Optional[List[str]] = None,
    xlingual: bool = False,
) -> Dict:
    """
    Compute task metrics.

    Args:
        predictions: List of predictions
        references: List of reference results
        groups: Optional grouping information (for computing grouped metrics)
        xlingual: Whether to use cross-lingual tokenizer

    Returns:
        Dictionary containing various metrics
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions and references count mismatch: {len(predictions)} vs {len(references)}"
        )

    # Compute overall metrics
    metrics = compute_metrics(predictions, references, xlingual=xlingual)

    # If grouping information is available, compute grouped metrics
    if groups is not None:
        if len(groups) != len(predictions):
            raise ValueError(
                f"Grouping information count mismatch: {len(groups)} vs {len(predictions)}"
            )
        grouped_metrics = compute_grouped_metrics(
            predictions, references, groups, xlingual=xlingual
        )
        metrics.update(grouped_metrics)

    return metrics


def save_predictions(
    predictions: List[Dict],
    file_path: Union[str, Path],
    format: str = "jsonl",
):
    """
    Save prediction results to file.

    Supported formats:
    - jsonl: One JSON object per line (recommended)
    - json: Single JSON array

    Args:
        predictions: List of prediction results, each element is a dictionary
        file_path: Save path
        format: Save format
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with open(file_path, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    elif format == "json":
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_predictions(
    file_path: Union[str, Path],
    format: str = "auto",
) -> List[Dict]:
    """
    Load prediction results from file.

    Args:
        file_path: File path
        format: File format, "auto" means auto-detect based on file extension

    Returns:
        List of prediction results
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if format == "auto":
        suffix = file_path.suffix.lower()
        if suffix == ".jsonl":
            format = "jsonl"
        elif suffix == ".json":
            format = "json"
        else:
            format = "jsonl"  # Default to jsonl

    if format == "jsonl":
        predictions = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    predictions.append(json.loads(line))
        return predictions
    elif format == "json":
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")


def evaluate_model(
    model,
    tokenizer,
    dataset: List[Dict],
    compute_metrics_func: callable = None,
    extract_input_func: callable = None,
    extract_reference_func: callable = None,
    batch_size: int = 8,
    device: Optional[str] = None,
    **generation_kwargs,
) -> Tuple[List[str], List[str], Dict]:
    """
    General model evaluation interface.

    Args:
        model: Model instance
        tokenizer: Tokenizer
        dataset: Test dataset
        compute_metrics_func: Function to compute metrics (defaults to compute_task_metrics)
        extract_input_func: Function to extract input from data items
        extract_reference_func: Function to extract reference from data items
        batch_size: Batch size
        device: Computing device
        **generation_kwargs: Other parameters passed to model.generate

    Returns:
        Tuple of (predictions, references, metrics)
    """
    if extract_input_func is None:
        def extract_input_func(item):
            if isinstance(item, dict):
                if "Instance" in item and isinstance(item["Instance"], dict):
                    instruction = item["Instance"].get("instruction", "")
                    sentence = item["Instance"].get("sentence", "")
                    if instruction and sentence:
                        return instruction.replace("{0}", sentence)
                    elif sentence:
                        return sentence
            return ""

    if extract_reference_func is None:
        def extract_reference_func(item):
            if isinstance(item, dict):
                if "Instance" in item and isinstance(item["Instance"], dict):
                    return item["Instance"].get("label", "")
            return ""

    # Extract inputs and references
    inputs = [extract_input_func(item) for item in dataset]
    references = [extract_reference_func(item) for item in dataset]

    # Generate predictions in batches
    predictions = []
    model.eval()

    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i : i + batch_size]
        batch_tokenized = tokenizer(
            batch_inputs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        if device:
            batch_tokenized = {k: v.to(device) for k, v in batch_tokenized.items()}

        with torch.no_grad():
            outputs = model.generate(
                **batch_tokenized,
                **generation_kwargs,
            )

        # Decode predictions
        batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(batch_predictions)

    # Compute metrics
    if compute_metrics_func is None:
        compute_metrics_func = compute_task_metrics

    metrics = compute_metrics_func(predictions, references)

    return predictions, references, metrics


def compare_results(
    result_files: List[Union[str, Path]],
    result_names: Optional[List[str]] = None,
    metric_keys: Optional[List[str]] = None,
) -> Dict:
    """
    Compare multiple prediction result files.

    Args:
        result_files: List of result file paths
        result_names: List of result names (defaults to file names)
        metric_keys: Metric keys to compare (defaults to all metrics)

    Returns:
        Dictionary containing comparison results
    """
    if result_names is None:
        result_names = [Path(f).stem for f in result_files]

    if len(result_names) != len(result_files):
        raise ValueError("Number of result names must match number of result files")

    results = []
    for file_path, name in zip(result_files, result_names):
        preds = load_predictions(file_path)
        # Extract predictions and references
        predictions = []
        references = []
        for item in preds:
            predictions.append(item.get("Prediction", ""))
            references.append(item.get("Reference", ""))

        if not predictions or not references:
            # Try other formats
            for item in preds:
                if "prediction" in item:
                    predictions.append(item["prediction"])
                if "reference" in item or "label" in item:
                    references.append(item.get("reference") or item.get("label", ""))

        metrics = compute_task_metrics(predictions, references)
        results.append({"name": name, "metrics": metrics, "num_samples": len(predictions)})

    # Build comparison results
    if metric_keys is None:
        # Collect all metric keys
        metric_keys = set()
        for r in results:
            metric_keys.update(r["metrics"].keys())
        metric_keys = sorted(list(metric_keys))

    comparison = {
        "results": results,
        "comparison_table": {},
    }

    # Create comparison table for each metric
    for metric_key in metric_keys:
        comparison["comparison_table"][metric_key] = {
            r["name"]: r["metrics"].get(metric_key, None) for r in results
        }

    return comparison


# Import torch for compatibility
try:
    import torch
except ImportError:
    torch = None

