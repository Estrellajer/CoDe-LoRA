import torch
import numpy as np
import os
import json
import time
import subprocess
import csv
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

SUPPORTED_DECODER_MODELS = ['codegen', 'bloomz', 'gpt-neox', 'llama']
SUPPORTED_SEQ2SEQ_MODELS = ['t5', 'flan-t5']
ANSWER_PREFIX = "Answer:"

def check_model(model_name: str, supported_models: List[str]) -> bool:
    """Check if the model is in the supported models list."""
    for sup_model in supported_models:
        if sup_model.lower() in model_name.lower():
            return True
    return False

def skip_instructions(model: Any, predictions_ids: Union[torch.Tensor, np.ndarray, List, tuple], tokenizer: Any, ignore_idx: int = -100) -> List[str]:
    """Clean up model predictions by removing instructions and padding."""
    if isinstance(predictions_ids, tuple):
        predictions_ids = predictions_ids[0]

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    clean_ids = []
    for seq in predictions_ids:
        # Handle different prediction shapes (e.g. from generate)
        if isinstance(seq, np.ndarray) and seq.ndim > 1:
            seq = seq[0]
        elif isinstance(seq, list) and len(seq) > 0 and isinstance(seq[0], list):
            seq = seq[0]

        seq_list = seq.tolist() if hasattr(seq, "tolist") else list(seq)
        clean_ids.append([int(i) if int(i) != ignore_idx else pad_token_id for i in seq_list])

    predictions = tokenizer.batch_decode(
        clean_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    final_predictions = []
    model_name = getattr(model.config, "_name_or_path", "")
    is_decoder = check_model(model_name, SUPPORTED_DECODER_MODELS)
    
    for pred in predictions:
        if is_decoder and ANSWER_PREFIX in pred:
            splits = pred.split(ANSWER_PREFIX)
            final_predictions.append(splits[-1].strip())
        elif is_decoder:
            final_predictions.append('')
        else:
            final_predictions.append(pred)

    return final_predictions


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _safe_int(val: Any) -> Optional[int]:
    try:
        return int(val)
    except Exception:
        return None


def _bytes_to_mib(x: Optional[int]) -> Optional[float]:
    if x is None:
        return None
    try:
        return round(float(x) / (1024.0 * 1024.0), 4)
    except Exception:
        return None


def _torch_cuda_snapshot() -> Dict[str, Any]:
    """Per-process GPU memory snapshot via torch.cuda (no external deps)."""
    snap: Dict[str, Any] = {
        "available": bool(torch.cuda.is_available()),
    }
    if not torch.cuda.is_available():
        return snap

    try:
        device_count = int(torch.cuda.device_count())
    except Exception:
        device_count = 0

    snap["device_count"] = device_count
    snap["current_device"] = _safe_int(torch.cuda.current_device()) if device_count > 0 else None
    snap["devices"] = []

    for d in range(device_count):
        try:
            name = torch.cuda.get_device_name(d)
        except Exception:
            name = None
        try:
            free_b, total_b = torch.cuda.mem_get_info(d)
        except Exception:
            free_b, total_b = None, None
        try:
            allocated_b = torch.cuda.memory_allocated(d)
            reserved_b = torch.cuda.memory_reserved(d)
            max_allocated_b = torch.cuda.max_memory_allocated(d)
            max_reserved_b = torch.cuda.max_memory_reserved(d)
        except Exception:
            allocated_b = reserved_b = max_allocated_b = max_reserved_b = None

        snap["devices"].append(
            {
                "index": d,
                "name": name,
                "mem_total_mib": _bytes_to_mib(total_b),
                "mem_free_mib": _bytes_to_mib(free_b),
                "torch_allocated_mib": _bytes_to_mib(allocated_b),
                "torch_reserved_mib": _bytes_to_mib(reserved_b),
                "torch_max_allocated_mib": _bytes_to_mib(max_allocated_b),
                "torch_max_reserved_mib": _bytes_to_mib(max_reserved_b),
            }
        )

    return snap


def _nvidia_smi_snapshot(timeout_s: float = 2.0) -> Dict[str, Any]:
    """System-level GPU snapshot via nvidia-smi (best-effort).

    Returns:
      {"ok": bool, "gpus": [...], "error": "..."} where gpus is a list of dicts.
    """
    query_fields = [
        "index",
        "uuid",
        "name",
        "memory.total",
        "memory.used",
        "memory.free",
        "utilization.gpu",
        "utilization.memory",
        "temperature.gpu",
        "power.draw",
        "power.limit",
    ]
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(query_fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout_s)
        text = out.decode("utf-8", errors="replace").strip()
        if not text:
            return {"ok": False, "gpus": [], "error": "empty nvidia-smi output"}

        rows = list(csv.reader(text.splitlines()))
        gpus: List[Dict[str, Any]] = []
        for row in rows:
            # nvidia-smi returns a stable number of columns; still guard.
            if len(row) < len(query_fields):
                continue
            row = [c.strip() for c in row]
            rec: Dict[str, Any] = {}
            for k, v in zip(query_fields, row):
                rec[k] = v

            # Normalize key names + cast numeric fields where possible
            gpus.append(
                {
                    "index": _safe_int(rec.get("index")),
                    "uuid": rec.get("uuid"),
                    "name": rec.get("name"),
                    "mem_total_mib": _safe_int(rec.get("memory.total")),
                    "mem_used_mib": _safe_int(rec.get("memory.used")),
                    "mem_free_mib": _safe_int(rec.get("memory.free")),
                    "util_gpu_pct": _safe_int(rec.get("utilization.gpu")),
                    "util_mem_pct": _safe_int(rec.get("utilization.memory")),
                    "temp_c": _safe_int(rec.get("temperature.gpu")),
                    "power_w": float(rec.get("power.draw")) if rec.get("power.draw") not in (None, "", "N/A") else None,
                    "power_limit_w": float(rec.get("power.limit")) if rec.get("power.limit") not in (None, "", "N/A") else None,
                }
            )

        return {"ok": True, "gpus": gpus}
    except Exception as exc:
        return {"ok": False, "gpus": [], "error": str(exc)}


def gpu_snapshot(include_nvidia_smi: bool = False) -> Dict[str, Any]:
    """Collect a best-effort GPU usage snapshot for logging/diagnostics."""
    snap: Dict[str, Any] = {
        "time": _now_iso(),
        "pid": os.getpid(),
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "RANK": os.environ.get("RANK"),
            "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
            "SLURM_PROCID": os.environ.get("SLURM_PROCID"),
        },
        "torch_cuda": _torch_cuda_snapshot(),
    }
    if include_nvidia_smi:
        snap["nvidia_smi"] = _nvidia_smi_snapshot()
    return snap


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append a dict as one JSON line (best-effort, creates parent dir)."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Keep training robust; logging must never crash the run.
        return

