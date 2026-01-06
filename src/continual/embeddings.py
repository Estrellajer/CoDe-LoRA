"""
Sentence embedding related utility functions (unified scaffold version).

Note:
- These functions were originally in `src/utils/embeddings.py`, used by continual methods like De-LoRA.
- To avoid scaffold fragmentation, this provides a unified entry point; `src/utils/embeddings.py` is kept as a compatibility layer.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Perform masked mean pooling over token dimension to get sentence embeddings."""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1e-9)
    return summed / lengths


@torch.no_grad()
def encode_texts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    device: torch.device,
    batch_size: int = 8,
    max_length: int = 512,
    pooling: str = "mean",
) -> torch.Tensor:
    """
    Encode a batch of texts into normalized sentence embeddings.

    Note: Methods like De-LoRA default to using `mean` pooling (sentence embeddings), not CLS.
    The pooling parameter is kept here only for compatibility/extension; unknown values will fall back to mean.
    """
    model.eval()
    encoder = model.get_encoder() if hasattr(model, "get_encoder") else model

    pooling = (pooling or "mean").lower()
    embeddings: List[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        outputs = encoder(**inputs, output_hidden_states=True)
        hidden = outputs.last_hidden_state

        # Uniformly use sentence embeddings: mask mean pooling
        # (If someone mistakenly passes cls/first, also fall back to mean to avoid conceptual confusion)
        pooled = _mean_pool(hidden, inputs.attention_mask)

        normed = F.normalize(pooled, p=2, dim=1)
        embeddings.append(normed.cpu())

    return torch.cat(embeddings, dim=0)


def load_base_encoder(model_name_or_path: str, device: str | torch.device):
    """
    Load encoder. Supports embedding_model specified in manifest.
    """
    # Check if it's a sentence-transformers model
    if "sentence-transformers" in model_name_or_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path)
    # For encoder-decoder models like T5, load full model (encode_texts will automatically extract encoder)
    elif "t5" in model_name_or_path.lower() or "bart" in model_name_or_path.lower():
        from transformers import AutoModelForSeq2SeqLM

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    elif "llama" in model_name_or_path.lower():
        from transformers import AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        try:
            from transformers import AutoModelForSeq2SeqLM

            model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        except Exception:
            model = AutoModel.from_pretrained(model_name_or_path)

    model.to(device)
    model.eval()
    return model, tokenizer


def extract_sentence_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Iterable[Dict],
    device: torch.device,
    extract_func: callable = None,
    batch_size: int = 8,
    max_length: int = 512,
    pooling: str = "mean",
) -> torch.Tensor:
    """
    Extract sentence embeddings from dataset.
    """
    if extract_func is None:

        def extract_func(item):
            if isinstance(item, dict):
                if "sentence" in item:
                    return item["sentence"]
                elif "Instance" in item and isinstance(item["Instance"], dict):
                    return item["Instance"].get("sentence", "")
            return ""

    texts = []
    for item in dataset:
        text = extract_func(item)
        if text:
            texts.append(text)

    if not texts:
        raise ValueError("No valid text extracted from dataset")

    return encode_texts(model, tokenizer, texts, device, batch_size=batch_size, max_length=max_length, pooling=pooling)


def save_embeddings(
    embeddings: Union[torch.Tensor, np.ndarray, Dict[str, torch.Tensor]],
    file_path: Union[str, Path],
    format: str = "auto",
) -> None:
    """Save sentence embeddings to file (numpy/pickle/json)."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "auto":
        suffix = file_path.suffix.lower()
        if suffix in [".npy", ".npz"]:
            format = "numpy"
        elif suffix == ".pkl":
            format = "pickle"
        elif suffix == ".json":
            format = "json"
        else:
            format = "numpy"

    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.cpu().numpy()
    elif isinstance(embeddings, np.ndarray):
        embeddings_np = embeddings
    elif isinstance(embeddings, dict):
        if format == "json":
            embeddings_dict = {
                k: (v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in embeddings.items()
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(embeddings_dict, f, indent=2, ensure_ascii=False)
            return
        embeddings_np = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in embeddings.items()}
    else:
        raise ValueError("Unsupported embeddings type")

    if format == "numpy":
        if isinstance(embeddings_np, dict):
            np.savez(file_path, **embeddings_np)
        else:
            np.save(file_path, embeddings_np)
    elif format == "pickle":
        with open(file_path, "wb") as f:
            pickle.dump(embeddings_np, f)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_embeddings(
    file_path: Union[str, Path],
    format: str = "auto",
    return_tensor: bool = False,
) -> Union[np.ndarray, torch.Tensor, Dict[str, np.ndarray]]:
    """Load sentence embedding file (numpy/pickle/json)."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if format == "auto":
        suffix = file_path.suffix.lower()
        if suffix in [".npy", ".npz"]:
            format = "numpy"
        elif suffix == ".pkl":
            format = "pickle"
        elif suffix == ".json":
            format = "json"
        else:
            raise ValueError(f"Cannot auto-detect format: {suffix}")

    if format == "numpy":
        if file_path.suffix.lower() == ".npz":
            data = np.load(file_path)
            out = {k: data[k] for k in data.files}
        else:
            out = np.load(file_path)
    elif format == "pickle":
        with open(file_path, "rb") as f:
            out = pickle.load(f)
    elif format == "json":
        with open(file_path, "r", encoding="utf-8") as f:
            out = json.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")

    if return_tensor:
        if isinstance(out, dict):
            return {k: torch.from_numpy(v) for k, v in out.items()}
        if isinstance(out, np.ndarray):
            return torch.from_numpy(out)
    return out


