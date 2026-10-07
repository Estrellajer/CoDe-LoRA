"""Download the four backbones into ``models/`` (or ``$CODELORA_MODELS``).

``python scripts/download_models.py [t5-large qwen3-0.6b llama2-7b qwen3.5-4b]``

Llama-2 is gated: accept the licence of ``meta-llama/Llama-2-7b-hf`` on the Hub and export ``HF_TOKEN`` first.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# name -> (Hub repository, directory, pinned revision of the snapshot used for the paper runs or None)
MODELS = {
    "t5-large": ("google-t5/t5-large", "t5-large", "150ebc2c4b72291e770f58e6057481c8d2ed331a"),
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", "Qwen3-0.6B", None),
    "llama2-7b": ("meta-llama/Llama-2-7b-hf", "Llama-2-7b-hf", None),
    "qwen3.5-4b": ("Qwen/Qwen3.5-4B", "Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
}

if __name__ == "__main__":
    root = Path(os.environ.get("CODELORA_MODELS", Path(__file__).resolve().parents[1] / "models"))
    if "llama2-7b" in (sys.argv[1:] or MODELS) and not os.environ.get("HF_TOKEN"):
        sys.exit(
            "Llama-2-7b-hf is gated: accept its licence on the Hub and export HF_TOKEN (or skip it by naming the others)"
        )
    for name in sys.argv[1:] or list(MODELS):
        repo, directory, revision = MODELS[name]
        print(f"{repo} -> {root / directory}")
        snapshot_download(
            repo,
            local_dir=root / directory,
            revision=revision,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "original/*", "*.gguf"],
        )
