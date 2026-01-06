"""
Sentence embedding extraction and saving utilities (compatibility layer).

Note:
- These capabilities have been migrated to unified scaffold: `src/continual/embeddings.py`
- This is kept as a compatibility import path for old scripts/tools: `from utils.embeddings import ...`
"""

from continual.embeddings import (  # noqa: F401
    encode_texts,
    load_base_encoder,
    extract_sentence_embeddings,
    save_embeddings,
    load_embeddings,
)

__all__ = [
    "encode_texts",
    "load_base_encoder",
    "extract_sentence_embeddings",
    "save_embeddings",
    "load_embeddings",
]


