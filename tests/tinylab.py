"""Offline tiny models, tokenizers and synthetic benchmarks for CPU tests.

Nothing here touches the network: the tokenizer is a byte-level BPE trained on a
synthetic corpus, the models are randomly initialised 2-layer transformers.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
    T5Config,
    T5ForConditionalGeneration,
)

WORDS = [f"w{i}" for i in range(60)]
LABELS = {
    "TC/alpha": ["World", "Sports", "Business", "Science or Technology"],
    "SC/beta": ["positive", "negative"],
    "TC/gamma": ["cat", "dog", "bird"],
    "NLI/delta": ["neutral", "entailment", "contradiction"],
    "TC/dbpedia": ["Company", "School", "Artist"],
    "SC/amazon": ["positive", "negative"],
}


def _corpus() -> list[str]:
    rng = random.Random(0)
    lines = [" ".join(rng.choice(WORDS) for _ in range(12)) for _ in range(200)]
    lines += [label for labels in LABELS.values() for label in labels]
    lines += ["Task:TC\nDataset:alpha\nOption: a, b, c \nAnswer:", "Title: x\nText: y, z. (Reuters) -- end!"]
    return lines


def make_tokenizer(path: Path, kind: str) -> None:
    """Byte-level BPE; ``t5`` appends ``</s>`` and has a pad token, ``causal`` has neither."""

    specials = ["<pad>", "</s>", "<unk>"] if kind == "t5" else ["<|endoftext|>", "<unk>"]
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        _corpus(),
        trainers.BpeTrainer(
            vocab_size=500, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        ),
    )
    if kind == "t5":
        tok.post_processor = processors.TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
        fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="</s>", unk_token="<unk>")
    else:
        fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>", unk_token="<unk>")
    fast.save_pretrained(path)


def copy_tokenizer(source: Path, path: Path) -> None:
    """Use a real tokenizer (e.g. from a local t5-large / Qwen3 snapshot) instead of the synthetic BPE."""

    path.mkdir(parents=True, exist_ok=True)
    for file in source.iterdir():
        if file.name not in ("config.json", "generation_config.json") and not file.name.endswith(
            (".safetensors", ".bin")
        ):
            shutil.copy(file, path / file.name)


def make_tiny_t5(path: Path, dropout: float = 0.0, seed: int = 0, tokenizer: Path | None = None) -> Path:
    if tokenizer is None:
        make_tokenizer(path, "t5")
    else:
        copy_tokenizer(tokenizer, path)
    vocab = 32128 if tokenizer is not None else len(PreTrainedTokenizerFast.from_pretrained(path))
    config = T5Config(
        vocab_size=vocab, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_decoder_layers=2, num_heads=4,
        dropout_rate=dropout, pad_token_id=0, eos_token_id=1, decoder_start_token_id=0,
    )  # fmt: skip
    torch.manual_seed(seed)
    T5ForConditionalGeneration(config).save_pretrained(path)
    return path


def make_tiny_qwen(path: Path, seed: int = 0, tokenizer_dir: Path | None = None) -> Path:
    if tokenizer_dir is None:
        make_tokenizer(path, "causal")
    else:
        copy_tokenizer(tokenizer_dir, path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    config = Qwen3Config(
        vocab_size=151936 if tokenizer_dir is not None else len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=1024, eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=True,
    )  # fmt: skip
    torch.manual_seed(seed)
    Qwen3ForCausalLM(config).save_pretrained(path)
    return path


def make_benchmark(
    root: Path, tasks: tuple[str, ...] = ("TC/alpha", "SC/beta", "TC/gamma"), train: int = 48, test: int = 12
) -> Path:
    """CL_Benchmark-format directories: ``train/dev/test.json`` of ``{sentence, label}`` plus ``labels.json``."""

    rng = random.Random(1)
    for task in tasks:
        directory = root / task
        directory.mkdir(parents=True, exist_ok=True)
        labels = LABELS[task]
        (directory / "labels.json").write_text(json.dumps(labels))
        for split, size in (("train", train), ("dev", 8), ("test", test)):
            records = []
            for _ in range(size):
                words = [rng.choice(WORDS) for _ in range(rng.randint(5, 14))]
                label = labels[int(words[0][1:]) % len(labels)]
                records.append({"sentence": f"Title: {words[0]}\nText: " + " ".join(words[1:]) + ".\n", "label": label})
            (directory / f"{split}.json").write_text(json.dumps(records))
    return root


TRACE_TEXT = {
    "C-STANCE": ("accuracy", ["support", "against", "neutral"]),
    "FOMC": ("accuracy", ["A", "B", "C"]),
    "MeetingBank": ("rouge_l_f1", None),
    "Py150": ("edit_similarity", None),
    "ScienceQA": ("scienceqa_accuracy", ["A. yes", "B. no"]),
    "20Minuten": ("sari", None),
}


def make_trace_benchmark(
    root: Path,
    tasks: tuple[str, ...] = ("C-STANCE", "MeetingBank", "Py150", "ScienceQA", "20Minuten"),
    train: int = 40,
    test: int = 10,
) -> Path:
    """TRACE-format task directories (``prompt``/``answer`` records; one task uses the ``input``/``output`` schema)."""

    rng = random.Random(2)
    for name in tasks:
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        _, answers = TRACE_TEXT[name]
        for split, size in (("train", train), ("eval", 6), ("test", test)):
            records = []
            for _ in range(size):
                text = " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 16)))
                answer = (
                    rng.choice(answers) if answers else " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 8)))
                )
                if name == "Py150":
                    records.append({"input": f"def f(x):\n    return {text}", "output": answer + " <NUM_LIT>"})
                else:
                    records.append({"prompt": f"Task {name}: {text}\nAnswer:", "answer": answer})
            (directory / f"{split}.json").write_text(json.dumps(records))
    return root


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1] if len(sys.argv) > 1 else "lab")
    make_tiny_t5(target / "t5")
    make_tiny_qwen(target / "qwen")
    make_benchmark(target / "data")
    make_trace_benchmark(target / "trace")
    print(f"tiny models and benchmarks written to {target}")
