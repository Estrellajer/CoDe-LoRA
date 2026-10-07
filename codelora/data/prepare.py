"""Build the repaired classification benchmark: train-derived dev splits, test kept byte-identical.

``python -m codelora.data.prepare --input-root CL_Benchmark --output-root CL_Benchmark_repaired``

The shipped ``dev.json`` of the original benchmark equals its ``test.json``, so a dev
score would leak the test set.  Every dataset therefore gets a new dev split carved out
of its training data by a label-stratified split ranked with SHA-256 (deterministic for a
seed), and ``test.json`` is copied byte for byte.  The one label outside ``labels.json``
(MNLI ``-``) is filtered from training and listed in ``test_exclusions.json`` for test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SPLIT_FILES = ("train.json", "dev.json", "test.json")
MNLI_INVALID_LABEL = "-"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _is_mnli(relative_dir: Path) -> bool:
    return tuple(relative_dir.parts[-2:]) == ("NLI", "MNLI")


def _partition_labels(
    records: list[dict], labels: list[str], relative_dir: Path, path: Path
) -> tuple[list[dict], list[dict]]:
    """``(kept records, rejected)``: only MNLI's ``-`` may fall outside ``labels.json``."""

    allowed = set(labels)
    kept, rejected = [], []
    for index, record in enumerate(records):
        label = record["label"]
        if label in allowed:
            kept.append(record)
        elif _is_mnli(relative_dir) and label == MNLI_INVALID_LABEL:
            rejected.append({"index": index, "label": label, "reason": "label_not_in_labels_json"})
        else:
            raise ValueError(f"{path} record {index} has unknown label {label!r}")
    return kept, rejected


def _stable_rank(record: dict, index: int, seed: int, dataset: str) -> bytes:
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{seed}\0{dataset}\0{index}\0{canonical}".encode()).digest()


def stratified_split(
    records: list[dict], labels: list[str], dev_fraction: float, seed: int, dataset: str
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Per label, the ``dev_fraction`` of records with the smallest SHA-256 rank go to dev."""

    by_label: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_label[record["label"]].append(index)
    dev_indices: set[int] = set()
    dev_counts: dict[str, int] = {}
    for label in labels:
        indices = by_label.get(label, [])
        count = 0 if len(indices) < 2 else min(max(round(len(indices) * dev_fraction), 1), len(indices) - 1)
        ranked = sorted(indices, key=lambda i: _stable_rank(records[i], i, seed, dataset))
        dev_indices.update(ranked[:count])
        dev_counts[label] = count
    train = [r for i, r in enumerate(records) if i not in dev_indices]
    dev = [r for i, r in enumerate(records) if i in dev_indices]
    return train, dev, dev_counts


def _label_counts(records: list[dict], labels: list[str]) -> dict[str, int]:
    counts = Counter(r["label"] for r in records)
    return {label: counts.get(label, 0) for label in labels}


def _prepare_dataset(
    source: Path, destination: Path, relative_dir: Path, dev_fraction: float, seed: int
) -> dict[str, Any]:
    labels_bytes = (source / "labels.json").read_bytes()
    labels = json.loads(labels_bytes)
    raw = {name: (source / name).read_bytes() for name in SPLIT_FILES}
    payloads = {name: json.loads(data) for name, data in raw.items()}
    train, filtered_train = _partition_labels(payloads["train.json"], labels, relative_dir, source / "train.json")
    _, test_exclusions = _partition_labels(payloads["test.json"], labels, relative_dir, source / "test.json")
    dataset = relative_dir.as_posix()
    out_train, out_dev, dev_counts = stratified_split(train, labels, dev_fraction, seed, dataset)

    destination.mkdir(parents=True)
    _write_json(destination / "train.json", out_train)
    _write_json(destination / "dev.json", out_dev)
    (destination / "test.json").write_bytes(raw["test.json"])
    (destination / "labels.json").write_bytes(labels_bytes)
    if test_exclusions:
        policy = "preserve_test_bytes_exclude_indices_during_evaluation"
        _write_json(
            destination / "test_exclusions.json",
            {"policy": policy, "source_file": "test.json", "excluded_records": test_exclusions},
        )
    return {
        "dataset": dataset,
        "source": {
            "counts": {name.removesuffix(".json"): len(payloads[name]) for name in SPLIT_FILES},
            "sha256": {
                name: hashlib.sha256(data).hexdigest()
                for name, data in sorted({**raw, "labels.json": labels_bytes}.items())
            },
        },
        "output": {
            "counts": {"train": len(out_train), "dev": len(out_dev), "test": len(payloads["test.json"])},
            "label_counts": {"train": _label_counts(out_train, labels), "dev": _label_counts(out_dev, labels)},
            "sha256": {p.name: _sha256_file(p) for p in sorted(destination.iterdir()) if p.is_file()},
        },
        "split": {"dev_counts_by_label": dev_counts, "train_dev_source_index_overlap": 0},
        "invalid_labels": {
            "train": {"action": "filtered_before_split", "count": len(filtered_train), "records": filtered_train},
            "test": {
                "action": "preserved_with_test_exclusions" if test_exclusions else "none_found",
                "count": len(test_exclusions),
                "records": test_exclusions,
            },
        },
    }


def prepare_benchmark(input_root: Path, output_root: Path, dev_fraction: float = 0.1, seed: int = 42) -> dict[str, Any]:
    """Prepare every dataset under ``input_root`` (a directory containing ``labels.json``) into ``output_root``."""

    source, destination = Path(input_root).resolve(), Path(output_root).resolve()
    if destination == source or source in destination.parents or destination.exists():
        raise ValueError("output root must be a new directory outside the input root")
    datasets = sorted((p.parent for p in source.rglob("labels.json")), key=lambda p: p.relative_to(source).as_posix())
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        stats = [
            _prepare_dataset(d, stage / d.relative_to(source), d.relative_to(source), dev_fraction, seed)
            for d in datasets
        ]
        manifest = {
            "schema_version": 1,
            "strategy": {
                "name": "sha256_ranked_stratified_train_dev_split",
                "seed": seed,
                "dev_fraction": dev_fraction,
                "source_dev_policy": "ignored",
                "test_policy": "byte_identical_copy",
                "mnli_invalid_label_policy": {
                    "label": MNLI_INVALID_LABEL,
                    "train": "filter_before_split",
                    "test": "preserve_and_emit_test_exclusions",
                },
            },
            "datasets": stats,
        }
        _write_json(stage / "manifest.json", manifest)
        files = sorted((p for p in stage.rglob("*") if p.is_file()), key=lambda p: p.relative_to(stage).as_posix())
        (stage / "SHA256SUMS").write_text(
            "\n".join(f"{_sha256_file(p)}  {p.relative_to(stage).as_posix()}" for p in files) + "\n", encoding="utf-8"
        )
        stage.rename(destination)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", type=Path, default=Path("CL_Benchmark"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = prepare_benchmark(args.input_root, args.output_root, args.dev_fraction, args.seed)
    train = sum(d["output"]["counts"]["train"] for d in manifest["datasets"])
    dev = sum(d["output"]["counts"]["dev"] for d in manifest["datasets"])
    print(f"Prepared {len(manifest['datasets'])} datasets at {args.output_root} (train={train}, dev={dev}).")


if __name__ == "__main__":
    main()
