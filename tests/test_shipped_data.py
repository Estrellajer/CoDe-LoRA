"""The data shipped in ``data/`` (repaired classification benchmark and the TRACE task files) is complete and intact."""

import hashlib
import json

import pytest

from codelora.config import CONFIG_ROOT, REPO_ROOT, load_config
from codelora.data.tasks import load_examples

DATA = REPO_ROOT / "data"
ORDERS = sorted(p.stem for p in (CONFIG_ROOT / "orders").glob("*.yaml"))


def _check_sums(root):
    lines = (root / "SHA256SUMS").read_text().splitlines()
    assert lines
    for line in lines:
        digest, name = line.split(maxsplit=1)
        assert hashlib.sha256((root / name.strip().lstrip("*")).read_bytes()).hexdigest() == digest, name


def test_checksums_of_the_shipped_files():
    _check_sums(DATA / "CL_Benchmark_repaired")
    _check_sums(DATA / "TRACE")


def test_manifest_counts_match_the_files():
    manifest = json.loads((DATA / "CL_Benchmark_repaired" / "manifest.json").read_text())
    assert len(manifest["datasets"]) == 15
    for entry in manifest["datasets"]:
        directory = DATA / "CL_Benchmark_repaired" / entry["dataset"]
        for split, count in entry["output"]["counts"].items():
            assert len(json.loads((directory / f"{split}.json").read_text())) == count, (entry["dataset"], split)
        labels = json.loads((directory / "labels.json").read_text())
        assert labels


@pytest.mark.parametrize("order", ORDERS)
def test_every_task_of_every_order_loads_with_the_default_roots(order, monkeypatch):
    for name in ("CODELORA_DATA", "CODELORA_TRACE"):
        monkeypatch.delenv(name, raising=False)
    cfg = load_config([f"orders/{order}"])
    for task in cfg.tasks:
        directory = DATA.parent / cfg.data.root if not cfg.data.root.startswith("/") else None
        directory = (directory or REPO_ROOT / cfg.data.root) / task.path
        splits = ("train", "eval", "test") if task.adapter == "trace" else ("train", "dev", "test")
        for split in splits:
            assert load_examples(task.name, directory, split, adapter=task.adapter, limit=4), (task.name, split)


def test_dev_and_test_share_no_record():
    root = DATA / "CL_Benchmark_repaired"

    def sentences(directory, split):
        return {r["sentence"] for r in json.loads((directory / f"{split}.json").read_text())}

    for labels in root.glob("*/*/labels.json"):
        assert not sentences(labels.parent, "dev") & sentences(labels.parent, "test"), labels.parent
