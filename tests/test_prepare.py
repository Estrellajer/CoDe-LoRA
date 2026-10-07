import hashlib
import json

import pytest

from codelora.data.prepare import prepare_benchmark


def raw_benchmark(root, invalid_test_label="-"):
    def write(task, train, test, labels):
        directory = root / task
        directory.mkdir(parents=True)
        (directory / "labels.json").write_text(json.dumps(labels))
        (directory / "train.json").write_text(json.dumps(train))
        (directory / "test.json").write_text(json.dumps(test))
        (directory / "dev.json").write_text(json.dumps(test))  # the original benchmark ships dev == test

    rows = lambda n, labels, tag: [{"sentence": f"{tag}{i}", "label": labels[i % len(labels)]} for i in range(n)]  # noqa: E731
    labels = ["a", "b", "c"]
    write("TC/plain", rows(60, labels, "tr"), rows(9, labels, "te"), labels)
    mnli_train = [*rows(30, labels, "mtr"), {"sentence": "bad", "label": "-"}]
    mnli_test = [*rows(6, labels, "mte"), {"sentence": "bad", "label": invalid_test_label}]
    write("NLI/MNLI", mnli_train, mnli_test, labels)
    return root


def test_dev_is_carved_from_train_and_test_is_untouched(tmp_path):
    source = raw_benchmark(tmp_path / "raw")
    manifest = prepare_benchmark(source, tmp_path / "out")
    for task in ("TC/plain", "NLI/MNLI"):
        train = json.loads((tmp_path / "out" / task / "train.json").read_text())
        dev = json.loads((tmp_path / "out" / task / "dev.json").read_text())
        original_train = json.loads((source / task / "train.json").read_text())
        sentences = lambda records: {r["sentence"] for r in records}  # noqa: E731
        assert not sentences(train) & sentences(dev)  # zero overlap
        assert sentences(train) | sentences(dev) == sentences(original_train) - {"bad"}
        assert {r["label"] for r in dev} == {"a", "b", "c"}  # stratified: every label has dev examples
        assert (tmp_path / "out" / task / "test.json").read_bytes() == (source / task / "test.json").read_bytes()
        test_sentences = sentences(json.loads((source / task / "test.json").read_text()))
        assert not test_sentences & (sentences(train) | sentences(dev))  # dev is not the test set
    assert {d["dataset"] for d in manifest["datasets"]} == {"TC/plain", "NLI/MNLI"}


def test_mnli_invalid_label_is_filtered_from_train_and_excluded_from_test(tmp_path):
    prepare_benchmark(raw_benchmark(tmp_path / "raw"), tmp_path / "out")
    exclusions = json.loads((tmp_path / "out" / "NLI/MNLI" / "test_exclusions.json").read_text())
    assert exclusions["excluded_records"] == [{"index": 6, "label": "-", "reason": "label_not_in_labels_json"}]
    assert not (tmp_path / "out" / "TC/plain" / "test_exclusions.json").exists()


def test_split_is_deterministic_and_seed_dependent(tmp_path):
    source = raw_benchmark(tmp_path / "raw")
    prepare_benchmark(source, tmp_path / "a", seed=1)
    prepare_benchmark(source, tmp_path / "b", seed=1)
    prepare_benchmark(source, tmp_path / "c", seed=2)
    dev = lambda root: (root / "TC/plain" / "dev.json").read_bytes()  # noqa: E731
    assert dev(tmp_path / "a") == dev(tmp_path / "b") and dev(tmp_path / "a") != dev(tmp_path / "c")


def test_manifest_hashes_match_the_files(tmp_path):
    manifest = prepare_benchmark(raw_benchmark(tmp_path / "raw"), tmp_path / "out")
    for entry in manifest["datasets"]:
        for name, digest in entry["output"]["sha256"].items():
            assert hashlib.sha256((tmp_path / "out" / entry["dataset"] / name).read_bytes()).hexdigest() == digest
    assert (tmp_path / "out" / "SHA256SUMS").is_file()


def test_unknown_label_outside_mnli_is_an_error(tmp_path):
    source = raw_benchmark(tmp_path / "raw")
    train = json.loads((source / "TC/plain" / "train.json").read_text())
    (source / "TC/plain" / "train.json").write_text(json.dumps([*train, {"sentence": "x", "label": "zzz"}]))
    with pytest.raises(ValueError, match="unknown label"):
        prepare_benchmark(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()  # nothing half-written is left behind
