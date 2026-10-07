import itertools
from pathlib import Path

import pytest

from codelora.config import CONFIG_ROOT, REPO_ROOT, load_config
from codelora.methods import METHODS


def test_fragments_merge_in_order_and_overrides_win():
    cfg = load_config(
        ["orders/standard-order1", "protocols/standard/t5-large", "methods/code-lora"], ["seed=43", "train.lr=5e-4"]
    )
    assert (cfg.method, cfg.seed, cfg.train.lr) == ("code-lora", 43, 5e-4)
    assert [t.name for t in cfg.tasks] == ["dbpedia", "amazon", "yahoo", "agnews"]
    assert cfg.lora.dropout == 0.0 and cfg.router.kind == "lda" and cfg.official_protocol


def test_environment_expansion_with_defaults(monkeypatch):
    monkeypatch.setenv("CODELORA_MODELS", "/weights")
    monkeypatch.delenv("CODELORA_DATA", raising=False)
    cfg = load_config(["orders/long-order4", "protocols/long/llama2-7b", "methods/lora"])
    assert cfg.backbone.path == "/weights/Llama-2-7b-hf"
    assert cfg.data.root == str(REPO_ROOT / "data" / "CL_Benchmark_repaired")  # unset: next to the package


def test_data_and_models_default_to_the_repository(monkeypatch):
    for name in ("CODELORA_DATA", "CODELORA_MODELS", "CODELORA_TRACE"):
        monkeypatch.delenv(name, raising=False)
    cfg = load_config(["orders/trace", "protocols/trace/t5-large", "methods/lora"])
    assert cfg.data.root == str(REPO_ROOT / "data" / "TRACE")
    assert cfg.backbone.path == str(REPO_ROOT / "models" / "t5-large")
    monkeypatch.setenv("CODELORA_TRACE", "/elsewhere")
    assert load_config(["orders/trace", "methods/lora"]).data.root == "/elsewhere"


def test_unknown_keys_are_rejected():
    with pytest.raises(KeyError, match="unknown"):
        load_config(["methods/lora"], ["train.learning_rate=1"])


def test_defaults_in_a_file_and_direct_paths(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 5\ntrain: {lr: 0.1}\n")
    (tmp_path / "top.yaml").write_text("defaults: [base]\ntrain: {grad_accum: 3}\n")
    cfg = load_config(tmp_path / "top.yaml")
    assert (cfg.seed, cfg.train.lr, cfg.train.grad_accum) == (5, 0.1, 3)


ORDERS = sorted(p.stem for p in (CONFIG_ROOT / "orders").glob("*.yaml"))
BACKBONES = ["t5-large", "qwen3-0.6b", "llama2-7b", "qwen3.5-4b"]


def _family(order: str) -> str:
    return "standard" if order.startswith("standard") else "long" if order.startswith("long") else "trace"


@pytest.mark.parametrize(("order", "backbone"), list(itertools.product(ORDERS, BACKBONES)))
def test_every_benchmark_backbone_method_combination_loads(order, backbone):
    for method in METHODS:
        cfg = load_config([f"orders/{order}", f"protocols/{_family(order)}/{backbone}", f"methods/{method}"])
        assert cfg.method == method and cfg.tasks
        # one epoch per task, effective batch 64, for every method
        assert cfg.train.micro_batch_size * cfg.train.grad_accum == 64
        assert cfg.train.max_steps is None and cfg.train.epochs == 1
        assert Path(cfg.backbone.path).name and cfg.backbone.name == backbone


def test_task_counts_match_the_paper():
    counts = {o: len(load_config([f"orders/{o}"]).tasks) for o in ORDERS}
    assert counts == {
        "standard-order1": 4,
        "standard-order2": 4,
        "standard-order3": 4,
        "long-order4": 15,
        "long-order5": 15,
        "long-order6": 15,
        "trace": 8,
    }


@pytest.mark.parametrize(
    "override",
    [
        "eval.matrix=diag",
        "nlora.reduction=max",
        "data.prompt_style=official",
        "eval.split=val",
        "colora.scaling=linear",
        "colora.projection=orthogonal",
    ],
)
def test_misspelled_choices_are_rejected(override):
    with pytest.raises(ValueError, match="must be one of"):
        load_config(["methods/code-lora"], [override])


def test_learning_rates_follow_the_final_protocol():
    def lr(backbone, method, order="standard-order1"):
        family = "trace" if order == "trace" else order.split("-")[0]
        return load_config([f"orders/{order}", f"protocols/{family}/{backbone}", f"methods/{method}"]).train.lr

    for order in ("standard-order1", "long-order4", "trace"):
        assert all(lr(b, m, order) == 1e-4 for b in ("llama2-7b", "qwen3.5-4b") for m in METHODS)
        assert (
            all(lr("t5-large", m, order) == 1e-3 for m in METHODS if m != "lora")
            and lr("t5-large", "lora", order) == 1e-4
        )
        shared = ("lora", "o-lora", "n-lora", "co-lora", "mole-cie")
        assert all(lr("qwen3-0.6b", m, order) == 1e-4 for m in shared)
        assert lr("qwen3-0.6b", "de-lora", order) == lr("qwen3-0.6b", "code-lora", order) == 1e-3


def test_mole_cie_keeps_an_effective_batch_of_64_with_smaller_forwards():
    mole = load_config(["orders/standard-order1", "protocols/standard/qwen3-0.6b", "methods/mole-cie"])
    lora = load_config(["orders/standard-order1", "protocols/standard/qwen3-0.6b", "methods/lora"])
    assert (mole.train.micro_batch_size, mole.train.grad_accum) == (16, 4)
    assert (lora.train.micro_batch_size, lora.train.grad_accum) == (32, 2)


def test_code_lora_ships_the_final_recipe():
    cfg = load_config(["orders/standard-order1", "protocols/standard/t5-large", "methods/code-lora"])
    assert (cfg.colora.scaling, cfg.colora.projection) == ("sqrt", "null_space")
    assert cfg.lora.dropout == 0.0 and cfg.fusion.enabled and cfg.fusion.stages == "final"
    off = load_config(["protocols/standard/t5-large", "methods/code-lora", "evaluation/no-fusion"])
    assert not off.fusion.enabled
    assert not load_config(["methods/lora"]).fusion.enabled


def test_candidate_scoring_batches_of_the_large_vocabulary_decoders():
    def score_batch(backbone):
        cfg = load_config([f"protocols/standard/{backbone}", "methods/code-lora"])
        return cfg.eval.score_batch_size or cfg.eval.batch_size

    assert (score_batch("qwen3-0.6b"), score_batch("qwen3.5-4b"), score_batch("llama2-7b")) == (32, 8, 32)
    assert load_config(["methods/lora"]).olora.lambda_l2 == 0.0
    assert load_config(["orders/standard-order1", "protocols/standard/llama2-7b", "methods/lora"]).eval.sort_by_length


def test_every_protocol_defaults_to_lda_router():
    for bb in BACKBONES:
        for proto, order in (("standard", "standard-order1"), ("long", "long-order4"), ("trace", "trace")):
            try:
                cfg = load_config([f"orders/{order}", f"protocols/{proto}/{bb}", "methods/code-lora"])
            except FileNotFoundError:
                continue
            assert cfg.router.kind == "lda" and cfg.router.threshold == 0.0
    cfg = load_config(["orders/standard-order1", "protocols/standard/t5-large", "routers/cosine", "methods/code-lora"])
    assert cfg.router.kind == "cosine" and cfg.router.threshold == 0.75
