"""Typed experiment configuration.

A config is one or more YAML fragments (a file path, or a name relative to ``configs/``)
deep-merged in order; a fragment may itself list ``defaults`` fragments that are merged
before its own keys.  ``per_method: {method: {...}}`` holds deviations that only apply to
that method (e.g. MoLE-CIE's smaller micro-batch), and ``key.sub=value`` overrides are
applied last.  ``${VAR}`` and ``${VAR:-default}`` are expanded in every string.
"""

from __future__ import annotations

import contextlib
import os
import re
import types
import typing
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"

# Used when the variable is not set: data and model roots live next to the package (an environment variable overrides).
ENV_DEFAULTS = {
    "CODELORA_DATA": str(REPO_ROOT / "data"),
    "CODELORA_MODELS": str(REPO_ROOT / "models"),
    "CODELORA_TRACE": str(REPO_ROOT / "data" / "TRACE"),
}


@dataclass
class BackboneCfg:
    name: str = "t5-large"  # t5-large | qwen3-0.6b | llama2-7b | qwen3.5-4b
    path: str = "${CODELORA_MODELS}/t5-large"


@dataclass
class LoraCfg:
    r: int = 8
    alpha: int = 32
    dropout: float = 0.0
    target_modules: list[str] | None = None  # None: the backbone default


@dataclass
class TaskCfg:
    name: str
    path: str  # relative to data.root
    adapter: str = "generic"  # generic | trace


@dataclass
class DataCfg:
    root: str = "${CODELORA_DATA}"
    prompt_style: str = "nlora_official"  # nlora_official | plain
    add_task_name: bool = True
    add_dataset_name: bool = True
    max_source_length: int = 512
    max_target_length: int = 50
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    # Upstream O-LoRA/N-LoRA protocol (T5 pre-truncation round trip, ``max_length`` decoding with
    # clean-up, punctuation-insensitive exact match).  None: on iff prompt_style == nlora_official.
    official_protocol: bool | None = None


@dataclass
class TrainCfg:
    lr: float = 1e-3
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    micro_batch_size: int = 32  # global batch of one forward; split across ranks under torchrun
    grad_accum: int = 2
    max_steps: int | None = None  # optimizer updates per task (branches summed); None: train `epochs`
    epochs: int = 1
    max_grad_norm: float = 1.0
    num_workers: int = 0


@dataclass
class EvalCfg:
    batch_size: int = 64  # generation batch
    score_batch_size: int | None = (
        None  # candidate-label scoring batch (fusion); None: batch_size.  The HF loss upcasts
    )
    # batch x length x vocabulary logits, so large-vocabulary decoders need a smaller one than their generation batch
    embedding_batch_size: int = 64
    matrix: str = "diagonal_final"  # full: every learned task after every task; diagonal_final: a_ii + final row
    split: str = "test"  # test | dev
    sort_by_length: bool = False  # batch generation by prompt length (faster bf16 decoding, not bit-identical)
    forward_transfer: bool = (
        False  # also score not-yet-learned tasks after every stage (FWT, against the untrained model)
    )


@dataclass
class RouterCfg:
    kind: str = "cosine"  # a registered router (routing/router.py): cosine | lda
    threshold: float = 0.75  # expert iff confidence > threshold, else the shared branch
    prototype_samples: int = 10
    lda_samples: int = 1000
    lda_shrinkage: float = 0.01
    embedding_input: str = "prompt"  # prompt | raw_source


@dataclass
class OLoraCfg:
    lambda_orth: float = 0.5
    lambda_l2: float = 0.0


@dataclass
class NLoraCfg:
    lambda_l1: float = 0.4
    reduction: str = "sum"  # sum | mean


@dataclass
class CoLoraCfg:
    """The Co branch: Co-LoRA's growing-rank N-LoRA training (``lambda_l1``, ``reduction``) and the fold of CoDe-LoRA."""

    lambda_l1: float = 0.4  # N-LoRA factor-product L1 on the block trained at each task
    reduction: str = "sum"  # sum | mean
    # CoDe-LoRA's fold Co_t = rank_r(c_t Co_{t-1} + s_t dW_t), see methods/co_branch.py (ablations: additive, none)
    scaling: str = "sqrt"  # sqrt: c_t = sqrt((t-1)/t), s_t = 1/sqrt(t) | additive: c_t = s_t = 1
    projection: str = "null_space"  # null_space: dW_t loses its component along the column space of Co_{t-1} | none


@dataclass
class FusionCfg:
    """PMI-calibrated Co/expert fusion of CoDe-LoRA's candidate-label scores (``codelora.evaluation.fusion``).

    Applies to tasks with a closed label set (``labels.json``); open generation (TRACE) is unaffected.
    """

    enabled: bool = False  # on in methods/code-lora; evaluation/no-fusion switches it off (generation)
    # final: candidate-label scoring and fusion only at the last evaluation stage (the row that defines AP); the
    # intermediate stages generate (diagonal and BWT).  all: every evaluation stage is fused.
    stages: str = "final"
    alphas: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.25, 0.5, 1.0])  # candidate Co weights
    dev_samples: int | None = None  # dev examples per task used for the priors and alpha; None: the whole dev split


@dataclass
class MoleCfg:
    num_experts: int = 4
    rank: int = 8
    top_k: int = 2
    alpha: float = 1.0
    beta: float = 1.0
    lora_scale: float = 1.0
    router_temperature: float = 1.0
    task_key_temperature: float = 0.1
    kd_temperature: float = 1.0
    router_kd_weight: float = 1.0
    key_kd_weight: float = 1.0
    task_key_supervision_weight: float = 1.0
    gate_reflection_weight: float = 1.0
    memory_budget_per_task: int = 100
    prototype_num_samples: int = 200
    theta_floor: float = 0.0
    replay_objective: str = "ce"  # ce | kd_only


@dataclass
class LogCfg:
    """What a run records besides the metrics (see docs/RESULTS.md); the heavy switches are off by default."""

    interval: int = 10  # a `train` event in log.jsonl / train.log every N optimizer steps
    stdout_interval: int = 1  # the JSON `train_step` line on standard output every N optimizer steps
    tensorboard: bool = False  # also write TensorBoard events to <run>/tb (needs the tensorboard package)
    wandb: str | None = None  # Weights & Biases project name; None: off (needs the wandb package)
    per_class: bool = True  # per-answer scores in results.json
    profile: bool = False  # phase breakdown of every task in results.json (synchronises the device at phase boundaries)


@dataclass
class Config:
    name: str = "run"
    method: str = "code-lora"
    seed: int = 42
    output_dir: str = "outputs"
    device: str = "auto"  # auto | cpu | cuda
    save_checkpoint: str = "final"  # none | final | all (every stage) | every:N
    backbone: BackboneCfg = field(default_factory=BackboneCfg)
    lora: LoraCfg = field(default_factory=LoraCfg)
    data: DataCfg = field(default_factory=DataCfg)
    tasks: list[TaskCfg] = field(default_factory=list)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    router: RouterCfg = field(default_factory=RouterCfg)
    olora: OLoraCfg = field(default_factory=OLoraCfg)
    nlora: NLoraCfg = field(default_factory=NLoraCfg)
    colora: CoLoraCfg = field(default_factory=CoLoraCfg)
    fusion: FusionCfg = field(default_factory=FusionCfg)
    mole: MoleCfg = field(default_factory=MoleCfg)
    log: LogCfg = field(default_factory=LogCfg)

    @property
    def official_protocol(self) -> bool:
        if self.data.official_protocol is not None:
            return self.data.official_protocol
        return self.data.prompt_style == "nlora_official"


# ------------------------------------------------------------------ loading

_CHOICES: dict[str, tuple] = {
    "data.prompt_style": ("nlora_official", "plain"),
    "eval.matrix": ("full", "diagonal_final"),
    "eval.split": ("test", "dev"),
    "router.embedding_input": ("prompt", "raw_source"),
    "nlora.reduction": ("sum", "mean"),
    "colora.reduction": ("sum", "mean"),
    "colora.scaling": ("sqrt", "additive"),
    "colora.projection": ("null_space", "none"),
    "fusion.stages": ("final", "all"),
    "mole.replay_objective": ("ce", "kd_only"),
}


def validate(cfg: Config) -> Config:
    """Reject misspelled enumerated values, which would otherwise silently select a default branch."""

    if isinstance(cfg.save_checkpoint, bool):  # YAML `true` / `false`
        cfg.save_checkpoint = "final" if cfg.save_checkpoint else "none"
    if not re.fullmatch(r"none|final|all|every:[1-9][0-9]*", cfg.save_checkpoint):
        raise ValueError(f"save_checkpoint must be none | final | all | every:N; got {cfg.save_checkpoint!r}")

    for dotted, allowed in _CHOICES.items():
        section, key = dotted.split(".")
        value = getattr(getattr(cfg, section), key)
        if value not in allowed:
            raise ValueError(f"{dotted} must be one of {allowed}; got {value!r}")
    return cfg


_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _VAR.sub(
            lambda m: os.environ.get(
                m.group(1), m.group(2) if m.group(2) is not None else ENV_DEFAULTS.get(m.group(1), m.group(0))
            ),
            value,
        )
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _merge(base: dict, other: dict) -> dict:
    merged = dict(base)
    for key, value in other.items():
        merged[key] = (
            _merge(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
        )
    return merged


def _resolve(source: str | Path, relative_to: Path | None = None) -> Path:
    path = Path(source)
    if path.is_file():
        return path
    for base in (relative_to, CONFIG_ROOT):
        if base is not None and (base / f"{source}.yaml").is_file():
            return base / f"{source}.yaml"
    raise FileNotFoundError(f"config fragment not found: {source}")


def _read(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    merged: dict = {}
    for fragment in raw.pop("defaults", []):
        merged = _merge(merged, _read(_resolve(fragment, path.parent)))
    return _merge(merged, raw)


def _build(cls: type, data: dict) -> Any:
    hints = typing.get_type_hints(cls)
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise KeyError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        hint = hints[name]
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if isinstance(hint, types.UnionType) or typing.get_origin(hint) is typing.Union:
            hint = args[0] if len(args) == 1 else hint
        if is_dataclass(hint):
            value = _build(hint, value)
        elif typing.get_origin(hint) is list and is_dataclass(typing.get_args(hint)[0]):
            value = [_build(typing.get_args(hint)[0], item) for item in value]
        kwargs[name] = value
    return cls(**kwargs)


def _parse(raw: str) -> Any:
    value = yaml.safe_load(raw)
    if isinstance(value, str):  # YAML reads ``5e-4`` (no dot) as a string
        with contextlib.suppress(ValueError):
            return float(value)
    return value


def _set(tree: dict, dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    for key in parents:
        tree = tree.setdefault(key, {})
    tree[leaf] = value


def load_config(sources: str | Path | Sequence[str | Path], overrides: Sequence[str] = ()) -> Config:
    """Merge config fragments in order, apply the method's ``per_method`` block, then ``a.b=value`` overrides."""

    tree: dict = {}
    for source in [sources] if isinstance(sources, str | Path) else sources:
        tree = _merge(tree, _read(_resolve(source)))
    per_method = tree.pop("per_method", {})
    tree = _merge(tree, per_method.get(tree.get("method"), {}))
    for item in overrides:
        key, _, raw = item.partition("=")
        _set(tree, key, _parse(raw))
    return validate(_build(Config, _expand(tree)))
