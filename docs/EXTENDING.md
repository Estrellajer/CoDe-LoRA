# Extending CoDe-LoRA

Everything below plugs into the existing loop: you never edit `runner.py` or `training/loop.py`. Each recipe lists the
files to touch; the method recipe is exercised by `tests/test_extending.py`.

## (a) A continual-learning method

A method learns tasks one after the other and predicts without being told the task. Subclass `ContinualMethod`,
register it, and implement five hooks. `ctx` carries the frozen base model, tokenizer, backbone description, device,
collator and the run's `Benchmark`.

```python
# codelora/methods/anchored.py
from collections.abc import Sequence
from typing import Any

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from .. import distributed
from ..config import TaskCfg
from ..data.tasks import Example
from ..evaluation.generate import generate_routed
from ..models import adapters
from ..training.loop import Branch, train_task
from .base import ContinualMethod, Prediction, register


@register("anchored-lora")  # `method: anchored-lora` in a config
class AnchoredLoRA(ContinualMethod):
    """Sequential LoRA pulled towards the update it had after the previous task."""

    def setup(self) -> None:  # build the adapter structure on ctx.base
        ctx = self.ctx
        self.lora_config = adapters.lora_config(ctx.base, ctx.backbone, self.cfg.lora)
        self.model = adapters.build_peft_model(ctx.base, self.lora_config, seed=self.cfg.seed).to(ctx.device)
        self.anchor = None

    def penalty(self):  # extra loss term of the branch, called every forward pass
        def penalty() -> torch.Tensor:
            total = 0.0
            for name, layer in adapters.lora_layers(self.model, "shared"):
                update = float(layer.scaling["shared"]) * layer.lora_B["shared"].weight @ layer.lora_A["shared"].weight
                total = total + (update - self.anchor[name]).pow(2).sum()
            return total

        return penalty

    def learn_task(self, index: int, task: TaskCfg, examples: Sequence[Example]) -> dict[str, Any]:
        distributed.sync_adapters(self.model)  # identical adapter weights on every rank before training
        branch = Branch(
            ["shared"], "shared", penalty=None if self.anchor is None else self.penalty(), rng_role="shared"
        )
        result = train_task(
            self.model,
            [branch],
            examples,
            self.ctx.collator,
            self.cfg.train,
            seed=self.cfg.seed + index,
            label=task.name,
        )
        distributed.sync_adapters(self.model)
        self.anchor = {n: b @ a for n, (b, a) in adapters.effective_factors(self.model, "shared").items()}
        return {"steps": result.steps, "loss": result.loss[0], "penalty": result.penalty[0]}  # logged in summary.json

    def predict(self, examples: Sequence[Example]) -> list[Prediction]:
        ctx = self.ctx
        texts = [e.input_text for e in examples]
        names = ["shared"] * len(texts)  # per example: an adapter name, or a tuple of names to stack
        outputs = generate_routed(self.model, ctx.tokenizer, ctx.backbone, texts, names, ctx.generation)
        return [Prediction(text) for text in outputs]  # Prediction(text, route=None, confidence=None, extra=None)

    def state_dict(self) -> dict[str, Any]:  # everything predict() needs after load_state_dict()
        return {"shared": get_peft_model_state_dict(self.model, adapter_name="shared")}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        set_peft_model_state_dict(self.model, state["shared"], adapter_name="shared")
```

Then wire it in:

1. import the module in `codelora/methods/__init__.py` (`from . import anchored`), which is what registers it;
2. add `configs/methods/anchored-lora.yaml` (`method: anchored-lora`, `lora: {dropout: 0.1}`, plus your constants);
3. run it like any method: `python -m codelora.train orders/standard-order1 protocols/standard/t5-large methods/anchored-lora`.

New hyper-parameters need a typed home: add a dataclass to `config.py`, a field on `Config`, and set it in the method
fragment (`anchored: {weight: 1.0}`); unknown YAML keys are rejected, so a typo cannot silently fall back to a default.

**The `Branch` list.** `train_task(model, branches, ...)` trains one or more branches on the same batches. A
`Branch(adapters, trainable, penalty, rng_role)` says which adapters are active in the forward pass (`adapters`, e.g.
`["shared", "expert_002_beta"]` for a frozen history plus a new adapter), which one receives gradients (`trainable`),
an optional regulariser added to the loss, and the name of an isolated dropout stream. Branches get independent
optimizers and one optimizer update each per iteration; set the class attribute `branches = len(branch_list)` so that
`train.max_steps` (optimizer updates) is split correctly.

**Building blocks.** A frozen expert per task and a router: `DeBranch` (`methods/de_branch.py`: `new_expert`,
`register`, `route`, `state_dict`). A fixed-rank consolidated adapter: `CoBranch.fold` (`methods/co_branch.py`). Growing
rank blocks: `models/growing.py`. Penalties: `methods/regularizers.py`. Candidate-label scoring:
`evaluation/branch_scores.py`. Adapter factors as dense `(B, A)` pairs: `adapters.effective_factors` / `write_factors`.

**State.** `state_dict` must be enough to rebuild the deployed model on a fresh `setup()`; the runner stores it in
`checkpoint.pt` and `python -m codelora.evaluate <run> --reload` re-scores from it. CoDe-LoRA stores only the experts and
replays its deterministic folds on load; do the same if your state is a function of what you store.

## (b) A backbone

1. Add an entry to `BACKBONES` in `codelora/models/backbone.py`:
   `Backbone(name, seq2seq, default_targets, task_type)`, where `default_targets` are the module-name suffixes that get
   LoRA (`("q", "v")` for T5, `("q_proj", "v_proj")` for most decoders; `("auto",)` plus a branch in `lora_targets` for
   models whose attention layers need discovery) and `task_type` is the PEFT task type (`SEQ_2_SEQ_LM` / `CAUSAL_LM`).
   `load_backbone` already loads any `AutoModelForSeq2SeqLM` / `AutoModelForCausalLM` checkpoint in bf16 from a local
   directory and configures padding, truncation and EOS for causal models; a model with a special loader (as
   Qwen3.5) gets one more `if name == ...` branch there.
2. Add one protocol file per benchmark family: `configs/protocols/{standard,long,trace}/<backbone>.yaml`, copying an
   existing one: `backbone: {name, path}`, micro batch size and accumulation (effective batch 64), evaluation batch
   sizes, the router.
3. Run it: `python -m codelora.train orders/standard-order1 protocols/standard/<backbone> methods/code-lora`.

To try a backbone without touching the repo, override on the command line: `backbone.name=qwen3-0.6b
backbone.path=/path/to/other-qwen lora.target_modules=[q_proj,k_proj,v_proj]`.

## (c) A benchmark or task order

A task order is a YAML fragment under `configs/orders/` with the data root and the ordered tasks:

```yaml
# configs/orders/my-order.yaml
data: {root: ${MY_DATA:-data}/my-benchmark}
tasks:
  - {name: sentiment, path: SC/sentiment}                  # adapter: generic (default)
  - {name: topics,    path: TC/topics}
  - {name: summaries, path: summaries, adapter: trace}     # open generation with its own metric
```

and is combined with an existing protocol: `python -m codelora.train orders/my-order protocols/standard/t5-large
methods/code-lora`.

* **Classification tasks** (`adapter: generic`): a directory with `train.json`, `dev.json`, `test.json` (lists of
  `{"sentence": ..., "label": ...}`) and `labels.json` (the candidate answers, shown in the prompt and used by the
  candidate-label fusion). The parent directory name is the task family that selects the instruction of the official
  prompt (`TC`, `SC`, `NLI`, ...; extend `OFFICIAL_INSTRUCTIONS` in `data/tasks.py` for a new family). The dev split must
  be disjoint from test; `python -m codelora.data.prepare` shows how it is derived for the shipped benchmark.
* **Generation tasks** (`adapter: trace`): TRACE-style `prompt` / `answer` records, scored with the metric
  `trace_metric(name)` returns (`data/trace.py`). Add the task there to give it a metric.
* **A new record format or metric**: `load_examples` in `data/tasks.py` turns a split into `Example(input_text,
  target_text, task, metric, source, index)` objects, and `score_prediction` maps a prediction to a score in `[0, 1]`;
  add an adapter branch to the former and a metric to the latter, then use `adapter: <yours>` in the order file
  (`TaskCfg.adapter` is passed through).

## (d) A router

A router sees the frozen base model's pooled hidden state of an input (`routing/embed.py`) and returns a task and a
confidence. It learns per-task statistics from a few support embeddings and must be able to save and restore them:

```python
# codelora/routing/nearest.py
import torch

from .router import Router, register_router


@register_router("nearest-mean")  # `router: {kind: nearest-mean}` in a config
class NearestMeanRouter(Router):
    seed_offset = 4000  # offsets the seed of the support draw, so routers draw independent supports

    def __init__(self, cfg) -> None:  # cfg is the RouterCfg (config.py)
        super().__init__(cfg)
        self.means: dict[str, torch.Tensor] = {}

    @property
    def support_samples(self) -> int:  # support examples per task; typically a RouterCfg field
        return self.cfg.prototype_samples

    def add_task(self, task: str, support: torch.Tensor) -> None:  # support: [n, hidden]
        self.means[task] = support.mean(dim=0)

    def route(self, embedding: torch.Tensor) -> tuple[str, float]:  # one [hidden] embedding
        distances = {t: float((embedding - m).norm()) for t, m in self.means.items()}
        task = min(distances, key=distances.get)
        return task, 1.0 / (1.0 + distances[task])

    def state_dict(self):
        return {"means": self.means}

    def load_state_dict(self, state) -> None:
        self.means = state["means"]
```

Import the module in `routing/router.py::build_router` (next to `lda, prototype`) so that it is registered, and select it
with `router.kind=nearest-mean` (or a `configs/routers/*.yaml` fragment). `DeBranch` uses it for both De-LoRA and
CoDe-LoRA: De-LoRA always follows the router, CoDe-LoRA follows it when `confidence > router.threshold` and falls back to
the consolidated shared branch otherwise, so choose a confidence scale to match (the cosine router returns a cosine
similarity, the LDA router a posterior). New router hyper-parameters go on `RouterCfg`.
