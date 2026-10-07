"""CoDe-LoRA: a Consolidated shared branch (Co) plus Decoupled per-task experts (De), trained in one pass.

A single plain LoRA adapter per task is trained on top of the frozen consolidated shared branch ``Co_{t-1}``.  After the
task it is (i) frozen and registered with the task router as expert ``t`` and (ii) folded into ``Co_t`` by the rank-``r``
SVD retraction (``CoBranch.fold``: the update is projected off the column space of ``Co_{t-1}`` and mixed in with the
``sqrt`` boundary scaling), so Co never grows.  One forward/backward pass per micro-batch.

Why the expert is trained on top of ``Co_{t-1}`` and not on the bare base: increments that are folded into one shared
branch only add up when each was learned in the context of the earlier ones (independent-from-base adapters are
optimised for a model without the others and their sum collapses).  The same adapter is therefore the expert, and an
expert is only valid in the context it was trained in: at inference it is stacked on ``Co_{t-1}``.  Nothing but the
experts and the router statistics is stored; the fold is a deterministic function of the experts, so loading replays the
folds in task order and rebuilds every ``Co_{t-1}`` (and the deployed ``Co_T``).

Inference routes an example to its task expert (on its ``Co_{t-1}`` context) when the router confidence exceeds
``router.threshold`` and to the shared branch otherwise.  For learned tasks with candidate labels, ``fusion.enabled``
(at the last stage only, unless ``fusion.stages: all``) replaces generation by the PMI-calibrated Co/expert fusion of ``evaluation/fusion.py`` (a task's dev split is only
read once the task has been learned).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .. import distributed, profiling
from ..config import TaskCfg
from ..data.tasks import Example
from ..evaluation import fusion
from ..evaluation.branch_scores import label_logprobs
from ..evaluation.generate import generate_routed
from ..models import adapters
from ..training.loop import Branch, train_task
from .base import ContinualMethod, Prediction, register
from .co_branch import CoBranch
from .de_branch import DeBranch


def context_adapter_name(index: int) -> str:
    return f"context_{index:03d}"


@register("code-lora")
class CoDeLoRA(ContinualMethod):
    def setup(self) -> None:
        ctx, cfg = self.ctx, self.cfg
        self.lora_config = adapters.lora_config(ctx.base, ctx.backbone, cfg.lora)
        self.model = adapters.build_peft_model(ctx.base, self.lora_config, seed=cfg.seed).to(ctx.device)
        self.co = CoBranch(cfg, self.model)
        self.de = DeBranch(ctx, self.model, self.lora_config)
        self.contexts: dict[str, str] = {}  # task -> adapter holding Co_{t-1} (from task 2 on; task 1 has no context)
        self._fusions: dict[str, fusion.PMIFusion] = {}  # per-task dev fits, valid for the current Co / experts

    def learn_task(self, index: int, task: TaskCfg, examples: Sequence[Example]) -> dict[str, Any]:
        cfg, model = self.cfg, self.model
        expert = self.de.new_expert(index, task.name)
        if index > 1:
            self._snapshot_context(index, task.name)
        adapters.freeze(model, "shared")  # Co_{t-1}: the history the expert is trained on (zero before task 1)
        distributed.sync_adapters(model)
        history = ["shared"] if index > 1 else []
        result = train_task(
            model,
            [Branch([*history, expert], expert, rng_role="expert")],
            examples,
            self.ctx.collator,
            cfg.train,
            seed=cfg.seed + index,
            label=task.name,
        )
        with profiling.phase("consolidation"):
            consolidation = self.co.fold(index, expert)
            distributed.sync_adapters(model)
        with profiling.phase("routing_registration"):
            self.de.register(index, task.name, examples)
            adapters.freeze(model, expert)
        self._fusions.clear()
        return {"steps": result.steps, "loss": result.loss[0], "consolidation": consolidation}

    def _snapshot_context(self, index: int, task_name: str) -> None:
        name = context_adapter_name(index)
        adapters.add_adapter(self.model, name, self.lora_config, self.cfg.seed, index, "context")
        adapters.write_factors(self.model, name, adapters.effective_factors(self.model, "shared"))
        adapters.freeze(self.model, name)
        self.contexts[task_name] = name

    # ------------------------------------------------------------ inference

    def serving_adapters(self, task: str) -> str | tuple[str, ...]:
        """The adapters active when ``task`` is served: the shared branch, or the task's expert on its context."""

        if task == "shared":
            return "shared"
        expert = self.de.experts[task]
        return (self.contexts[task], expert) if task in self.contexts else expert

    def _serve(self, examples: Sequence[Example]) -> tuple[list[tuple[str, float]], list[str], list[Any]]:
        """Router decisions, the selected route per example (a task or ``shared``) and the adapters serving it."""

        with profiling.phase("route"):
            routes = self.de.route(examples)
        selected = [task if confidence > self.cfg.router.threshold else "shared" for task, confidence in routes]
        return routes, selected, [self.serving_adapters(task) for task in selected]

    def _fuse_now(self) -> bool:
        """Fusion is on, and (``fusion.stages: final``) the evaluation is of the last stage or of the final model."""

        fusion_cfg, stage = self.cfg.fusion, self.ctx.stage
        return fusion_cfg.enabled and (fusion_cfg.stages == "all" or stage is None or stage >= len(self.cfg.tasks))

    def predict(self, examples: Sequence[Example]) -> list[Prediction]:
        learned = (
            bool(examples) and examples[0].task in self.de.experts
        )  # unseen tasks (forward transfer) are generated
        labels = self.ctx.benchmark.labels(examples[0].task) if self._fuse_now() and learned else None
        if labels is not None:
            return self._predict_fused(examples, labels)
        ctx = self.ctx
        routes, selected, names = self._serve(examples)
        texts = [e.input_text for e in examples]
        with profiling.phase("generate"):
            outputs = generate_routed(self.model, ctx.tokenizer, ctx.backbone, texts, names, ctx.generation)
        return [
            Prediction(text, route, confidence)
            for text, route, (_, confidence) in zip(outputs, selected, routes, strict=True)
        ]

    # ------------------------------------------------------------ PMI fusion (fusion.enabled)

    def _scores(self, examples: Sequence[Example], labels: Sequence[str]) -> tuple[Any, Any, list, list]:
        """Candidate log-probabilities under the shared branch and under the routed expert, plus the routing."""

        ctx = self.ctx
        routes, selected, names = self._serve(examples)
        args = (self.model, ctx.backbone, ctx.collator, examples, labels)
        batch = self.cfg.eval.score_batch_size or self.cfg.eval.batch_size
        with profiling.phase("label_scores"):
            co = label_logprobs(*args, ["shared"] * len(examples), batch)
            expert = label_logprobs(*args, names, batch)
        return co, expert, routes, selected

    def _fit_fusion(self, task: str, labels: Sequence[str]) -> fusion.PMIFusion:
        """Priors and alpha of ``task`` from its dev split (fitted once per state of the model)."""

        if task not in self._fusions:
            dev = self.ctx.benchmark.dev(task)
            co, expert, _, _ = self._scores(dev, labels)
            gold = [labels.index(e.target_text.strip()) for e in dev]
            self._fusions[task] = fusion.fit(co, expert, gold, self.cfg.fusion.alphas)
        return self._fusions[task]

    def _predict_fused(self, examples: Sequence[Example], labels: Sequence[str]) -> list[Prediction]:
        rule = self._fit_fusion(examples[0].task, labels)
        co, expert, routes, selected = self._scores(examples, labels)
        best = rule.predict(co, expert)
        return [
            Prediction(
                labels[i],
                route,
                confidence,
                {"fusion": {"alpha": rule.alpha, "co": co[n].tolist(), "expert": expert[n].tolist()}},
            )
            for n, (i, route, (_, confidence)) in enumerate(zip(best, selected, routes, strict=True))
        ]

    # ------------------------------------------------------------ checkpoint

    def state_dict(self) -> dict[str, Any]:
        return self.de.state_dict()  # experts + router statistics; the Co branch and the contexts are replayed

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.de.load_state_dict(state)
        self._fusions.clear()
        self.replay()

    def replay(self) -> None:
        """Rebuild the shared branch and the per-task contexts from the experts.

        ``shared`` must still be in its initial state (zero update); the folds then run in task order exactly as they
        did at the end of every task.
        """

        for index, (task, expert) in enumerate(self.de.experts.items(), start=1):
            if index > 1:
                self._snapshot_context(index, task)
            self.co.fold(index, expert)
