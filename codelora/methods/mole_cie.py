"""MoLE-CIE: mixture of LoRA experts with task keys, replay memory and gate reflection.

Every target linear layer is wrapped by ``MoLELinear``.  A new task adds one task
expert and key (initialised from a masked mean of the layer's own inputs on a support
set) and is trained on its data interleaved with replayed exemplars of earlier tasks;
the router and the old task keys are regularised towards a snapshot taken after the
previous task (gate reflection).  Inference is task-agnostic: each example's task is
chosen from key similarity (majority over encoder layers for T5, at prefill for decoders).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .. import distributed, tracking
from ..config import TaskCfg
from ..data.tasks import Example
from ..evaluation.generate import generate_batch
from ..models.backbone import lora_targets
from ..models.mole_layer import ExemplarMemory, MoLELinear, install_attention_mask_hook
from ..training.optim import build_adamw
from .base import ContinualMethod, Prediction, register
from .de_branch import select_support


def _set_child(model: nn.Module, qualified_name: str, replacement: nn.Module) -> None:
    parent_name, _, child = qualified_name.rpartition(".")
    setattr(model.get_submodule(parent_name) if parent_name else model, child, replacement)


def input_embeddings(
    model: nn.Module, tokenizer, texts: Sequence[str], *, batch_size: int, max_length: int
) -> torch.Tensor:
    """Masked mean of the input-embedding layer, sharded over ranks (every rank gets the same vectors)."""

    embedding = model.get_input_embeddings()
    device = next(model.parameters()).device

    def embed(shard: list[str]) -> list[torch.Tensor]:
        rows: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(shard), batch_size):
                encoded = tokenizer(
                    shard[start : start + batch_size],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                mask = encoded["attention_mask"].to(device).unsqueeze(-1)
                hidden = embedding(encoded["input_ids"].to(device))
                pooled = (hidden * mask.to(hidden.dtype)).sum(1) / mask.sum(1).clamp_min(1)
                rows.extend(row.clone() for row in pooled.detach().float().cpu().unbind(0))
        return rows

    return torch.stack(distributed.sharded_map(embed, list(texts)), dim=0)


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, int]:
    """Mean over tokens of a layer input, ignoring padding when the mask matches the layer's sequence."""

    if hidden.ndim == 1:
        return hidden.float(), 1
    if hidden.ndim == 2:
        return hidden.float().mean(dim=0), int(hidden.shape[0])
    flat = hidden.reshape(-1, hidden.shape[-1]).float()
    if mask is not None and hidden.ndim == 3 and tuple(mask.shape) == tuple(hidden.shape[:2]):
        weights = mask.to(device=hidden.device, dtype=hidden.dtype).reshape(-1)
        return (flat * weights.unsqueeze(-1)).sum(dim=0) / weights.sum().clamp_min(1.0), int(weights.sum())
    return flat.mean(dim=0), int(flat.shape[0])


@register("mole-cie")
class MoLECIE(ContinualMethod):
    def setup(self) -> None:
        ctx, m = self.ctx, self.cfg.mole
        if ctx.backbone.causal:
            install_attention_mask_hook(ctx.base)
        for parameter in ctx.base.parameters():
            parameter.requires_grad_(False)
        targets = set(lora_targets(ctx.base, ctx.backbone, self.cfg.lora.target_modules))
        matched = [
            (name, module)
            for name, module in ctx.base.named_modules()
            if isinstance(module, nn.Linear) and (name in targets or name.rsplit(".", 1)[-1] in targets)
        ]
        self.wrappers: dict[str, MoLELinear] = {}
        for name, module in matched:
            wrapper = MoLELinear(
                module,
                num_experts=m.num_experts,
                rank=m.rank,
                top_k=m.top_k,
                alpha=m.alpha,
                beta=m.beta,
                lora_scale=m.lora_scale,
                router_temperature=m.router_temperature,
                task_key_temperature=m.task_key_temperature,
                theta_floor=m.theta_floor,
            )
            _set_child(ctx.base, name, wrapper)
            self.wrappers[name] = wrapper
        self.model = ctx.base.to(ctx.device)
        self.memory = ExemplarMemory(m.memory_budget_per_task)
        distributed.sync_module_parameters(list(self.wrappers.values()))

    # ------------------------------------------------------------- training

    def _task_key_prototypes(self, examples: Sequence[Example]) -> dict[str, torch.Tensor]:
        """One prototype per wrapped layer: the masked mean of that layer's actual input on ``examples``."""

        model, device = self.model, self.ctx.device
        pending: dict[str, list[torch.Tensor]] = {name: [] for name in self.wrappers}
        totals: dict[str, torch.Tensor] = {}
        counts = dict.fromkeys(self.wrappers, 0)
        handles = []

        def capture(name: str):
            def hook(_module, args):
                pending[name].append(args[0].detach())

            return hook

        for name, wrapper in self.wrappers.items():
            handles.append(wrapper.register_forward_pre_hook(capture(name)))
        was_training = model.training
        model.eval()
        try:
            loader = DataLoader(
                list(examples),
                batch_size=self.cfg.eval.embedding_batch_size,
                shuffle=False,
                collate_fn=self.ctx.collator,
            )
            for raw in loader:
                batch = {k: v.to(device) for k, v in raw.items()}
                for values in pending.values():
                    values.clear()
                kwargs: dict[str, Any] = {}
                if not model.config.is_encoder_decoder:
                    # only the wrappers' inputs are needed: skip the loss and the batch x length x vocab logits
                    batch.pop("labels", None)
                    kwargs["logits_to_keep"] = 1
                with torch.inference_mode():
                    model(**batch, **kwargs)
                for name, values in pending.items():
                    for hidden in values:
                        vector, count = _masked_mean(hidden, batch.get("attention_mask"))
                        cpu_vector = vector.detach().float().cpu()
                        totals[name] = cpu_vector * count if name not in totals else totals[name] + cpu_vector * count
                        counts[name] += count
        finally:
            for handle in handles:
                handle.remove()
            model.train(was_training)
        return {name: totals[name] / counts[name] for name in self.wrappers}

    def _train(self, example_groups: Mapping[str, Sequence[Example]], current: str, seed: int) -> list[float]:
        cfg, m, model, device = self.cfg.train, self.cfg.mole, self.model, self.ctx.device
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = build_adamw(parameters, cfg)
        tracking.recorder().trainable_parameters = sum(p.numel() for p in parameters)
        captured: dict[str, torch.Tensor] = {}
        handles = [
            wrapper.register_forward_pre_hook(lambda _m, args, key=name: captured.__setitem__(key, args[0]))
            for name, wrapper in self.wrappers.items()
        ]
        losses: list[float] = []
        steps = micro = 0
        optimizer.zero_grad(set_to_none=True)
        epochs = itertools.count() if cfg.max_steps else range(cfg.epochs)
        try:
            for epoch in epochs:
                loaders = {
                    task: DataLoader(
                        list(examples),
                        batch_size=cfg.micro_batch_size,
                        shuffle=True,
                        collate_fn=self.ctx.collator,
                        generator=torch.Generator().manual_seed(seed + epoch * 1009 + group),
                    )
                    for group, (task, examples) in enumerate(example_groups.items())
                }
                iterators = {task: iter(loader) for task, loader in loaders.items()}
                replay_tasks = sorted(set(loaders) - {current})
                for batch_index in range(max(len(loader) for loader in loaders.values())):
                    step_tasks = [current, replay_tasks[batch_index % len(replay_tasks)]] if replay_tasks else [current]
                    for batch_task in step_tasks:
                        try:
                            raw = next(iterators[batch_task])
                        except StopIteration:
                            iterators[batch_task] = iter(loaders[batch_task])
                            raw = next(iterators[batch_task])
                        for wrapper in self.wrappers.values():
                            wrapper.active_task = batch_task
                        # Token-weighted task loss, sample-weighted gate losses (their means run over a padded
                        # length that is identical on every rank).
                        local, loss_weight = distributed.shard_batch(raw, model)
                        local_count = next(iter(local.values())).shape[0]
                        sample_weight = (
                            local_count / next(iter(raw.values())).shape[0]
                            if distributed.get_context().enabled
                            else 1.0
                        )
                        is_replay = batch_task != current
                        task_value = gate_value = 0.0
                        if local_count > 0:
                            batch = {k: v.to(device) for k, v in local.items()}
                            captured.clear()
                            model.train()
                            task_loss = model(**batch).loss
                            gate = self._gate_loss(captured, batch_task, task_loss)
                            loss = (
                                gate * sample_weight
                                if is_replay and m.replay_objective == "kd_only"
                                else task_loss * loss_weight + gate * sample_weight
                            )
                            if not torch.isfinite(loss):
                                raise RuntimeError("MoLE-CIE produced a non-finite objective")
                            (loss / cfg.grad_accum).backward()
                            task_value = float(task_loss.detach().float().cpu()) * loss_weight
                            gate_value = float(gate.detach().float().cpu()) * sample_weight
                        task_value, gate_value = distributed.all_reduce_sum(task_value, gate_value, device=device)
                        losses.append(task_value)
                        if is_replay:
                            continue
                        micro += 1
                        if micro % cfg.grad_accum:
                            continue
                        distributed.all_reduce_gradients(parameters)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            parameters, cfg.max_grad_norm, error_if_nonfinite=True
                        )
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        steps += 1
                        if distributed.is_main():
                            record = {
                                "adapter": current,
                                "loss": task_value,
                                "penalty": gate_value,
                                "grad_norm": float(grad_norm),
                            }
                            tracking.recorder().train_step(current, steps, [record], optimizer.param_groups[0]["lr"])
                        if cfg.max_steps and steps >= cfg.max_steps:
                            return losses
        finally:
            for wrapper in self.wrappers.values():
                wrapper.active_task = current
            for handle in handles:
                handle.remove()
        return losses

    def _gate_loss(
        self, captured: Mapping[str, torch.Tensor], batch_task: str, task_loss: torch.Tensor
    ) -> torch.Tensor:
        """Gate reflection (router + old-key KL) plus task-key supervision, averaged over wrapped layers."""

        m = self.cfg.mole
        gate = task_loss * 0.0
        regularized = 0
        for name, wrapper in self.wrappers.items():
            hidden = captured.get(name)
            if hidden is None:
                continue
            regularized += 1
            gate = gate + wrapper.gate_reflection_loss(
                hidden, temperature=m.kd_temperature, router_weight=m.router_kd_weight, key_weight=m.key_kd_weight
            )
            probabilities = wrapper.task_probabilities(hidden, trainable_task=batch_task)["probabilities"]
            key_loss = (
                -probabilities[..., wrapper.task_ids.index(batch_task)]
                .clamp_min(torch.finfo(probabilities.dtype).tiny)
                .log()
                .mean()
            )
            gate = gate + m.task_key_supervision_weight * key_loss
        if regularized:
            gate = gate / regularized
        return gate * m.gate_reflection_weight

    def learn_task(self, index: int, task: TaskCfg, examples: Sequence[Example]) -> dict[str, Any]:
        cfg, m = self.cfg, self.cfg.mole
        pool = min(len(examples), max(m.prototype_num_samples, self.memory.budget * 4))
        candidates = select_support(examples, pool, cfg.seed + index)
        texts = [e.input_text for e in candidates]
        embeddings = input_embeddings(
            self.model,
            self.ctx.tokenizer,
            texts,
            batch_size=cfg.eval.embedding_batch_size,
            max_length=cfg.data.max_source_length,
        )
        keys = self._task_key_prototypes(candidates)
        for wrapper in self.wrappers.values():
            wrapper.add_task(task.name)
        for name, wrapper in self.wrappers.items():
            wrapper.set_task_key(task.name, keys[name])
        distributed.sync_module_parameters(list(self.wrappers.values()))
        replay = self.memory.examples()
        losses = self._train({task.name: examples, **replay}, task.name, cfg.seed + index)
        self.memory.update(task.name, candidates, embeddings)
        for wrapper in self.wrappers.values():
            wrapper.snapshot_gates()
        distributed.sync_module_parameters(list(self.wrappers.values()))
        return {
            "replay_examples": sum(map(len, replay.values())),
            "loss": losses,
            "memory": {t: len(e) for t, e in self.memory.entries.items()},
        }

    # ------------------------------------------------------------ inference

    def predict(self, examples: Sequence[Example]) -> list[Prediction]:
        ctx, model = self.ctx, self.model
        settings = ctx.generation
        wrappers = self.wrappers
        task_ids = next(iter(wrappers.values())).task_ids
        encoder_wrappers = [w for name, w in wrappers.items() if name.startswith("encoder.")]

        def set_encoder_mask(encoded: Mapping[str, torch.Tensor]) -> None:
            # Sentence-level task routing must ignore batch padding, so a padded batch routes like batch size 1.
            for wrapper in wrappers.values():
                wrapper.encoder_mask = encoded["attention_mask"]
            if not ctx.backbone.seq2seq or not encoder_wrappers:
                return
            # Teacher-forced training applies one task expert to every layer of an example.  Choose that task once
            # from the encoder's key similarities (majority over encoder layers) and hold it fixed while decoding.
            for wrapper in encoder_wrappers:
                wrapper.start_route_trace()
            with torch.inference_mode():
                model.get_encoder()(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
            votes = torch.zeros(encoded["input_ids"].shape[0], len(task_ids), dtype=torch.long)
            for wrapper in encoder_wrappers:
                for entry in wrapper.finish_route_trace():
                    selected = torch.tensor(entry["selected"], dtype=torch.long)
                    votes[torch.arange(len(selected)), selected] += 1
            choice = votes.argmax(dim=-1).to(encoded["input_ids"].device)
            for wrapper in wrappers.values():
                wrapper.forced_task_indices = choice
            for wrapper in encoder_wrappers:
                wrapper.start_route_trace()

        def generate_shard(shard: list[Example]) -> list[tuple[str, dict[str, int], float, int]]:
            generated: list[Any] = [None] * len(shard)
            order = (
                sorted(range(len(shard)), key=lambda i: len(shard[i].input_text))
                if settings.sort_by_length
                else list(range(len(shard)))
            )
            for start in range(0, len(order), settings.batch_size):
                chunk_indices = order[start : start + settings.batch_size]
                chunk = [shard[i] for i in chunk_indices]
                for wrapper in wrappers.values():
                    wrapper.start_route_trace()
                try:
                    outputs = generate_batch(
                        model,
                        ctx.tokenizer,
                        ctx.backbone,
                        [e.input_text for e in chunk],
                        settings,
                        on_encoded=set_encoder_mask,
                    )
                finally:
                    for wrapper in wrappers.values():
                        wrapper.encoder_mask = None
                        wrapper.forced_task_indices = None
                traces = [entry for wrapper in wrappers.values() for entry in wrapper.finish_route_trace()]
                for row, output in enumerate(outputs):
                    route_counts: dict[str, int] = {}
                    confidence_sum = 0.0
                    for entry in traces:
                        task = task_ids[entry["selected"][row]]
                        route_counts[task] = route_counts.get(task, 0) + 1
                        confidence_sum += float(entry["confidence"][row])
                    generated[chunk_indices[row]] = (output, route_counts, confidence_sum, len(traces))
            return generated

        predictions = []
        for text, counts, confidence_sum, decisions in distributed.sharded_map(generate_shard, list(examples)):
            route = max(counts.items(), key=lambda item: item[1])[0]
            predictions.append(Prediction(text, route, confidence_sum / decisions))
        return predictions

    def state_dict(self) -> dict[str, Any]:
        return {
            "wrappers": {name: w.export_state() for name, w in self.wrappers.items()},
            "memory": self.memory.examples(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name, wrapper in self.wrappers.items():
            wrapper.load_state(state["wrappers"][name])
        for task, examples in state["memory"].items():
            self.memory.entries[task] = examples
