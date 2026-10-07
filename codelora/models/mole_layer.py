"""MoLE-CIE layer (EMNLP 2025): token-level top-k routing over generic LoRA experts,
sentence-level routing to task experts through trainable task keys, a shared expert,
and gate reflection (router / key knowledge distillation).

``MoLELinear`` wraps a frozen ``nn.Linear`` and computes

    base(x) + alpha * sum_i(router_i * generic_i(x))
            + beta * (theta_t * task_t(x) + (1 - theta_t) * shared(x)).
"""

from __future__ import annotations

import base64
import copy
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Decoder-only backbones: the sentence-level task key must ignore batch padding, so a padded batch
# routes exactly like batch size 1.  A forward pre-hook on the causal LM publishes the current
# attention mask here.
_CURRENT_ATTENTION_MASK: list[Tensor | None] = [None]
_CAUSAL_BACKBONE = [False]


def install_attention_mask_hook(model: nn.Module) -> None:
    """Publish ``attention_mask`` of every top-level forward of ``model`` to MoLE layers."""

    _CAUSAL_BACKBONE[0] = True

    def pre(_module, _args, kwargs):
        _CURRENT_ATTENTION_MASK[0] = kwargs.get("attention_mask")

    def post(_module, _args, _kwargs, _output):
        _CURRENT_ATTENTION_MASK[0] = None

    model.register_forward_pre_hook(pre, with_kwargs=True)
    model.register_forward_hook(post, with_kwargs=True)


def _module_key(task: str) -> str:
    """Encode an arbitrary task name as a valid ``ModuleDict`` key."""

    return "task_" + base64.urlsafe_b64encode(task.encode()).decode("ascii").rstrip("=")


class LoRAExpert(nn.Module):
    """``scale * B(A(x))`` with the standard LoRA initialisation."""

    def __init__(self, in_features: int, out_features: int, rank: int, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.B(self.A(hidden_states)) * self.scale


class MoLELinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        num_experts: int,
        rank: int,
        top_k: int,
        alpha: float,
        beta: float,
        lora_scale: float,
        router_temperature: float,
        task_key_temperature: float,
        theta_floor: float,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        self.rank, self.top_k = rank, top_k
        self.alpha, self.beta, self.lora_scale = alpha, beta, lora_scale
        self.router_temperature, self.task_key_temperature = router_temperature, task_key_temperature
        self.theta_floor = theta_floor  # official MoLE-CIE clamps the selected task-key weight from below

        self.generic_experts = nn.ModuleList(
            LoRAExpert(self.in_features, self.out_features, rank, lora_scale).to(base_layer.weight)
            for _ in range(num_experts)
        )
        self.router_w1 = nn.Linear(self.in_features, num_experts, bias=False).to(base_layer.weight)
        self.router_w2 = nn.Linear(num_experts, num_experts, bias=False).to(base_layer.weight)
        self.shared_expert = LoRAExpert(self.in_features, self.out_features, rank, lora_scale).to(base_layer.weight)
        self.task_experts = nn.ModuleDict()
        self.task_keys = nn.ParameterDict()
        self.task_ids: list[str] = []
        self.active_task: str | None = None
        self._gate_snapshot: dict[str, Any] | None = None
        self._route_trace: list[dict[str, Any]] | None = None
        self._prefill_task_route: dict[str, Tensor] | None = None
        # Evaluation hooks: the encoder padding mask (sentence means ignore padding) and, for
        # encoder-decoder models, the per-example task chosen from the encoder input so every
        # decoder step uses the same task expert, as in teacher-forced training.
        self.encoder_mask: Tensor | None = None
        self.forced_task_indices: Tensor | None = None

    # ---------------------------------------------------------------- tasks

    def add_task(self, task: str) -> None:
        """Add the only trainable task expert/key and freeze earlier experts."""

        key = _module_key(task)
        if task not in self.task_ids:
            self.task_experts[key] = LoRAExpert(self.in_features, self.out_features, self.rank, self.lora_scale).to(
                self.base_layer.weight
            )
            task_key = torch.empty(
                self.in_features, device=self.base_layer.weight.device, dtype=self.base_layer.weight.dtype
            )
            nn.init.normal_(task_key, std=self.in_features**-0.5)
            self.task_keys[key] = nn.Parameter(task_key)
            self.task_ids.append(task)
        for known in self.task_ids:
            self.task_experts[_module_key(known)].requires_grad_(known == task)
            # Keys stay plastic for replay/KD; forward() detaches non-active keys so a current-task
            # batch cannot drag historical prototypes.
            self.task_keys[_module_key(known)].requires_grad_(True)
        self.active_task = task

    def set_task_key(self, task: str, value: Tensor) -> None:
        normalized = F.normalize(value.detach().to(self.base_layer.weight), dim=0)
        with torch.no_grad():
            self.task_keys[_module_key(task)].copy_(normalized)

    def start_route_trace(self) -> None:
        self._route_trace = []

    def finish_route_trace(self) -> list[dict[str, Any]]:
        trace, self._route_trace = list(self._route_trace or []), None
        return trace

    # -------------------------------------------------------------- routing

    def router_logits(self, hidden_states: Tensor, temperature: float | None = None) -> Tensor:
        tau = self.router_temperature if temperature is None else float(temperature)
        return self.router_w2(torch.tanh(self.router_w1(hidden_states))) / tau

    def route_tokens(self, hidden_states: Tensor) -> Tensor:
        probabilities = F.softmax(self.router_logits(hidden_states), dim=-1)
        top_values, top_indices = probabilities.topk(self.top_k, dim=-1)
        top_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(top_values.dtype).tiny)
        return torch.zeros_like(probabilities).scatter(-1, top_indices, top_weights)

    @staticmethod
    def _sentence_mean(hidden_states: Tensor, attention_mask: Tensor | None) -> Tensor:
        if hidden_states.ndim == 1:
            return hidden_states
        if attention_mask is None:
            return hidden_states.mean(dim=-2) if hidden_states.ndim > 2 else hidden_states
        mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if hidden_states.ndim == 2:
            return (hidden_states * mask.unsqueeze(-1)).sum(dim=0) / mask.sum().clamp_min(1.0)
        return (hidden_states * mask.unsqueeze(-1)).sum(dim=-2) / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)

    def task_probabilities(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        *,
        trainable_task: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Tensor]:
        sentence = self._sentence_mean(hidden_states, attention_mask)
        if not self.task_ids:
            empty = sentence.new_empty((*sentence.shape[:-1], 0))
            return {"scores": empty, "probabilities": empty}
        keys = torch.stack(
            [
                self.task_keys[_module_key(task)]
                if trainable_task is None or task == trainable_task
                else self.task_keys[_module_key(task)].detach()
                for task in self.task_ids
            ]
        )
        scores = F.cosine_similarity(sentence.unsqueeze(-2), keys, dim=-1)
        tau = self.task_key_temperature if temperature is None else float(temperature)
        return {"scores": scores, "probabilities": F.softmax(scores / tau, dim=-1)}

    def _task_outputs(self, hidden_states: Tensor, index: Tensor) -> Tensor:
        """Output of the per-example selected task expert (``index`` broadcasts over tokens)."""

        outputs = torch.stack([self.task_experts[_module_key(t)](hidden_states) for t in self.task_ids], dim=-2)
        if hidden_states.ndim >= 3:
            index = index.unsqueeze(-1).expand(*hidden_states.shape[:-1])
        gather = index.unsqueeze(-1).unsqueeze(-1).expand(*outputs.shape[:-2], 1, self.out_features)
        return outputs.gather(-2, gather).squeeze(-2)

    def forward(self, hidden_states: Tensor) -> Tensor:
        # Training has an observed task identity: force the current task expert so it receives gradients
        # even when a fresh key loses the argmax to an older task.  Evaluation routes by key similarity.
        task_id = self.active_task if self.training else None
        attention_mask = None
        if (
            self.encoder_mask is not None
            and hidden_states.ndim == 3
            and hidden_states.shape[:2] == self.encoder_mask.shape
        ):
            attention_mask = self.encoder_mask
        current = _CURRENT_ATTENTION_MASK[0]
        if (
            attention_mask is None
            and current is not None
            and hidden_states.ndim == 3
            and hidden_states.shape[:2] == current.shape
        ):
            attention_mask = current
        base = self.base_layer(hidden_states)
        weights = self.route_tokens(hidden_states)
        generic_outputs = torch.stack([expert(hidden_states) for expert in self.generic_experts], dim=-2)
        generic = (generic_outputs * weights.unsqueeze(-1)).sum(dim=-2)

        task_route = self.task_probabilities(
            hidden_states, attention_mask, trainable_task=task_id if self.training else None
        )
        if not self.training and hidden_states.ndim == 3 and _CAUSAL_BACKBONE[0]:
            # Decoder generation: decide the task once from the whole prompt (prefill) and keep it for every
            # decode step; a one-token "sentence" is not a task signal.
            if hidden_states.shape[1] > 1:
                self._prefill_task_route = task_route
            elif (
                self._prefill_task_route is not None
                and self._prefill_task_route["probabilities"].shape[0] == hidden_states.shape[0]
            ):
                task_route = self._prefill_task_route
        probabilities = task_route["probabilities"]
        selected: Any = None
        theta = probabilities.new_zeros(probabilities.shape[:-1])
        task_output = torch.zeros_like(base)
        if self.task_ids:
            if task_id is not None:
                theta = probabilities[..., self.task_ids.index(task_id)]
                task_output = self.task_experts[_module_key(task_id)](hidden_states)
            else:
                if self.forced_task_indices is not None:
                    selected = self.forced_task_indices.to(probabilities.device)
                    while selected.ndim < probabilities.ndim - 1:
                        selected = selected.unsqueeze(-1)
                    selected = selected.expand(probabilities.shape[:-1])
                    theta = probabilities.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
                else:
                    theta, selected = probabilities.max(dim=-1)
                task_output = self._task_outputs(hidden_states, selected)

        if self._route_trace is not None and self.task_ids and selected is not None:
            chosen = selected.detach().long().cpu().reshape(-1)
            top = (
                probabilities.detach()
                .float()
                .cpu()
                .reshape(-1, len(self.task_ids))
                .topk(min(2, len(self.task_ids)), dim=-1)
                .values
            )
            margin = top[:, 0] - top[:, 1] if top.shape[1] > 1 else top[:, 0]
            self._route_trace.append(
                {"selected": chosen.tolist(), "confidence": top[:, 0].tolist(), "margin": margin.tolist()}
            )

        if self.task_ids and self.theta_floor > 0.0:
            theta = theta.clamp_min(self.theta_floor)
        while theta.ndim < hidden_states.ndim - 1:
            theta = theta.unsqueeze(-1)
        theta = theta.unsqueeze(-1)
        mixture = theta * task_output + (1.0 - theta) * self.shared_expert(hidden_states)
        return base + self.alpha * generic + self.beta * mixture

    # ------------------------------------------------------- gate reflection

    def snapshot_gates(self) -> None:
        """Freeze a detached teacher copy of the router weights and existing task keys."""

        self._gate_snapshot = {
            "router_w1": self.router_w1.weight.detach().cpu().clone(),
            "router_w2": self.router_w2.weight.detach().cpu().clone(),
            "task_keys": {t: self.task_keys[_module_key(t)].detach().cpu().clone() for t in self.task_ids},
        }

    @staticmethod
    def _kl(teacher: Tensor, student: Tensor) -> Tensor:
        tiny = torch.finfo(student.dtype).tiny
        return (teacher * (teacher.clamp_min(tiny).log() - student.clamp_min(tiny).log())).sum(dim=-1)

    def gate_reflection_loss(
        self, hidden_states: Tensor, *, temperature: float, router_weight: float, key_weight: float
    ) -> Tensor:
        """``KL(previous || current)`` over token routing and (for >= 2 old tasks) task-key scores."""

        zero = hidden_states.sum() * 0.0
        if self._gate_snapshot is None:
            return zero
        snapshot = self._gate_snapshot
        w1, w2 = snapshot["router_w1"].to(hidden_states), snapshot["router_w2"].to(hidden_states)
        teacher_logits = F.linear(torch.tanh(F.linear(hidden_states, w1)), w2) / temperature
        student_logits = self.router_logits(hidden_states, temperature)
        router_loss = self._kl(F.softmax(teacher_logits, dim=-1), F.softmax(student_logits, dim=-1)).mean()

        old = [t for t in self.task_ids if t in snapshot["task_keys"]]
        if len(old) < 2:
            key_loss = zero
        else:
            sentence = self._sentence_mean(hidden_states, None)
            teacher_keys = torch.stack([snapshot["task_keys"][t].to(hidden_states) for t in old])
            student_keys = torch.stack([self.task_keys[_module_key(t)] for t in old])
            teacher_scores = F.cosine_similarity(sentence.unsqueeze(-2), teacher_keys, dim=-1) / temperature
            student_scores = F.cosine_similarity(sentence.unsqueeze(-2), student_keys, dim=-1) / temperature
            key_loss = self._kl(F.softmax(teacher_scores, -1), F.softmax(student_scores, -1)).mean()
        return float(router_weight) * router_loss + float(key_weight) * key_loss

    # ----------------------------------------------------------- checkpoint

    def export_state(self) -> dict[str, Any]:
        tensors = {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if not k.startswith("base_layer.")}
        return {"task_ids": list(self.task_ids), "tensors": tensors, "snapshot": copy.deepcopy(self._gate_snapshot)}

    def load_state(self, state: dict[str, Any]) -> None:
        for task in state["task_ids"]:
            self.add_task(task)
        self.load_state_dict(state["tensors"], strict=False)
        self._gate_snapshot = copy.deepcopy(state["snapshot"])


class ExemplarMemory:
    """Per-task replay memory: the ``budget`` examples nearest to the task centroid."""

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.entries: dict[str, list[Any]] = {}

    def update(self, task: str, examples: Sequence[Any], embeddings: Tensor) -> None:
        normalized = F.normalize(embeddings.detach().float(), dim=-1)
        centroid = F.normalize(normalized.mean(dim=0), dim=0)
        distances = 1.0 - normalized @ centroid
        # Python's stable sort makes the original example order the tie breaker.
        order = sorted(range(len(distances)), key=lambda i: (float(distances[i]), i))[: min(self.budget, len(examples))]
        self.entries[task] = [examples[i] for i in order]

    def examples(self) -> dict[str, list[Any]]:
        return {task: list(items) for task, items in self.entries.items()}
