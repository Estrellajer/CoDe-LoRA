"""Model-level tests of the growing-rank adapter, SVD retraction and the MoLE layer on tiny CPU models."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from codelora.config import LoraCfg
from codelora.models import adapters, growing
from codelora.models.backbone import BACKBONES
from codelora.models.mole_layer import ExemplarMemory, MoLELinear


def tiny_peft(lab, rank=4):
    base = AutoModelForCausalLM.from_pretrained(lab["qwen"], dtype=torch.float32)
    config = adapters.lora_config(
        base, BACKBONES["qwen3-0.6b"], LoraCfg(r=rank, alpha=16, target_modules=["q_proj", "v_proj"])
    )
    return adapters.build_peft_model(base, config, seed=0), config


def test_growing_rank_forward_is_history_plus_current(lab):
    model, _ = tiny_peft(lab)
    growing.install_growing_rank(model, "shared")
    x = torch.randn(3, 5, 32)
    _, layer = next(adapters.lora_layers(model, "shared"))
    a, b = layer.lora_A["shared"], layer.lora_B["shared"]
    with torch.no_grad():
        b.current.weight.normal_()
    before = b(a(x))
    assert growing.commit_task(model, "shared") == 4
    assert a.current is None and torch.allclose(b(a(x)), before)  # committing does not change the function
    growing.begin_task(model, "shared", 4)
    assert a.history_rank == 4 and a.current.weight.shape == (4, 32) and not b.current.weight.any()  # B starts at zero
    assert torch.allclose(b(a(x)), before)  # the new block contributes nothing yet
    with torch.no_grad():
        b.current.weight.normal_()
    expected = x @ a.history_weight.T @ b.history_weight.T + x @ a.current.weight.T @ b.current.weight.T
    assert torch.allclose(b(a(x)), expected, atol=1e-5)
    assert growing.commit_task(model, "shared") == 8


def test_growing_history_never_receives_gradients(lab):
    model, _ = tiny_peft(lab)
    growing.install_growing_rank(model, "shared")
    growing.commit_task(model, "shared")
    growing.begin_task(model, "shared", 4)
    params = adapters.activate(model, "shared", "shared")
    assert len(params) == 2 * sum(1 for _ in adapters.lora_layers(model, "shared"))  # current A and B only
    ids = torch.randint(0, 100, (2, 6))
    model(input_ids=ids, labels=ids).loss.backward()
    _, layer = next(adapters.lora_layers(model, "shared"))
    assert not layer.lora_A["shared"].history_weight.requires_grad
    assert layer.lora_A["shared"].current.weight.grad is not None


def test_growing_state_round_trip(lab):
    model, _ = tiny_peft(lab)
    growing.install_growing_rank(model, "shared")
    growing.commit_task(model, "shared")
    state = growing.export_state(model, "shared")
    other, _ = tiny_peft(lab)
    growing.install_growing_rank(other, "shared")
    growing.replace_history(other, "shared", {n: (s["B"], s["A"]) for n, s in state.items()})
    for (_, a), (_, b) in zip(
        adapters.lora_layers(model, "shared"), adapters.lora_layers(other, "shared"), strict=True
    ):
        assert torch.equal(a.lora_A["shared"].history_weight, b.lora_A["shared"].history_weight)
        assert b.lora_A["shared"].current is None


def test_effective_factor_round_trip_folds_the_peft_scaling(lab):
    model, _ = tiny_peft(lab)  # scaling = alpha / r = 4
    with torch.no_grad():
        for _, layer in adapters.lora_layers(model, "shared"):
            layer.lora_B["shared"].weight.normal_()
    factors = adapters.effective_factors(model, "shared")
    name, layer = next(adapters.lora_layers(model, "shared"))
    b_raw = layer.lora_B["shared"].weight.detach().clone()
    assert torch.allclose(factors[name][0], 4.0 * b_raw)
    adapters.write_factors(model, "shared", {k: (b * 0.5, a) for k, (b, a) in factors.items()})
    assert torch.allclose(layer.lora_B["shared"].weight, 0.5 * b_raw)


def test_write_factors_zero_pads_lower_rank(lab):
    model, _ = tiny_peft(lab)
    current = adapters.effective_factors(model, "shared")
    factors = {k: (torch.randn(b.shape[0], 2), torch.randn(2, a.shape[1])) for k, (b, a) in current.items()}
    adapters.write_factors(model, "shared", factors)
    for _, layer in adapters.lora_layers(model, "shared"):
        assert not layer.lora_B["shared"].weight[:, 2:].any() and not layer.lora_A["shared"].weight[2:].any()
        assert layer.lora_B["shared"].weight[:, :2].any()


def test_activate_trains_only_the_chosen_adapter(lab):
    model, config = tiny_peft(lab)
    adapters.add_adapter(model, "other", config, seed=0, index=1, role="expert")
    params = adapters.activate(model, ["shared", "other"], "other")
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(".other." in n for n in trainable) and len(params) == len(trainable)
    adapters.freeze(model, "other")
    assert not any(p.requires_grad for p in model.parameters())


def test_branch_switch_matches_a_full_activation(lab):
    model, config = tiny_peft(lab)
    for index, name in enumerate(("shared_update", "expert"), start=1):
        adapters.add_adapter(model, name, config, seed=0, index=index, role=name)
    branches = [
        (["shared", "shared_update"], "shared_update"),
        (["expert"], "expert"),
        (["shared"], "shared"),
        (["expert"], "expert"),
    ]
    switch = adapters.BranchSwitch(model, branches)
    for position in (0, 1, 2, 3, 0, 0, 2):
        switch.select(position)
        switched = {n for n, p in model.named_parameters() if p.requires_grad}
        active = model.base_model.active_adapters
        expected = adapters.activate(model, *branches[position])
        assert switched == {n for n, p in model.named_parameters() if p.requires_grad}
        assert [id(p) for p in expected] == [id(p) for p in switch.parameters[position]]
        assert model.base_model.active_adapters == active
        switch._selected = None  # activate() changed the state behind the switch's back: force the next select


def test_adapter_initialisation_is_keyed_by_role_not_by_global_rng(lab):
    model, config = tiny_peft(lab)
    torch.manual_seed(1)
    adapters.add_adapter(model, "e1", config, seed=7, index=2, role="expert")
    torch.manual_seed(999)
    adapters.add_adapter(model, "e2", config, seed=7, index=2, role="expert")
    (_, l1), (_, l2) = next(adapters.lora_layers(model, "e1")), next(adapters.lora_layers(model, "e2"))
    assert torch.equal(l1.lora_A["e1"].weight, l2.lora_A["e2"].weight)


# ---------------------------------------------------------------- MoLE


def make_layer(**kw) -> MoLELinear:
    torch.manual_seed(0)
    args = {
        "num_experts": 4, "rank": 2, "top_k": 2, "alpha": 1.0, "beta": 1.0, "lora_scale": 1.0,
        "router_temperature": 1.0, "task_key_temperature": 1.0, "theta_floor": 0.0,
    }  # fmt: skip
    return MoLELinear(nn.Linear(8, 6), **{**args, **kw})


def test_mole_routes_to_the_nearest_task_key_at_inference():
    layer = make_layer()
    for task in ("a", "b"):
        layer.add_task(task)
    layer.set_task_key("a", torch.tensor([1.0] + [0.0] * 7))
    layer.set_task_key("b", torch.tensor([0.0, 1.0] + [0.0] * 6))
    layer.eval()
    layer.start_route_trace()
    x = torch.zeros(2, 3, 8)
    x[0, :, 0], x[1, :, 1] = 1.0, 1.0
    layer(x)
    (entry,) = layer.finish_route_trace()
    assert entry["selected"] == [0, 1]  # example 0 -> task a, example 1 -> task b


def test_mole_training_uses_the_active_task_expert_only():
    layer = make_layer()
    layer.add_task("a")
    layer.add_task("b")
    old, new = (layer.task_experts[k] for k in layer.task_experts)
    assert not any(p.requires_grad for p in old.parameters()) and all(p.requires_grad for p in new.parameters())
    layer.train()
    layer(torch.randn(2, 3, 8)).sum().backward()
    assert new.B.weight.grad.abs().sum() > 0  # B starts at zero, so the first gradient lands on B
    assert old.B.weight.grad is None and layer.active_task == "b"


def test_gate_reflection_is_zero_until_a_snapshot_exists_and_positive_after_drift():
    layer = make_layer()
    hidden = torch.randn(2, 5, 8)
    layer.add_task("a")
    kwargs = {"temperature": 1.0, "router_weight": 1.0, "key_weight": 1.0}
    assert layer.gate_reflection_loss(hidden, **kwargs).item() == 0
    layer.snapshot_gates()
    assert abs(layer.gate_reflection_loss(hidden, **kwargs).item()) < 1e-7
    with torch.no_grad():
        layer.router_w2.weight.add_(torch.randn_like(layer.router_w2.weight))
    assert layer.gate_reflection_loss(hidden, **kwargs).item() > 0


def test_mole_state_round_trip_reproduces_outputs():
    layer = make_layer(theta_floor=0.75)
    for task in ("a", "b"):
        layer.add_task(task)
    layer.snapshot_gates()
    state = layer.export_state()
    restored = make_layer(theta_floor=0.75)
    restored.base_layer.load_state_dict(layer.base_layer.state_dict())
    restored.load_state(state)
    layer.eval()
    restored.eval()
    x = torch.randn(3, 4, 8)
    assert torch.equal(layer(x), restored(x))
    assert restored.task_ids == ["a", "b"]


def test_exemplar_memory_keeps_the_examples_nearest_the_centroid():
    memory = ExemplarMemory(budget=2)
    embeddings = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [1.0, 0.1]])
    memory.update("t", ["e0", "e1", "e2", "e3"], embeddings)
    assert sorted(memory.examples()["t"]) == ["e1", "e3"]  # the outlier e2 is dropped, nearest first
    memory.update("t", ["x"], torch.tensor([[1.0, 1.0]]))
    assert memory.examples() == {"t": ["x"]}  # replaced, not merged


def test_qwen35_auto_targets_cover_text_attention_only():
    from codelora.models.backbone import lora_targets

    names = [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.k_proj",
        "model.language_model.layers.0.self_attn.v_proj",
        "model.language_model.layers.1.linear_attn.in_proj_qkv",
        "model.language_model.layers.1.linear_attn.in_proj_z",
        "model.language_model.layers.1.linear_attn.out_proj",
        "model.language_model.layers.1.mlp.gate_proj",
        "model.visual.blocks.0.attn.q_proj",
        "model.visual.blocks.0.attn.out_proj",
    ]
    model = nn.Module()
    for name in names:
        parent = model
        *path, leaf = name.split(".")
        for part in path:
            if not hasattr(parent, part):
                parent.add_module(part, nn.Module())
            parent = getattr(parent, part)
        parent.add_module(leaf, nn.Linear(4, 4))
    targets = lora_targets(model, BACKBONES["qwen3.5-4b"], None)
    assert targets == [
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.v_proj",
        "model.language_model.layers.1.linear_attn.in_proj_qkv",
        "model.language_model.layers.1.linear_attn.in_proj_z",
        "model.language_model.layers.1.linear_attn.out_proj",
    ]
    assert lora_targets(model, BACKBONES["t5-large"], None) == ["q", "v"]
    assert lora_targets(model, BACKBONES["t5-large"], ["o"]) == ["o"]
