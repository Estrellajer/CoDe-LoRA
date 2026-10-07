import os
import socket

import pytest
import torch
import torch.multiprocessing as mp

from codelora import distributed


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _worker(rank: int, world: int, port: int, job: str) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world), RANK=str(rank), LOCAL_RANK=str(rank)
    )
    distributed.init_from_env()
    try:
        globals()[f"_job_{job}"](rank, world)
    finally:
        distributed.shutdown()


def run_distributed(job: str, world: int = 2) -> None:
    mp.spawn(_worker, args=(world, _port(), job), nprocs=world, join=True)


def _job_sharded_map(rank, world):
    def square(items):
        return [i * i for i in items]

    assert distributed.sharded_map(square, list(range(11))) == [i * i for i in range(11)]
    assert distributed.sharded_map(square, []) == []
    assert distributed.sharded_map(square, [3]) == [9]  # fewer items than ranks


def _job_gradients_and_scalars(rank, world):
    p = torch.nn.Parameter(torch.zeros(3))
    q = torch.nn.Parameter(torch.zeros(2, 2))
    p.grad = torch.full((3,), float(rank + 1))
    q.grad = None if rank == 1 else torch.ones(2, 2)  # a rank without gradient counts as zero
    distributed.all_reduce_gradients([p, q])
    assert torch.equal(p.grad, torch.full((3,), 3.0)) and torch.equal(q.grad, torch.ones(2, 2))
    assert distributed.all_reduce_sum(1.0 + rank, 10.0, device=torch.device("cpu")) == (3.0, 20.0)
    value = torch.full((2,), float(rank * 5 + 1))
    distributed.broadcast_parameters([value])
    assert torch.equal(value, torch.ones(2))
    assert distributed.main_computes(lambda: 7 + rank) == 7


class _Seq2Seq:
    class config:
        is_encoder_decoder = True


class _Causal:
    class config:
        is_encoder_decoder = False


def _job_shard_batch(rank, world):
    labels = torch.tensor([[1, 2, -100], [3, -100, -100], [4, 5, 6], [7, 8, -100], [9, -100, -100]])
    batch = {"labels": labels, "input_ids": torch.arange(5)}
    local, weight = distributed.shard_batch(batch, _Seq2Seq())
    assert abs(sum(distributed.all_gather_objects(weight)) - 1.0) < 1e-12
    lengths = distributed.all_gather_objects(len(local["input_ids"]))
    assert sum(lengths) == 5
    # decoder-only: the first label position is not a loss token
    local, weight = distributed.shard_batch(batch, _Causal())
    assert abs(sum(distributed.all_gather_objects(weight)) - 1.0) < 1e-12


def _job_empty_shard(rank, world):
    batch = {"labels": torch.tensor([[1, 2]]), "input_ids": torch.tensor([0])}  # one sample, two ranks
    local, weight = distributed.shard_batch(batch, _Seq2Seq())
    assert (len(local["input_ids"]), weight) == ((0, 0.0) if rank == 0 else (1, 1.0))


def _job_sharded_generation(rank, world):
    """Sharded generation returns the same texts, in the same order, as one process (fp32: no batch-shape noise)."""

    import dataclasses

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from codelora.config import LoraCfg
    from codelora.evaluation.generate import GenerationSettings, generate_routed
    from codelora.models import adapters
    from codelora.models.backbone import BACKBONES

    path = os.environ["CODELORA_TEST_QWEN"]
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    backbone = BACKBONES["qwen3-0.6b"]
    config = adapters.lora_config(base, backbone, LoraCfg(target_modules=["q_proj", "v_proj"]))
    model = adapters.build_peft_model(base, config, seed=0)
    adapters.add_adapter(model, "other", config, seed=1, index=1, role="expert")
    torch.manual_seed(0)  # every rank must hold identical weights
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.3)
    texts = [" ".join(f"w{j}" for j in range(3 + (i * 7) % 11)) for i in range(13)]
    adapter_names = ["shared" if i % 3 else "other" for i in range(13)]
    settings = GenerationSettings(64, 6, batch_size=4, official=False)
    sharded = generate_routed(model, tokenizer, backbone, texts, adapter_names, settings)
    context = distributed._CONTEXT
    distributed._CONTEXT = dataclasses.replace(context, rank=0, world_size=1)
    try:
        single = generate_routed(model, tokenizer, backbone, texts, adapter_names, settings)
    finally:
        distributed._CONTEXT = context
    assert sharded == single, (sharded, single)
    assert len(set(single)) > 1, single


@pytest.mark.parametrize("job", ["sharded_map", "gradients_and_scalars", "shard_batch", "empty_shard"])
def test_gloo_two_ranks(job):
    run_distributed(job)


def test_sharded_generation_matches_single_process(lab, monkeypatch):
    monkeypatch.setenv("CODELORA_TEST_QWEN", str(lab["qwen"]))
    run_distributed("sharded_generation")


def test_single_process_helpers_are_identities():
    assert distributed.get_context().world_size == 1 and distributed.is_main()
    assert distributed.sharded_map(lambda xs: [x + 1 for x in xs], [1, 2]) == [2, 3]
    assert distributed.all_reduce_sum(2.0, device=torch.device("cpu")) == (2.0,)
    batch = {"labels": torch.tensor([[1]])}
    assert distributed.shard_batch(batch, _Seq2Seq()) == (batch, 1.0)
    assert distributed.rng_role_suffix() == ""
