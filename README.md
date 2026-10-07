# CoDe-LoRA

[![arXiv](https://img.shields.io/badge/arXiv-2610.08312-b31b1b.svg)](https://arxiv.org/abs/2610.08312)
[![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026-blue.svg)](https://arxiv.org/abs/2610.08312)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

**CoDe-LoRA: Mitigating the Orthogonality Dilemma in Continual Learning of LLMs via Knowledge Consolidation and Decoupling**
Maoqi Liu, Quan Fang, Yufei He (BUPT, NUS)

Official code of the paper: continual learning of language models with LoRA, where shared knowledge is **Co**nsolidated into
one low-rank branch and task-specific knowledge is **De**coupled into per-task experts behind a router. The repository also
contains the baselines, benchmarks and backbones of the paper behind one interface.

**News:** CoDe-LoRA is accepted to EMNLP 2026 (Main Conference).

## Method in brief

- One plain LoRA adapter per task, trained on top of the frozen consolidated shared branch (Co) of the previous tasks.
- After the task, the adapter is frozen as the task's expert (De) and folded into Co: its update is projected onto the
  null space of the column space of `Co_{t-1}` and mixed in with square-root boundary scaling,
  `Co_t = rank_r(sqrt((t-1)/t) Co_{t-1} + 1/sqrt(t) dW_t)`, where the retraction is an exact rank-8 SVD, so Co never grows.
- Checkpoints store only the experts and the router statistics (replay-only): the fold is a deterministic function of the
  experts, so loading replays the folds in task order and rebuilds every `Co_{t-1}` and the deployed `Co_T`.
- An LDA router sends an example to its task expert, stacked on its `Co_{t-1}`, when it is confident, and to the shared
  branch otherwise. For tasks with candidate labels, a PMI-calibrated fusion of the two branches (on by default) can
  replace generation.

Adapters use upstream [`peft`](https://github.com/huggingface/peft); nothing is vendored. See [docs/DESIGN.md](docs/DESIGN.md).

## Installation

```bash
pip install -e .            # torch, transformers==5.12.1, peft==0.20.0, PyYAML, sentencepiece, TRACE metric packages
pip install -e ".[dev]"     # + pytest, pytest-xdist, ruff
pip install -e ".[qwen35]"  # optional: fast kernels for Qwen3.5's linear attention
```

The paper runs used torch 2.10.0 (CUDA 12.8), transformers 5.12.1 and peft 0.20.0 (Python >= 3.10).

## Quickstart (CPU, no downloads)

`tests/tinylab.py` builds randomly initialised 2-layer T5 and Qwen models with a synthetic tokenizer and a toy
three-task benchmark, so the whole pipeline runs offline in seconds:

```bash
python -m tests.tinylab lab                                    # tiny models and data under ./lab
python -m codelora.train tiny/quickstart methods/code-lora     # train 3 tasks, evaluate, write outputs/run/
python -m codelora.train tiny/quickstart methods/o-lora name=olora
python -m codelora.evaluate outputs/run --reload               # audit: recompute AP/BWT, reload checkpoint.pt, re-score
```

The scores are meaningless (random models); the point is that every code path runs. `pytest` covers the same paths.

## Data

The data is part of the repository, so nothing has to be downloaded or prepared:

| path | contents |
|---|---|
| `data/CL_Benchmark_repaired` | the 15 classification datasets of the Standard (order 1-3) and Long (order 4-6) benchmarks: `train` / `dev` / `test` / `labels`, `manifest.json`, `SHA256SUMS` |
| `data/TRACE` | the 8 TRACE tasks used (C-STANCE, FOMC, MeetingBank, Py150, ScienceQA, NumGLUE-cm, NumGLUE-ds, 20Minuten): `train` / `eval` / `test` |

The data root is `<repo>/data` unless `CODELORA_DATA` / `CODELORA_TRACE` is set. The classification benchmark replaces the
original dev split (which equals the test split) by a train-derived one; the test split is untouched
([data/README.md](data/README.md), [docs/DATA.md](docs/DATA.md)). **The datasets have their own licences, some
non-commercial: read [DATA_LICENSES.md](DATA_LICENSES.md) before using or redistributing them.**

## Models

```bash
export HF_TOKEN=...                      # Llama-2 is gated: accept its licence on the Hub first
python scripts/download_models.py        # T5-large, Qwen3-0.6B, Llama-2-7B, Qwen3.5-4B -> models/
python scripts/download_models.py t5-large qwen3-0.6b    # a subset (no token needed)
```

Models are loaded from `models/{t5-large,Qwen3-0.6B,Llama-2-7b-hf,Qwen3.5-4B}` (override with `CODELORA_MODELS` or
`backbone.path`) in bf16.

## Training and evaluation

A run is a composition of config fragments under `configs/` plus optional `key=value` overrides:

```bash
# T5-large, Standard CL order 1, CoDe-LoRA, seed 41 (one GPU)
python -m codelora.train orders/standard-order1 protocols/standard/t5-large methods/code-lora seed=41

# the same on 8 GPUs of one node
torchrun --nproc_per_node=8 -m codelora.train orders/standard-order1 protocols/standard/t5-large methods/code-lora seed=41

# other backbones / benchmarks / methods: swap the fragments
python -m codelora.train orders/long-order4 protocols/long/llama2-7b methods/mole-cie
python -m codelora.train orders/trace protocols/trace/qwen3-0.6b methods/o-lora
```

| fragment | contents |
|---|---|
| `orders/*` | the task sequence and data root (`standard-order1..3`, `long-order4..6`, `trace`) |
| `protocols/{standard,long,trace}/<backbone>` | backbone, batch sizes, learning rate, 1-epoch budget, evaluation batches and router |
| `methods/*` | method hyper-parameters |
| `routers/{cosine,lda}` | optional: switch the router (every protocol defaults to LDA) |
| `evaluation/no-fusion` | CoDe-LoRA: generate instead of fusing candidate-label scores (list it after `methods/code-lora`) |
| `logging/analysis` | full accuracy matrix and forward transfer, for the analysis figures |

Results go to `outputs/<name>/`: `summary.json` (performance matrix, AP, BWT, efficiency), `evaluations/<stage>/<task>.json`
(every prediction with its route and score), `results.json`, `config.yaml` (the resolved config) and `checkpoint.pt`; the
layout is in [docs/RESULTS.md](docs/RESULTS.md). Every protocol trains one epoch per task at an effective batch of 64 in
bf16 ([docs/CONFIG_PRINCIPLES.md](docs/CONFIG_PRINCIPLES.md)); `train.micro_batch_size` is the global batch of one forward
pass, split over the ranks, so the same config runs on 1 or N GPUs.

**PMI fusion.** `methods/code-lora` sets `fusion.enabled: true`: for tasks with candidate labels, both branches score every
label, each is calibrated by PMI against its mean predicted label probability on the task's dev split, and the argmax of
`alpha * PMI_Co + (1 - alpha) * PMI_expert` is predicted (`alpha` from `{0, 0.1, 0.25, 0.5, 1}`, chosen on dev). By default
(`fusion.stages=final`) this is applied only to the last evaluation stage, the row that defines AP; intermediate stages
generate, so the diagonal entries `a_ii` and BWT contain the difference between the two decoding rules. `fusion.stages=all`
fuses every stage. Pure generation is `evaluation/no-fusion` (`fusion.enabled=false`): AP, BWT and the diagonal then all use
generation. TRACE has no candidate labels and always generates.

`python -m codelora.reproduce` expands the paper's two experiment tables (`table1`: T5-large on Standard and Long;
`table2`: the three decoder backbones on Standard, Long and TRACE) into the commands above with the shipped settings and no
overrides: it prints them by default, `--run [--gpus N]` executes them (finished runs are skipped), `--generation` makes
CoDe-LoRA generate, and `--summarize` collects AP / BWT as mean and standard deviation over seeds 41, 42, 43.
Analysis tables and figures: `pip install -e ".[analysis]"` and [docs/ANALYSIS.md](docs/ANALYSIS.md).

## Implemented methods

The methods below are **re-implementations in one common framework** (same data, prompts, backbones, router interface,
budget and metrics), not the official code; the original repositories are linked.

| `method` | method | venue | original code |
|---|---|---|---|
| `lora` | sequential fine-tuning of one LoRA adapter (LoRA: Hu et al.) | ICLR 2022 | [microsoft/LoRA](https://github.com/microsoft/LoRA) |
| `o-lora` | O-LoRA: orthogonal subspace learning (Wang et al.) | Findings of EMNLP 2023, [paper](https://aclanthology.org/2023.findings-emnlp.715/) | [cmnfriend/O-LoRA](https://github.com/cmnfriend/O-LoRA) |
| `n-lora` | N-LoRA: Is Parameter Collision Hindering Continual Learning in LLMs? (Yang et al.) | COLING 2025, [paper](https://aclanthology.org/2025.coling-main.286/) | [PKU-YuanGroup/N-LoRA](https://github.com/PKU-YuanGroup/N-LoRA) |
| `mole-cie` | MoLE-CIE: Mixture of LoRA Experts for Continual Information Extraction with LLMs (Wang, Wang, Hu) | Findings of EMNLP 2025, [paper](https://aclanthology.org/2025.findings-emnlp.718/) | [nju-websoft/MOLE-CIE](https://github.com/nju-websoft/MOLE-CIE) |
| `co-lora` | ablation of this work: the consolidated shared branch alone (growing-rank N-LoRA training with SVD retraction to rank `r`) | this paper | this repository |
| `de-lora` | ablation of this work: one isolated expert per task plus the router | this paper | this repository |
| `code-lora` | CoDe-LoRA | EMNLP 2026, [arXiv](https://arxiv.org/abs/2610.08312) | this repository |

CLoRA (Lu et al., *Controlled Low-Rank Adaptation with Subspace Regularization for Continued Training on Large Language
Models*, ACL 2025, [paper](https://aclanthology.org/2025.acl-long.940/), code [sutakori/CLoRA](https://github.com/sutakori/CLoRA))
is compared in the paper but is not included in this repository.

## Repository structure

```
codelora/
  train.py evaluate.py reproduce.py runner.py config.py distributed.py tracking.py profiling.py
  data/        task loading, official prompt, collators, TRACE metrics, benchmark preparation
  models/      backbone loading, PEFT helpers, growing-rank adapter, MoLE layer
  methods/     interface + registry, lora, growing (O/N/Co-LoRA), delora, codelora, mole_cie, co_branch, de_branch
  routing/     router interface + registry, cosine prototypes, LDA, routing embeddings
  training/    the training loop, optimizer, random streams
  evaluation/  grouped batched generation, candidate-label scores + PMI fusion, AP / BWT, forward transfer
  analysis/    tables and figures from run directories
configs/       orders/ protocols/ methods/ routers/ evaluation/ logging/ tiny/
data/          CL_Benchmark_repaired/  TRACE/  (see data/README.md)
docs/          DESIGN, EXTENDING, CONFIG_PRINCIPLES, DATA, RESULTS, ANALYSIS
scripts/       download_models.py
tests/         unit tests, tiny-model smoke, tinylab (offline tiny models and data)
```

Tests: `pytest` (CPU, tiny models, a few minutes; `pytest -n 4` with pytest-xdist), `ruff check . && ruff format --check .`.

## Extending

Adding a method, backbone, benchmark or router is described in [docs/EXTENDING.md](docs/EXTENDING.md).

## Citation

```bibtex
@article{liu2026codelora,
  title   = {{CoDe-LoRA}: Mitigating the Orthogonality Dilemma in Continual Learning of {LLMs} via Knowledge Consolidation and Decoupling},
  author  = {Liu, Maoqi and Fang, Quan and He, Yufei},
  journal = {arXiv preprint arXiv:2610.08312},
  year    = {2026},
  note    = {Accepted to EMNLP 2026}
}
```

See also [CITATION.cff](CITATION.cff).

## License

The code is released under the Apache License 2.0 ([LICENSE](LICENSE)). The datasets in `data/` are not covered by it; see
[DATA_LICENSES.md](DATA_LICENSES.md).

## Acknowledgements

Built on [PyTorch](https://pytorch.org), [Transformers](https://github.com/huggingface/transformers) and
[PEFT](https://github.com/huggingface/peft). The classification benchmark follows the continual-learning benchmark of
[O-LoRA](https://github.com/cmnfriend/O-LoRA) and [N-LoRA](https://github.com/PKU-YuanGroup/N-LoRA); the TRACE data and
metrics come from [TRACE](https://github.com/BeyonderXX/TRACE).
