# Design notes

CoDe-LoRA is implemented as a small library: one continual loop (`runner.py`) that drives any method behind a four-method
interface, a shared multi-branch training loop, and a handful of reusable components (consolidation, experts, routers,
candidate-label fusion). This document explains how the pieces fit; [EXTENDING.md](EXTENDING.md) shows how to add to them.

## 1. Package

```
codelora/
  config.py        typed Config (dataclasses) + fragment composition (defaults, per_method, key=value overrides)
  train.py         python -m codelora.train  (single GPU) / torchrun ... -m codelora.train  (single-node data parallel)
  evaluate.py      audit a run: recompute AP/BWT, check cells against per-example scores, reload the checkpoint
  reproduce.py     table1 / table2 -> runs -> commands | execution | results table
  runner.py        the continual loop: learn task -> score stage -> summary.json  (one code path for every method)
  distributed.py   replicated model + sharded work; identity when WORLD_SIZE == 1
  tracking.py      structured logs, run.json, efficiency counters;  profiling.py  phase timers (log.profile)
  data/            tasks (loading, official prompt, scoring), collate, trace (TRACE prompt + metrics), prepare
  models/          backbone (load, LoRA targets), adapters (PEFT helpers), growing (growing-rank LoRA), mole_layer
  methods/         base (interface + registry), lora, growing (o/n/co-lora), delora, codelora, mole_cie,
                   co_branch (fold), de_branch (experts + router), regularizers, consolidation (low-rank SVD)
  routing/         router (interface + registry), prototype (cosine), lda, embed (frozen-base sentence embeddings)
  training/        loop (multi-branch), optim, rng
  evaluation/      generate (grouped, sharded), branch_scores + fusion (candidate labels), metrics (AP / BWT),
                   forward_transfer (+ callbacks), results
  analysis/        tables and figures from run directories
```

## 2. Method interface

```python
class ContinualMethod:
    branches: int                      # optimizer updates per training iteration (default 1)
    setup()                            # build the adapter structure on the loaded base model
    learn_task(index, task, examples)  # train (+ consolidate, + register routing statistics); returns statistics
    predict(examples) -> [Prediction]  # task-agnostic inference (routing / adapter composition)
    state_dict() / load_state_dict()   # checkpoint
```

`@register("code-lora")` fills `METHODS`; `runner.run` only uses this interface, so a method never touches the loop,
the metrics or the logging.

| method | implementation |
|---|---|
| `lora` | `methods/lora.py`: one adapter `shared`, trained on every task |
| `o-lora`, `n-lora`, `co-lora` | `methods/growing.py`: one growing-rank adapter (`models/growing.py`: frozen history blocks + a trainable current block). The three differ in the penalty and at the boundary: O-LoRA (orthogonality) and N-LoRA (L1) commit the block, Co-LoRA (N-LoRA penalty) replaces history + block by their rank-`r` SVD. Official O-LoRA / N-LoRA seed protocol: re-seed before every block, same data order every task |
| `de-lora` | `methods/delora.py` over `DeBranch`: a frozen expert per task, selected by the router |
| `code-lora` | `methods/codelora.py` over `CoBranch` + `DeBranch`, see section 3 |
| `mole-cie` | `methods/mole_cie.py` over `models/mole_layer.py`: mixture of LoRA experts, task keys, replay, gate reflection (two data streams, so it has its own training loop) |

## 3. CoDe-LoRA

**Training (one pass).** At task `t` one plain LoRA adapter is trained on top of the frozen consolidated branch
`Co_{t-1}` (`Branch([shared, expert_t], trainable=expert_t)`; no penalty). Afterwards it is

1. frozen and registered as expert `t` with the router (statistics from a seeded support subset of the task's training
   data, computed with the frozen base model), and
2. folded into the shared branch: `Co_t = rank_r(c_t Co_{t-1} + s_t E_t^perp)`, the exact truncated SVD of the sum
   computed on the low-rank factors (QR of the joined factors, SVD of the small core, float32). The branch never grows.
   `E_t^perp = (B - U U^T B) A` is the expert's update with its component along the column space of `Co_{t-1}` removed
   (`U`: left singular vectors of `Co_{t-1}` above `1e-6` of the largest; `colora.projection: null_space`), and
   `c_t = sqrt((t-1)/t)`, `s_t = 1/sqrt(t)` (`colora.scaling: sqrt`; `c_1 = 0`, so Co_1 is the first expert). The
   ablations `colora.projection: none` and `colora.scaling: additive` (`c_t = s_t = 1`) are one-line switches; the
   experts themselves are applied on `Co_{t-1}` either way, so only `Co_T` (the Co-only predictions and the Co side of
   the PMI fusion) depends on the fold.

Why the expert is trained on `Co_{t-1}`: increments that are folded into one shared branch only add up when each was
learned in the context of the earlier ones; adapters learned independently on the bare base are optimised for a model
without the others and their sum collapses. Consequently the same adapter is the expert, and an expert is only valid in
the context it was trained in.

**Inference.** The router (LDA by default) maps an example to a task and a confidence. If the confidence exceeds
`router.threshold` the example is served by that task's expert stacked on its context `Co_{t-1}`, otherwise by the
shared branch `Co_T`. The LDA protocols use threshold 0, i.e. always the routed expert.

**Checkpoints store the experts only.** The fold is a deterministic function of the experts, so a checkpoint holds the
experts and the router statistics; `load_state_dict` replays the folds in task order, which rebuilds every `Co_{t-1}`
(kept in memory as frozen adapters, one adapter-sized tensor set per task) and the deployed `Co_T`.

**Candidate-label fusion (`fusion.enabled`, on in `methods/code-lora`).** For tasks with a closed label set CoDe-LoRA can score the candidate
answers instead of generating (`evaluation/fusion.py`). Both branches score every candidate (`evaluation/branch_scores.py`:
sum of answer-token log-probabilities, EOS included, no length normalisation) and are calibrated by pointwise mutual
information against their own prior, the branch's mean predicted label probability on the task's dev split,

```
PMI_b(c|x)  = log_softmax_c( log p_b(c|x) - log(q_b(c) + 1e-9) ),      b in {Co, expert}
score(c|x)  = alpha * PMI_Co(c|x) + (1 - alpha) * PMI_expert(c|x)
```

The Co weight `alpha` is chosen per task on dev from `fusion.alphas` (default `{0, 0.1, 0.25, 0.5, 1}`), ties to the
smaller alpha; priors and alpha are fitted once per model state on dev only and applied unchanged to the evaluation
split. Open generation (TRACE, no `labels.json`) is never fused. The dev split is the train-derived one produced by
`codelora.data.prepare` (see [DATA.md](DATA.md)).

`fusion.stages` (default `final`) decides which evaluation stages are fused. With `final` the candidate-label scoring and
the dev fits run only at the last stage, i.e. the final row of the performance matrix that defines AP; the intermediate
stages generate, which is what the diagonal entries `a_ii` are. This removes the label scoring of every intermediate
stage, which dominates the evaluation time. Consequence for BWT: BWT compares the final row with the diagonal, so under
`final` the final row is fused and the diagonal is generated, and BWT then also contains the difference between the two
decoding rules. With `fusion.stages=all` every stage is fused (diagonal and final row alike, the BWT of one decoding
rule), at the cost of one label-scoring pass per stage. The generation-only numbers of the paper's tables correspond to
`fusion.enabled=false` (fragment `evaluation/no-fusion`), where AP, BWT and the diagonal all use generation.

## 4. Training loop and multi-GPU

`training/loop.py` serves every method except MoLE-CIE: a list of `Branch(adapters, trainable, penalty, rng_role)`.
The `grad_accum` micro-batches of an optimizer step visit the branches in order (`adapters.BranchSwitch` changes the
active adapters once per branch and step); then each branch does all-reduce, clip, step. Losses, penalties and gradient
norms stay on the device and are read back once per optimizer step. Dropout of a branch draws from a stream keyed by
`(seed, micro_step, role)` (`training/rng.py`), so adapter creation never perturbs the training stream and corresponding
runs see identical masks.

Data parallelism is "replicated model, sharded work", not DDP: the trainable branch changes inside a step and the
active adapter set changes between forwards, which DDP's reducer cannot express, and adapters are a few MB, so one flat
all-reduce is simpler. Each rank takes the same global micro-batch (shared DataLoader seed) and a contiguous slice; the
slice's loss is weighted by its share of the loss tokens, so `sum_r w_r * mean_loss_r` equals the single-process token
mean. Generation, label scoring, routing embeddings and MoLE key extraction are strided over ranks (`sharded_map`) and
gathered in order. `micro_batch_size` is the global batch, so one config is statistically equivalent on any rank count.
With `WORLD_SIZE=1` every collective is the identity and the code path is the same.

## 5. Configuration

`configs/orders/*` (task sequence + data root), `configs/protocols/<benchmark>/<backbone>` (backbone, batch sizes,
budget, eval batches, router), `configs/methods/*`, `configs/routers/*` and `configs/evaluation/no-fusion`. A config is
the deep merge of fragments; `per_method:` blocks carry the few per-method deviations (see
[CONFIG_PRINCIPLES.md](CONFIG_PRINCIPLES.md)). Protocols train `train.epochs: 1` per task (`train.max_steps: null`); a
number in `train.max_steps` caps the optimizer updates per task instead. Unknown keys and misspelled enumerated values are rejected at load time.

## 6. Numerics (bf16)

Base weights are bf16; PEFT keeps the adapters in fp32, so the AdamW state is fp32. Kept in high precision: SVD / QR
consolidation (fp32), LDA statistics and solve (float64), routing embeddings (fp32), loss and token counts (float64),
fusion statistics (float64). MoLE-CIE's experts, router and keys follow the backbone dtype.

## 7. Evaluation protocol

Every method is evaluated with the upstream O-LoRA / N-LoRA protocol when the official prompt is used (T5 inputs
pre-truncated by a tokenize/decode round trip, generation capped with `max_length` and decoded with clean-up, exact match
ignoring punctuation; `data.official_protocol`). `eval.matrix: diagonal_final` scores `a_ii` and the final row, which is
all AP and BWT need; `full` scores every learned task after every stage; `eval.forward_transfer` additionally scores the
unseen tasks against the untrained model.

## 8. Data

`python -m codelora.data.prepare` builds the repaired benchmark (a train-derived dev split, the test split untouched) and
writes `manifest.json` and `SHA256SUMS`. See [DATA.md](DATA.md).
