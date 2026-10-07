# Run artifacts (schema version 1)

Every run writes one directory, `<output_dir>/<name>/`, written by rank 0 only (evaluation is sharded over ranks and
merged before anything is written). Everything the analysis needs comes from this directory; nothing has to be re-run
to draw a figure. All JSON files carry `"schema_version": 1`; a field is only ever added, never changed, within a version.

| file | content | cost |
|---|---|---|
| `config.yaml` | the resolved config (every default filled in) | none |
| `run.json` | identity, environment, parameter counts, timestamps | none |
| `log.jsonl` | structured event log (below) | one line per `log.interval` steps |
| `train.log` | the same events as readable text | same |
| `summary.json` | per-task statistics, performance matrix, AP / BWT / FWT, efficiency | none |
| `results.json` | analysis tables: matrix, per-class scores, routing counts, label sets | negligible |
| `evaluations/<stage>/<task>.json` | one record per evaluated example (below) | negligible |
| `checkpoint.pt`, `checkpoints/after_task_NNN.pt` | model state, by `save_checkpoint` | one write per saved stage |
| `tb/` | TensorBoard events (`log.tensorboard`) | optional |

Standard output keeps the JSON `train_step` line of every optimizer step (`log.stdout_interval`, default 1).

## `run.json`

```json
{"schema_version": 1, "name": "...", "method": "code-lora", "seed": 41, "world_size": 8, "config_sha256": "...",
 "environment": {"python": "3.11.9", "platform": "Linux-...", "torch": "2.10.0", "transformers": "5.12.1", "peft": "0.20.0",
                 "cuda": "12.8", "gpus": ["NVIDIA A100-SXM4-80GB", "..."], "git_commit": "abc123 or null"},
 "parameters": {"base": 737668096, "adapter_after_task": {"dbpedia": 2359296, "amazon": 4718592}},
 "started_at": 1790000000.0, "finished_at": 1790003600.0}
```
`adapter_after_task[t]` counts every parameter and growing-rank history buffer that is not part of the frozen base model.

## `log.jsonl`

One JSON object per line with `t` (unix seconds) and `event`:

| event | fields |
|---|---|
| `run_start` | `name`, `method`, `seed`, `world_size` |
| `task_start` | `index`, `task`, `n_train` |
| `train` | `index`, `task`, `step`, `branches` (`[{adapter, loss, penalty, grad_norm}]`, global values), `lr`, `loss_mean` (mean loss of the branches over the last `log.interval` steps) |
| `task_end` | `index`, `task`, `steps`, `train_seconds`, `trainable_parameters`, `adapter_parameters`, `peak_memory_mib` (CUDA max allocated since the task began, 0 on CPU) |
| `eval` | `stage`, `task`, `score`, `n`, `seconds` |
| `stage_end` | `stage`, `eval_seconds` |
| `run_end` | `AP`, `BWT`, `FWT`, `total_seconds` |

`penalty` is the regulariser of that branch (N-LoRA / O-LoRA / Co-LoRA L1 or orthogonality term, MoLE-CIE gate loss; 0 when the
branch has none). Every method but MoLE-CIE trains one branch per step.

## `summary.json`

Keys of version 1: `status`, `method`, `seed`, `world_size`, `versions`, `tasks` (list: `task`, `train_seconds`, `steps`,
`loss`, `penalty`, ... method statistics), `performance_matrix` (`{after_task_NNN: {task: score}}`), `average_performance`
(AP), `backward_transfer` (BWT), `diagonal_average_performance`, `diagonal_scores`, `evaluation_seconds`, `peak_allocated_mib`,
`forward_transfer_*` matrices (with `eval.forward_transfer`), plus

* `schema_version`, `FWT`: mean over tasks `i >= 2` of (score on task `i` before it is learned, i.e. after stage `i-1`) minus the
  untrained model's score; `null` unless `eval.forward_transfer` is on;
* `efficiency`: `train_seconds`, `eval_seconds`, `total_seconds` (wall clock of the whole run, device-synchronised), `peak_memory_mib`,
  `base_parameters`, `adapter_parameters` (final, stored), `adapter_fraction` (adapter / base), `trainable_parameters` (max over tasks).

`eval.matrix: diagonal_final` (default) scores `a_ii` and the final row only, which is all that AP and BWT need.
`eval.matrix: full` scores every learned task after every stage (needed for the AP-vs-number-of-tasks curves; evaluation
costs up to `T/2` times more), and `eval.forward_transfer: true` additionally scores the unseen tasks.

## `results.json`

```json
{"schema_version": 1, "method": "code-lora", "seed": 41, "tasks": ["dbpedia", "amazon"], "eval_split": "test",
 "matrix_kind": "diagonal_final", "performance": {"after_task_001": {"dbpedia": 0.98}},
 "metrics": {"AP": 0.9, "BWT": -0.01, "FWT": null, "diagonal_average": 0.91},
 "per_class": {"after_task_002": {"dbpedia": {"Company": {"n": 100, "score": 0.97}}}},
 "routing": {"after_task_002": {"dbpedia": {"counts": {"dbpedia": 990, "shared": 10}, "mean_confidence": 0.93}}},
 "label_sets": {"dbpedia": ["Company", "School"]}}
```
* `per_class`: mean score and count per gold answer, only for tasks with at most 50 distinct answers (classification).
* `routing`: for routed methods (de-lora, code-lora, mole-cie) how many examples of the evaluated task were sent to each
  route (`shared` = CoDe-LoRA's fallback branch) and the mean router confidence. The confusion matrix of a stage is the stack of its tasks.
* `label_sets`: candidate answers per task, present with `fusion.enabled`.
* `efficiency`: the `summary.json` block above plus `per_task`, one entry per task: `train_seconds`, `eval_seconds` (the stage
  that follows the task; `eval_task_seconds` splits it per evaluated task), `train_peak_memory_mib`, `eval_peak_memory_mib`,
  `adapter_parameters`, `trainable_parameters`, and `train_phases` / `eval_phases` (`{phase: {seconds, calls}}`, empty unless
  `log.profile`). All timings synchronise the device before reading the clock.

`log.profile: true` adds the phase breakdown (data loading, adapter activation, forward/backward per branch, loss read-out, gradient
all-reduce, clip and optimizer step, consolidation, routing registration, routing/generation during evaluation). It synchronises
the device at every phase boundary and therefore slows the run; use it to find where the time goes, not to time a run.

## `evaluations/<stage>/<task>.json`

A JSON list, one record per evaluated example in evaluation order:

```json
{"index": 17, "input": "...", "target": "Company", "prediction": "company", "route": "dbpedia", "confidence": 0.97, "score": 1.0}
```
`index` is the example's position in the original split file (before `max_eval_samples` sub-sampling and before
`test_exclusions.json`). `route` / `confidence` are `null` for unrouted methods. With `fusion.enabled` (CoDe-LoRA,
tasks with candidate labels) `prediction` is the fused answer and every record also has
`fusion`: `{"alpha": 0.25, "co": [..], "expert": [..]}`, the task's dev-chosen Co weight and the log-probability of every
candidate answer (order of `label_sets[task]`, sum over answer tokens, EOS included) under the shared branch alone and under
the routed expert alone (the shared branch again when the example fell back to it). Cost: `2 x |labels|` teacher-forced
forward passes per evaluated example, plus the same on the task's dev split once per stage (14 labels for DBpedia, about 28
passes per example).

## Checkpoints (`save_checkpoint`)

`none`; `final` (default: `checkpoint.pt` at the end); `all` (`checkpoints/after_task_NNN.pt` after every task plus `checkpoint.pt`);
`every:N` (every N-th task and the last). `python -m codelora.evaluate <run> --reload` loads `checkpoint.pt`.

## Overhead

Default settings add a few JSON lines per `log.interval` steps and one pass over the evaluation records at the end of each stage
(milliseconds). Switches that cost real time: `eval.matrix=full` (evaluation time), `eval.forward_transfer` (evaluation of
unseen tasks), `fusion.enabled` (about `1 + 2 x |labels|` times the routed forward passes of generation, teacher-forced passes
being cheaper than generation, plus the dev split of every evaluated task), `save_checkpoint=all` (checkpoint size times tasks).
