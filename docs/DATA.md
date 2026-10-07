# Data

## Classification benchmarks (Standard CL order1-3, Long order4-6)

The 15 datasets of the O-LoRA / N-LoRA continual-learning benchmark (`CL_Benchmark`, from
[cmnfriend/O-LoRA](https://github.com/cmnfriend/O-LoRA), pinned commit `07117e1fc4a5f5ad9308a815a42cee8f46502dc8`).
Each dataset is a directory with `train.json`, `dev.json`, `test.json` (lists of `{"sentence", "label"}`) and
`labels.json` (the option list that the prompt shows to the model).

**The original `dev.json` is identical to `test.json`**, so scoring on it (for hyper-parameter or checkpoint selection)
measures the test set. We therefore never read the shipped dev split. `python -m codelora.data.prepare` builds
`CL_Benchmark_repaired`:

| | |
|---|---|
| `train.json` | the original training data minus a dev split |
| `dev.json` | 10 % of the training data per label, chosen by the smallest SHA-256 rank of `(seed, dataset, index, record)`; deterministic for `--seed` (default 42) |
| `test.json` | byte-identical copy of the original test split |
| `test_exclusions.json` | only for MNLI: the test records whose label `-` is not in `labels.json`; they are kept in `test.json` and skipped when the test split is loaded |
| `manifest.json`, `SHA256SUMS` | per-dataset counts, label counts, file hashes, split policy |

So dev and test share **no record** (the dev records are a subset of the original training set and the tests in
`tests/test_prepare.py` check train ∩ dev = ∅ and that the test file is untouched). MNLI training records with the
invalid label `-` are dropped before the split.

The repaired benchmark is shipped in this repository as `data/CL_Benchmark_repaired` (with `manifest.json` and
`SHA256SUMS`); nothing has to be downloaded or prepared. To rebuild it from the upstream files:

```bash
git clone --filter=blob:none --no-checkout https://github.com/cmnfriend/O-LoRA.git
git -C O-LoRA sparse-checkout set CL_Benchmark && git -C O-LoRA checkout 07117e1fc4a5f5ad9308a815a42cee8f46502dc8
python -m codelora.data.prepare --input-root O-LoRA/CL_Benchmark --output-root data/CL_Benchmark_repaired
```

The data root defaults to `<repo>/data`; set `CODELORA_DATA` to use another location. See [../data/README.md](../data/README.md)
for the provenance of the shipped files and [../DATA_LICENSES.md](../DATA_LICENSES.md) for the licences of the datasets.

`configs/orders/` lists the task orders: `standard-order1` = dbpedia, amazon, yahoo, agnews; `standard-order2` = dbpedia,
amazon, agnews, yahoo; `standard-order3` = yahoo, amazon, agnews, dbpedia; `long-order4/5/6` are the 15-task sequences.

Scores on the dev split (for tuning) are available with `eval.split=dev`; reported numbers use `test`.

## Prompt

With `data.prompt_style: nlora_official` (default for classification) every input is wrapped as in upstream O-LoRA/N-LoRA:

```
Task:TC
Dataset:dbpedia
What is the topic of the following paragraph? Choose one from the option.
Option: Company, School, ... 
<sentence>
Answer:
```

The instruction depends on the task family (the parent directory name: `NLI`, `QQP`, `SC`, `TC`, `BoolQA`, `MultiRC`,
`WiC`, `COPA`; `COPA` has none). `data.add_task_name` / `data.add_dataset_name` drop the first two lines. Decoder-only
models see `prompt + " " + answer + EOS`, with the loss on the answer only.

## TRACE

The 8-task TRACE order used in the paper (`C-STANCE, FOMC, MeetingBank, Py150, ScienceQA, NumGLUE-cm, NumGLUE-ds,
20Minuten`), official release revision `462e39f`. Each task directory holds `train.json`, `eval.json`, `test.json`
(`prompt` / `answer` records); TRACE prompts are used verbatim (`prompt_style: plain`) and scored with the task's official
metric (accuracy, ScienceQA accuracy, ROUGE-L, edit similarity, SARI). The eight task folders are shipped in `data/TRACE` (only the `train` / `eval` / `test` files that the code
reads); `CODELORA_TRACE` points elsewhere.

## Models

Local directories (no network access at run time): `t5-large`, `Qwen3-0.6B`, `Llama-2-7b-hf`, `Qwen3.5-4B` under
`models/` of the repository (`python scripts/download_models.py`; `CODELORA_MODELS` or `backbone.path` change the location).
All are loaded in bf16.
