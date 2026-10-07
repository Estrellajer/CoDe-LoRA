# Data shipped with the repository

**Licences:** the datasets have their own licences, several of them non-commercial or research-only. Read
[../DATA_LICENSES.md](../DATA_LICENSES.md) before using or redistributing anything in this directory; the Apache-2.0 licence
of the code does not apply to it. Files are byte-for-byte the processed files used for the paper's runs.

## `CL_Benchmark_repaired/`

The classification benchmark of the continual-learning papers O-LoRA and N-LoRA (15 datasets in 8 task families:
`TC` agnews, dbpedia, yahoo; `SC` amazon, yelp, IMDB, SST-2; `NLI` MNLI, CB, RTE; `QQP`; `BoolQA`; `MultiRC`; `WiC`;
`COPA`), derived from `CL_Benchmark` of [cmnfriend/O-LoRA](https://github.com/cmnfriend/O-LoRA) (commit
`07117e1fc4a5f5ad9308a815a42cee8f46502dc8`). Every dataset directory holds `train.json`, `dev.json`, `test.json` (lists of
`{"sentence", "label"}`) and `labels.json` (the answer options shown in the prompt); `manifest.json` lists counts, label
counts, split policy and file hashes, `SHA256SUMS` the hashes of every file (`shasum -a 256 -c SHA256SUMS`).

**The dev-split repair.** In the original benchmark `dev.json` is identical to `test.json`, so tuning on it would tune on
the test set. `python -m codelora.data.prepare` therefore builds a new dev split: 10 % of the original training records per
label (stratified), chosen by the smallest SHA-256 rank of `(seed 42, dataset, index, record)`, removed from `train.json`.
`test.json` is a byte-identical copy of the original test split (checked against the original files for all 15 datasets);
dev and test share no record. MNLI training records with the label `-` (not in `labels.json`) are dropped before the split;
the 131 MNLI test records with that label stay in `test.json` and are skipped when the test split is loaded
(`test_exclusions.json`). Dev is only used for the PMI-fusion fit (`fusion.*`) and `eval.split=dev`; reported scores use test.

| split | size |
|---|---|
| train | 224 (CB) to 12 600 (DBpedia) records, see `manifest.json` |
| dev | 26 (CB) to 1 400 (DBpedia) |
| test | the original test split: 56 (CB) to 7 600 |

## `TRACE/`

The eight tasks of the TRACE benchmark order used in the paper, from the official `LLM-CL-Benchmark_5000` release
(archive SHA-256 `11152d50f8a093b1fc9c6c924ec207915d173b9e2d715396ec8f8a837e2668a8`, code revision
`462e39f616134f4f819efeb3baea8638c03c7db4` of [BeyonderXX/TRACE](https://github.com/BeyonderXX/TRACE)): C-STANCE, FOMC,
MeetingBank, Py150, ScienceQA, NumGLUE-cm, NumGLUE-ds, 20Minuten. Only the files the code reads are included:
`<task>/train.json`, `eval.json`, `test.json` (lists of `{"prompt", "answer"}` records), unmodified.

| task | train | eval | test |
|---|---|---|---|
| C-STANCE | 5000 | 2000 | 2000 |
| FOMC | 5000 | 496 | 496 |
| MeetingBank | 5000 | 687 | 692 |
| Py150 | 5000 | 2000 | 2000 |
| ScienceQA | 5000 | 2000 | 2000 |
| NumGLUE-cm | 5000 | 41 | 81 |
| NumGLUE-ds | 5000 | 164 | 325 |
| 20Minuten | 5000 | 200 | 200 |

`SHA256SUMS` lists the hash of every file; the hashes of the upstream files are identical. Verify with
`cd data/TRACE && shasum -a 256 -c SHA256SUMS` (on Linux `sha256sum -c`). The largest file is
`MeetingBank/train.json` (83 MB).
