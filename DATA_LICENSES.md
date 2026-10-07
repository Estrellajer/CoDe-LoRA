# Data licences

The code of this repository is Apache-2.0 ([LICENSE](LICENSE)). **The files in `data/` are third-party datasets and are not
covered by that licence.** Each keeps the terms of its source. This file records what could be established on 2026-10-07 from
the dataset pages, papers and repositories; it is not legal advice. "To be verified" marks what could not be confirmed.

## Read this first: restrictive or unresolved terms

Redistributing these inside a public repository is questionable. Decide before making the repository public.

**Non-commercial / research-only / no redistribution stated by the source**

- **AG News**: provided "for research purposes" and for "any non-commercial use"; the corpus page asks users to contact the author.
- **Yahoo! Answers**: Yahoo Webscope data-sharing agreement, academic research only, gated download.
- **Yelp**: Yelp Dataset User Agreement, academic research only, and it bars distributing the data.
- **WiC**: CC BY-NC 4.0.
- **MultiRC**: "Research and Academic Use License".
- **FOMC** (TRACE): CC BY-NC 4.0.
- **MeetingBank** (TRACE): CC BY-NC-SA 4.0.
- **ScienceQA** (TRACE): CC BY-NC-SA 4.0 in the dataset card's licensing section (the card metadata says CC BY-SA 4.0; the two conflict, treat as non-commercial).

**Share-alike (attribution and same licence for derivatives)**: DBpedia (CC BY-SA 3.0 and GFDL), BoolQ (CC BY-SA 3.0), MNLI (parts: CC BY-SA 3.0), MeetingBank and ScienceQA (see above).

**No explicit licence found or not verifiable**: Amazon reviews, QQP, RTE, CB, COPA, SST-2, IMDB, C-STANCE, 20Minuten data, the Py150 variant used by TRACE, the component datasets of NumGLUE.

The classification benchmark in `data/CL_Benchmark_repaired` is a processed copy of the `CL_Benchmark` of
[cmnfriend/O-LoRA](https://github.com/cmnfriend/O-LoRA) (MIT-licensed code, no statement about the data), and the TRACE files are
a copy of the official TRACE release (Apache-2.0 repository code, no statement about the data bundle). Neither upstream
licence extends to the underlying datasets.

If a dataset cannot be redistributed, remove its directory and restore it from the original source with
`python -m codelora.data.prepare` ([docs/DATA.md](docs/DATA.md)); the checksums in `data/*/SHA256SUMS` identify the exact files.

## Classification benchmark (`data/CL_Benchmark_repaired`)

Processed as described in [data/README.md](data/README.md) (train/dev split repaired; `train` is a subsample of the original
training data as shipped by O-LoRA; `test` is the original test split). Zhang et al. (2015) = X. Zhang, J. Zhao, Y. LeCun,
*Character-level Convolutional Networks for Text Classification*, NeurIPS 2015.

| dataset | source | licence (as stated) | redistribution |
|---|---|---|---|
| AG News | AG's corpus ([Gulli](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)), version of Zhang et al. 2015 | research purposes, "any non-commercial use"; HF card `unknown` | non-commercial only; the source asks to be contacted |
| Amazon (polarity) | McAuley and Leskovec, [SNAP web-Amazon](https://snap.stanford.edu/data/web-Amazon.html), version of Zhang et al. 2015 | no licence on the SNAP page (the HF mirror claims Apache-2.0, not from the authors) | unclear, to be verified |
| Yahoo! Answers | Yahoo Webscope L6 (Comprehensive Questions and Answers), version of Zhang et al. 2015 | Webscope data-sharing agreement: academic research only | not allowed without the agreement |
| DBpedia | DBpedia ontology classes, version of Zhang et al. 2015 | CC BY-SA 3.0 and GFDL | yes, attribution and share-alike |
| Yelp (polarity) | [Yelp Open Dataset](https://www.yelp.com/dataset), version of Zhang et al. 2015 | Yelp Dataset User Agreement: academic research only, no distribution | not allowed |
| MNLI | Williams et al. 2018, [site](https://cims.nyu.edu/~sbowman/multinli/) (via GLUE) | mixed: OANC terms for most genres, CC BY-SA 3.0 / CC BY 3.0 / public domain for the fiction genre | yes, attribution; share-alike for part of the data |
| QQP | Quora Question Pairs (via GLUE) | no explicit licence (HF card `unknown`); Quora terms may apply | unclear, to be verified |
| RTE | RTE1/2/3/5 challenges (via GLUE/SuperGLUE) | no licence found; GLUE defers to the original licences | to be verified |
| CB | [CommitmentBank](https://github.com/mcdm/CommitmentBank) (via SuperGLUE) | no licence on the repository | unclear, to be verified |
| WiC | [Pilehvar and Camacho-Collados](https://pilehvar.github.io/wic/) (via SuperGLUE) | CC BY-NC 4.0 | non-commercial only |
| COPA | Roemmele et al. 2011 (via SuperGLUE) | could not be confirmed (page unreachable) | to be verified |
| BoolQ | Clark et al. 2019, [google/boolq](https://huggingface.co/datasets/google/boolq) | CC BY-SA 3.0 | yes, attribution and share-alike |
| MultiRC | Khashabi et al. 2018, [CogComp/multirc](https://github.com/CogComp/multirc) | "Research and Academic Use License" | non-commercial / research only |
| SST-2 | Socher et al. 2013, [Stanford SST](https://nlp.stanford.edu/sentiment/index.html) (via GLUE) | no licence stated | unclear, to be verified |
| IMDB | Maas et al. 2011, [site](https://ai.stanford.edu/~amaas/data/sentiment/) | no licence stated (citation requested) | unclear, to be verified |

## TRACE (`data/TRACE`)

TRACE: *A Comprehensive Benchmark for Continual Learning in Large Language Models* (Wang et al.), repository
[BeyonderXX/TRACE](https://github.com/BeyonderXX/TRACE) (Apache-2.0 for the code; the README states nothing about the data
licences). The release bundles processed versions of the following datasets:

| task | underlying dataset | licence (as stated) | redistribution |
|---|---|---|---|
| C-STANCE | [chenyez/C-STANCE](https://github.com/chenyez/C-STANCE) (Weibo posts) | no licence in the repository | unclear, to be verified |
| FOMC | [gtfintechlab/fomc_communication](https://huggingface.co/datasets/gtfintechlab/fomc_communication) | CC BY-NC 4.0 | non-commercial only |
| MeetingBank | [huuuyeah/meetingbank](https://huggingface.co/datasets/huuuyeah/meetingbank) | CC BY-NC-SA 4.0 | non-commercial, share-alike |
| Py150 | ETH Py150 ([SRI Lab](https://www.sri.inf.ethz.ch/py150)); the redistributable [ETH Py150 Open](https://huggingface.co/datasets/google-research-datasets/eth_py150_open) is Apache-2.0 with per-file licences | which variant TRACE ships was not established | source files keep their own (permissive) licences; to be verified |
| ScienceQA | [derek-thomas/ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) | CC BY-NC-SA 4.0 (licensing section; metadata says CC BY-SA 4.0) | non-commercial, share-alike |
| NumGLUE-cm, NumGLUE-ds | [allenai/numglue](https://github.com/allenai/numglue) | ODC-By (the compilation); component dataset licences not traced | yes with attribution for the compilation; components to be verified |
| 20Minuten | [ZurichNLP/20Minuten](https://github.com/ZurichNLP/20Minuten) (Rios et al. 2021; Kew et al. 2023) | code Apache-2.0; no data licence stated; the articles are from the publisher 20 Minuten | unclear, probably not redistributable, to be verified |

The TRACE release also lists a replay set (LIMA) with its own licence; it is not used here and not included.

## Models

Model weights are not included. They are downloaded from the Hugging Face Hub under their own licences (T5-large:
Apache-2.0; Qwen3-0.6B and Qwen3.5-4B: see their model cards; Llama-2-7B: Meta's Llama 2 Community License, gated).
