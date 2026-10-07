# Data licences

The code of this repository is Apache-2.0 ([LICENSE](LICENSE)). The files in `data/` are third-party datasets: each keeps the
terms of its source, listed below. They are included so that the experiments run out of the box, as in the benchmark
repositories they come from ([O-LoRA](https://github.com/cmnfriend/O-LoRA), [TRACE](https://github.com/BeyonderXX/TRACE)).
If you are a rights holder and want a dataset removed, please open an issue. Original files can be restored from the sources
with `python -m codelora.data.prepare` ([docs/DATA.md](docs/DATA.md)); `data/*/SHA256SUMS` identify the exact files.

## Classification benchmark (`data/CL_Benchmark_repaired`)

Processed as described in [data/README.md](data/README.md) (train/dev split repaired; `train` is a subsample of the original
training data as shipped by O-LoRA; `test` is the original test split). Zhang et al. (2015) = X. Zhang, J. Zhao, Y. LeCun,
*Character-level Convolutional Networks for Text Classification*, NeurIPS 2015.

| dataset | source | licence (as stated) |
|---|---|---|
| AG News | AG's corpus ([Gulli](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)), version of Zhang et al. 2015 | research purposes, "any non-commercial use" |
| Amazon (polarity) | McAuley and Leskovec, [SNAP web-Amazon](https://snap.stanford.edu/data/web-Amazon.html), version of Zhang et al. 2015 | no licence stated on the SNAP page |
| Yahoo! Answers | Yahoo Webscope L6 (Comprehensive Questions and Answers), version of Zhang et al. 2015 | Webscope data-sharing agreement: academic research only |
| DBpedia | DBpedia ontology classes, version of Zhang et al. 2015 | CC BY-SA 3.0 and GFDL |
| Yelp (polarity) | [Yelp Open Dataset](https://www.yelp.com/dataset), version of Zhang et al. 2015 | Yelp Dataset User Agreement: academic research only, no distribution |
| MNLI | Williams et al. 2018, [site](https://cims.nyu.edu/~sbowman/multinli/) (via GLUE) | mixed: OANC terms for most genres, CC BY-SA 3.0 / CC BY 3.0 / public domain for the fiction genre |
| QQP | Quora Question Pairs (via GLUE) | no explicit licence ; Quora terms may apply |
| RTE | RTE1/2/3/5 challenges (via GLUE/SuperGLUE) | no licence found; GLUE defers to the original licences |
| CB | [CommitmentBank](https://github.com/mcdm/CommitmentBank) (via SuperGLUE) | no licence on the repository |
| WiC | [Pilehvar and Camacho-Collados](https://pilehvar.github.io/wic/) (via SuperGLUE) | CC BY-NC 4.0 |
| COPA | Roemmele et al. 2011 (via SuperGLUE) | not stated |
| BoolQ | Clark et al. 2019, [google/boolq](https://huggingface.co/datasets/google/boolq) | CC BY-SA 3.0 |
| MultiRC | Khashabi et al. 2018, [CogComp/multirc](https://github.com/CogComp/multirc) | "Research and Academic Use License" |
| SST-2 | Socher et al. 2013, [Stanford SST](https://nlp.stanford.edu/sentiment/index.html) (via GLUE) | no licence stated |
| IMDB | Maas et al. 2011, [site](https://ai.stanford.edu/~amaas/data/sentiment/) | no licence stated (citation requested) |

## TRACE (`data/TRACE`)

TRACE: *A Comprehensive Benchmark for Continual Learning in Large Language Models* (Wang et al.), repository
[BeyonderXX/TRACE](https://github.com/BeyonderXX/TRACE) (Apache-2.0 for the code; the README states nothing about the data
licences). The release bundles processed versions of the following datasets:

| task | underlying dataset | licence (as stated) |
|---|---|---|
| C-STANCE | [chenyez/C-STANCE](https://github.com/chenyez/C-STANCE) (Weibo posts) | no licence in the repository |
| FOMC | [gtfintechlab/fomc_communication](https://huggingface.co/datasets/gtfintechlab/fomc_communication) | CC BY-NC 4.0 |
| MeetingBank | [huuuyeah/meetingbank](https://huggingface.co/datasets/huuuyeah/meetingbank) | CC BY-NC-SA 4.0 |
| Py150 | ETH Py150 ([SRI Lab](https://www.sri.inf.ethz.ch/py150)); [ETH Py150 Open](https://huggingface.co/datasets/google-research-datasets/eth_py150_open) is Apache-2.0 with per-file licences | per-file licences of the source repositories |
| ScienceQA | [derek-thomas/ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) | CC BY-NC-SA 4.0 (licensing section; metadata says CC BY-SA 4.0) |
| NumGLUE-cm, NumGLUE-ds | [allenai/numglue](https://github.com/allenai/numglue) | ODC-By (the compilation); components keep their own licences |
| 20Minuten | [ZurichNLP/20Minuten](https://github.com/ZurichNLP/20Minuten) (Rios et al. 2021; Kew et al. 2023) | code Apache-2.0; no data licence stated; articles from 20 Minuten |

## Models

Model weights are not included. They are downloaded from the Hugging Face Hub under their own licences (T5-large:
Apache-2.0; Qwen3-0.6B and Qwen3.5-4B: see their model cards; Llama-2-7B: Meta's Llama 2 Community License, gated).
