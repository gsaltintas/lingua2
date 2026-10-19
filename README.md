# MoTo: Mixture of Tokenizers - Training Code Base (Lingua)

**Gül Sena Altıntaş**, **Colin Raffel**  
University of Toronto · Vector Institute

[![Paper](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/XXXX.XXXXX)
[![Models 🤗](https://img.shields.io/badge/Models-%F0%9F%A4%97%20HuggingFace-yellow)](https://huggingface.co/flexitok)
[![Tokenizers 🤗](https://img.shields.io/badge/Tokenizers-%F0%9F%A4%97-yellow)](flexitok/tokenizer_training/)
[![Experimental Code ⚙️](https://img.shields.io/badge/Training-blue)](https://github.com/r-three/moto)

This repository contains the training and evaluation code for the MoTo paper. It is built on top of [Meta Lingua](https://github.com/facebookresearch/lingua) — see [README.md](README.md) for framework setup, SLURM job launching, debugging, and checkpoint management.

---

## Setup

```bash
git clone https://github.com/r-three/moto
cd moto
bash setup/create_env.sh   # or: sbatch setup/create_env.sh on SLURM
conda activate lingua_<date>
```

---

## Tokenizers

All 21 per-language BPE tokenizers are released on HuggingFace under the [`flexitok`](https://huggingface.co/flexitok) namespace (e.g. `flexitok/bpe_tur_Latn_8000`). Tokenizer training scripts live in `flexitok/tokenizer_training/`.

```python
from transformers import PreTrainedTokenizerFast
tok = PreTrainedTokenizerFast.from_pretrained("flexitok/bpe_tur_Latn_8000")
tok.encode("Türkiye'de güzel bir gün.")
```

The superset vocabulary construction and language-ID-based routing are implemented in [`lingua/data.py`](lingua/data.py).

---

## Model Training

All configs are under `apps/main/configs/flexitok/`. Every run composes a base config with a model-specific config:

```bash
python -m lingua.stool \
  script=apps.main.train \
  config=["apps/main/configs/flexitok/llama_1b_base.yaml",
          "apps/main/configs/flexitok/fineweb_2_hq/<config>.yaml"] \
  nodes=1 partition=<partition>
```

| Config (`fineweb_2_hq/`) | Description | Superset size |
|---|---|---|
| `superset_lang_tokenizers_8k.yaml` | **MoTo** — script-aware pre-tok, 8K/lang | 119,716 |
| *(ablations/)* `2.10_bpe_on_all_langs.yaml` | MoTo-GPT4 — GPT-4 pre-tok, 8K/lang | 113,764 |
| *(ablations/)* `8k_v3.yaml` | MoTo-16K — 16K/lang | 245,153 |
| `superset_lang_tokenizers_8k.yaml` (script variant) | MoTo-Script — per script family | 165,022 |
| `../superset_albert.yaml` | Superset-Albert — 14 existing tokenizers, ALBERT factorization | 851,586 |

**Baselines** (`apps/main/configs/flexitok/`):

| Config | Description |
|---|---|
| `llama_1b_base.yaml` | Llama-3.2 tokenizer |
| `llama_bpe_dropout.yaml` | Llama + BPE-Dropout |
| `other_baselines/` | mBPE-GPT4, mBPE-Boundless, ByT5 |

---

## Evaluation

```bash
python -m lingua.stool \
  script=apps.main.eval \
  config=apps/main/configs/flexitok/fineweb_2_hq/eval_superset_fineweb_2_hq.yaml \
  nodes=1 partition=<partition>
```

Evaluates on: HellaSwag/PIQA/BoolQ/ARC/CommonsenseQA (EN), Okapi, XCOPA, XStoryCloze, XNLI, PAWS-X, Wino-X, TokSuite, and FLORES-200 bits-per-byte across 21 languages. Eval configs are in `apps/main/configs/flexitok/eval/` and `fineweb_2_hq/`.

---

## Multilingual Addition

Controlled synthetic task: number words → sum in 21 languages. Trains a small (~3M param) model.

```bash
python -m lingua.stool \
  script=apps.main.train \
  config=apps/main/configs/flexitok/multilingual-addition/<config>.yaml \
  nodes=1 partition=<partition>
```

---

## Citation

```bibtex
@inproceedings{
altintas2026moto,
title={{MoTo}: Mixture of Tokenizers Towards Fair Multilingual Language Modeling},
author = {Alt{\i}nta\c{s}, G{\"u}l Sena and Raffel, Colin},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url    = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

---

*Built on [Meta Lingua](https://github.com/facebookresearch/lingua) (BSD-3-Clause). See [LICENSE](LICENSE)*, see [README_lingua](README_lingua.md) for lingua instructions. 
