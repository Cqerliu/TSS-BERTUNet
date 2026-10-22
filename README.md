# ED-TSSL-DNABERT-V2.0

**TSS-BERTUNet**: pretrained representation alignment with neighborhood supervision for base-resolution TSS localization.

Official code for *TSS-BERTUNet: Pretrained Representation Alignment with Neighborhood Supervision for Base-Resolution TSS Localization*.

## Overview

Transcription start sites (TSS) are nucleotide-resolution genomic coordinates that define promoter location, transcription direction, and the 5′ end of transcripts. While genomic foundation models such as DNABERT-2 learn transferable contextual DNA representations, applying them to **DNA-sequence-only, base-resolution TSS localization** remains difficult for two reasons:

1. **Resolution mismatch** — DNABERT-2 uses variable-length BPE tokens, so token-level hidden states are not aligned with single-nucleotide TSS coordinates.
2. **Sparse supervision** — a single annotated TSS provides a highly sparse target on long sequences, which is a weak positional constraint for localization learning.

**TSS-BERTUNet** addresses both issues by combining DNABERT-2 with a 1D U-Net regression head:

- DNABERT-2 encodes the input DNA sequence into contextual token representations.
- An **Offset Mapping** module projects each token feature onto the nucleotides it covers and fuses them with base one-hot encodings, yielding coordinate-aligned nucleotide-level features (768 + 4 = 772 dims).
- A **1D U-Net** integrates local patterns and multi-scale context to produce a continuous TSS score for every base.
- Four neighborhood regression label strategies (**Gaussian**, **impulse**, **triangle**, **Poisson**) provide denser positional supervision around the annotated TSS and are compared across five input lengths (**50 / 100 / 200 / 400 / 600 bp**).

### Main contributions

- **Unified base-level TSS framework** — extends promoter-level discrimination to per-nucleotide TSS scoring with DNABERT-2 + 1D U-Net.
- **Token-to-nucleotide alignment** — Offset Mapping recovers base-aligned features from BPE token states for coordinate-consistent prediction.
- **Neighborhood label supervision** — Gaussian / impulse / triangle / Poisson soft labels mitigate single-point sparsity and stabilize localization learning.

### Data (paper setting)

Human TSS annotations from **EPDnew**; reference genome **GRCh38** (NCBI RefSeq `GCF_000001405.40`). Benchmark set: **29,598** TSS-positive and **29,598** TSS-negative sequences, split **8:1:1** into train / validation / test. TSS position within each window is randomly offset to reduce positional bias. Evaluation further includes sliding-window validation on held-out chromosomes, DNABERT-2 ablation, and single-base occlusion analysis.

## Project structure

```
train_dnabert.py          # Training entry point
model/                    # DNABERT + U-Net regression head
training/                 # Trainer, losses, metrics, callbacks
data/                     # Data loading and label-generation pipeline
scripts/                  # Sliding-window validation, inference, interpretability
config/                   # Training and validation configs
DNABERT-2-117M/           # Model config and tokenizer (weights must be downloaded separately)
results/                  # Selected validation metrics
```

## Dependencies

```bash
pip install -r requirements.txt
```

Python 3.9+ is recommended, with a PyTorch build that matches your local CUDA setup.

## Pretrained weights

This repository **does not include** `DNABERT-2-117M/pytorch_model.bin` (~447 MB). Download it from Hugging Face:

- Model: [`zhihan1996/DNABERT-2-117M`](https://huggingface.co/zhihan1996/DNABERT-2-117M)

Place the weights under `DNABERT-2-117M/`, or update `dnabert.model_path` in `config/training_config.json`.

## Data


### Not included (too large)

| Content | How to obtain |
|------|----------|
| GRCh38 genome FASTA | NCBI / original local project under `data/dataset/` |
| Full 50/100/200/400/600 bp labeled CSV | Original project, or regenerate with `generate_labels` |
| Fine-tuned checkpoints `*.pth` | Train locally |

## Quick start

```bash
# 1. Install dependencies and prepare DNABERT-2 weights
# 2. Edit data paths in config/training_config.json
#    (defaults point to 600 bp CSV; provide your own or switch to the uploaded 50/100 CSV)
# 3. Train
python train_dnabert.py --config config/training_config.json
```

Sliding-window validation example:

```bash
python scripts/sliding_window_validate.py --config config/sliding_window_validate_config.json
```

