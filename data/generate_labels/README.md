# Promoter sequence extraction and labeling (generate_labels)

This directory implements **promoter sequence extraction and labeling**, decoupled from the training pipeline. The output CSV is compatible with `merged_dataset.csv` and can be used by `train_dnabert.py`.

- **Implementation**: `config.py`, `io_utils.py`, `extract.py`, `annotate.py`, `run_pipeline.py`.

## Feature overview

1. Extract promoter sequences from BED + genome FASTA; extract length is configurable (50/100/200/400/600 bp).
2. Four annotation strategies: Gaussian (`gaussian`), triangle (`triangle`), impulse (`impulse`), and Poisson (`poisson`).
3. Annotation region is fixed as 20 bp upstream of TSS + TSS + 20 bp downstream of TSS (41 bp total).
4. The 41 bp window is placed at a random offset `k` within the full promoter sequence.

## Dependencies

```bash
pip install pyfaidx pandas numpy
```

If the genome FASTA has no `.fai` index yet, pyfaidx will try to build one on first open (large files may be slow). You can also run:

```bash
samtools faidx data/dataset/GCF_000001405.40_GRCh38.p14_genomic.fna
```

## How to run

From the **project root** (recommended):

```bash
python -m data.generate_labels.run_pipeline --help
python -m data.generate_labels.run_pipeline -n 100 -L 600 -s gaussian -o data/promoter_labeled.csv
```

Or run directly under `data/generate_labels`:

```bash
cd data/generate_labels
python run_pipeline.py --help
python run_pipeline.py -n 100 -L 600 -s gaussian -o ../promoter_labeled.csv
```

### Common arguments

| Argument | Description |
|------|------|
| `--bed` | BED file path (default `data/dataset/human_epdnew_Z471I.bed`) |
| `--genome` | Genome FASTA path |
| `-o`, `--output` | Output CSV path (default `data/promoter_labeled.csv`) |
| `-L`, `--length` | Extract length (if omitted, sampled from 100/200/400/600) |
| `-s`, `--strategy` | Annotation strategy: `gaussian` / `triangle` / `impulse` / `poisson` |
| `-n`, `--max-records` | Maximum records to process (use a small value for testing) |
| `--seed` | Random seed |

## Output format

CSV columns: `unique_id`, `chromosome`, `sequence`, `classification_label`, `tss_position`, `regression_label`.  
`regression_label` is a comma-separated float string matching the sequence length, compatible with existing `load_data`.

## Example commands
### Gaussian
python -m data.generate_labels.run_pipeline -n 60000 -L 50 -a 11 -s gaussian -o data/proceed_promoters/promoter_labeled_gaussian_50_11.csv

### Triangle
python -m data.generate_labels.run_pipeline -n 60000 -L 600 -a 11 -s triangle -o data/proceed_promoters/promoter_labeled_triangle_600_11.csv

### Impulse
python -m data.generate_labels.run_pipeline -n 60000 -L 200 -a 11 -s impulse -o data/proceed_promoters/promoter_labeled_impulse_200_11.csv

### Poisson
python -m data.generate_labels.run_pipeline -n 60000 -L 50 -a 11 -s poisson -o data/proceed_promoters/promoter_labeled_poisson_50_11.csv
