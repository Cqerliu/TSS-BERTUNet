# -*- coding: utf-8 -*-
"""
Promoter extraction and labeling — configuration
Paths, extract-length list, random seed, chr→contig mapping, etc.
"""

import os

# Project root (two levels above data/generate_labels)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATASET = os.path.join(_ROOT, "data", "dataset")

# ---------- Paths ----------
BED_PATH = os.path.join(_DATASET, "human_epdnew_Z471I.bed")
GENOME_FASTA_PATH = os.path.join(_DATASET, "GCF_000001405.40_GRCh38.p14_genomic.fna")
# Output CSV, compatible with merged_dataset.csv; written under data/ by default
OUTPUT_CSV_PATH = os.path.join(_ROOT, "data", "promoter_labeled.csv")

# ---------- Extract lengths ----------
# Customizable; supports 100 / 200 / 400 / 600 bp; a run may use a fixed length or sample from the list
EXTRACT_LENGTHS = [100, 200, 400, 600]

# ---------- Annotation window (fixed 41 bp) ----------
# 20 bp upstream of TSS + TSS + 20 bp downstream of TSS
TSS_UPSTREAM = 20
TSS_DOWNSTREAM = 20
ANNOTATION_LEN = TSS_UPSTREAM + 1 + TSS_DOWNSTREAM  # 41

# ---------- Random seed ----------
RANDOM_SEED = 42

# ---------- chr → RefSeq contig map (GRCh38.p14) ----------
# BED uses chr1; FASTA uses NC_000001.11, etc.
CHR_TO_CONTIG = {
    "chr1": "NC_000001.11",
    "chr2": "NC_000002.12",
    "chr3": "NC_000003.12",
    "chr4": "NC_000004.12",
    "chr5": "NC_000005.10",
    "chr6": "NC_000006.12",
    "chr7": "NC_000007.14",
    "chr8": "NC_000008.11",
    "chr9": "NC_000009.12",
    "chr10": "NC_000010.11",
    "chr11": "NC_000011.10",
    "chr12": "NC_000012.12",
    "chr13": "NC_000013.11",
    "chr14": "NC_000014.9",
    "chr15": "NC_000015.10",
    "chr16": "NC_000016.10",
    "chr17": "NC_000017.11",
    "chr18": "NC_000018.10",
    "chr19": "NC_000019.10",
    "chr20": "NC_000020.11",
    "chr21": "NC_000021.9",
    "chr22": "NC_000022.11",
    "chrX": "NC_000023.11",
    "chrY": "NC_000024.10",
    "chrM": "NC_012920.1",
    "chrMT": "NC_012920.1",
}


def get_contig(chr_name: str) -> str:
    """Map BED chromosome name to FASTA contig; return unchanged if unmapped (FASTA may already use chr)."""
    return CHR_TO_CONTIG.get(chr_name, chr_name)
