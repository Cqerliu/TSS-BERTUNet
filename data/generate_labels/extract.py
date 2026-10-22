# -*- coding: utf-8 -*-
"""
Promoter sequence extraction and labeling — extraction logic
For each BED record: sample L and k, compute the genomic interval, fetch the sequence,
handle strand, and return the sequence with metadata.
"""

import random
import logging
from typing import List, Optional, Tuple, Any

import pandas as pd
import numpy as np

from . import config
from .io_utils import GenomeFetcher, load_bed, reverse_complement

logger = logging.getLogger(__name__)


def _sample_k(L: int, rng: random.Random, annotation_len: int = None) -> int:
    """Uniformly sample integer k in [0, L-annotation_len]."""
    ann_len = annotation_len if annotation_len is not None else config.ANNOTATION_LEN
    if L < ann_len:
        return 0
    return rng.randint(0, L - ann_len)


def _genome_interval(tss: int, k: int, L: int, tss_upstream: int = None) -> Tuple[int, int]:
    """Compute the extract interval [genome_start, genome_end), 0-based."""
    up = tss_upstream if tss_upstream is not None else config.TSS_UPSTREAM
    genome_start = tss - up - k
    genome_end = genome_start + L
    return genome_start, genome_end


def _annotation_params(annotation_length: Optional[int] = None) -> Tuple[int, int]:
    """From total annotation window length, return (annotation_len, tss_upstream). TSS is centered."""
    if annotation_length is not None:
        ann_len = annotation_length
        tss_upstream = (ann_len - 1) // 2
        return ann_len, tss_upstream
    return config.ANNOTATION_LEN, config.TSS_UPSTREAM


def extract_one(
    row: pd.Series,
    fetcher: GenomeFetcher,
    L: int,
    k: Optional[int] = None,
    rng: Optional[random.Random] = None,
    annotation_length: Optional[int] = None,
) -> Optional[dict]:
    """
    Extract a promoter sequence of length L for one BED record.
    The annotation window starts at index k in the sequence (sampled at random if not provided).

    annotation_length: total annotation window length (bp); defaults to config 41. TSS is at the window center.

    Returns
    -------
    dict with: sequence, chr, strand, name, L, k, tss_in_seq, annotation_start, annotation_end,
             genome_start, genome_end, unique_id
    Returns None if out of bounds or sequence fetch fails.
    """
    if rng is None:
        rng = random.Random(config.RANDOM_SEED)
    ann_len, tss_upstream = _annotation_params(annotation_length)
    chr_name = str(row["chr"]).strip()
    tss = int(row["start"])
    strand = str(row["strand"]).strip() in ("+", "1") and "+" or "-"
    name = str(row.get("name", ""))

    if k is None:
        k = _sample_k(L, rng, ann_len)
    genome_start, genome_end = _genome_interval(tss, k, L, tss_upstream)
    if genome_start < 0:
        logger.debug("Skip %s:%d start out of bounds genome_start=%d", chr_name, tss, genome_start)
        return None

    contig = config.get_contig(chr_name)
    try:
        seq_raw = fetcher.fetch(contig, genome_start, genome_end)
    except Exception as e:
        # Interval may exceed contig length
        logger.debug("Fetch failed %s %d-%d: %s", contig, genome_start, genome_end, e)
        return None
    if len(seq_raw) != L:
        logger.debug("Length mismatch %s expected %d got %d", name, L, len(seq_raw))
        return None

    if strand == "-":
        sequence = reverse_complement(seq_raw)
        # After reverse complement, annotation window is [L-ann_len-k, L-k); TSS at L-tss_upstream-k
        annotation_start = L - ann_len - k
        annotation_end = L - k
        tss_in_seq = L - tss_upstream - k
    else:
        sequence = seq_raw
        annotation_start = k
        annotation_end = k + ann_len
        tss_in_seq = k + tss_upstream + 1

    return {
        "sequence": sequence,
        "chromosome": chr_name,
        "strand": strand,
        "name": name,
        "L": L,
        "k": k,
        "tss_in_seq": tss_in_seq,
        "annotation_start": annotation_start,
        "annotation_end": annotation_end,
        "genome_start": genome_start,
        "genome_end": genome_end,
        "unique_id": name or f"{chr_name}_{tss}_{strand}",
    }


def extract_all(
    bed_path: str,
    genome_path: str,
    extract_length: Optional[int] = None,
    length_list: Optional[List[int]] = None,
    max_records: Optional[int] = None,
    seed: int = config.RANDOM_SEED,
    annotation_length: Optional[int] = None,
) -> List[dict]:
    """
    Load BED and extract one sequence per record; each record uses a fixed length or a random L from length_list, then a random k.
    annotation_length: total annotation window length (bp); defaults to config 41.
    Returns the list of successfully extracted records (each a dict from extract_one).
    """
    df = load_bed(bed_path)
    length_list = length_list or config.EXTRACT_LENGTHS
    rng = random.Random(seed)
    results = []
    with GenomeFetcher(genome_path) as fetcher:
        for idx, row in df.iterrows():
            if max_records is not None and len(results) >= max_records:
                break
            L = extract_length if extract_length is not None else rng.choice(length_list)
            rec = extract_one(row, fetcher, L, k=None, rng=rng, annotation_length=annotation_length)
            if rec is not None:
                results.append(rec)
    return results
