# -*- coding: utf-8 -*-
"""
Promoter extraction and labeling — I/O utilities
BED parsing, genomic interval fetch (pyfaidx), reverse complement, CSV writing.
"""

import os
import logging
from typing import List, Tuple, Optional, Any

import pandas as pd
import numpy as np

from . import config

logger = logging.getLogger(__name__)

# Reverse complement
_COMPL = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a sequence, preserving case."""
    return seq.translate(_COMPL)[::-1]


def load_bed(bed_path: str) -> pd.DataFrame:
    """
    Load a BED file.
    Columns: chr, start, end, name, score, strand
    TSS is the 0-based start.
    """
    df = pd.read_csv(
        bed_path,
        sep="\t",
        header=None,
        names=["chr", "start", "end", "name", "score", "strand"],
        dtype={"chr": str, "start": int, "end": int, "name": str, "score": str, "strand": str},
    )
    return df


class GenomeFetcher:
    """Fetch sequence by contig and interval from FASTA (via pyfaidx). Uses Fasta 0-based half-open slicing."""

    def __init__(self, fasta_path: str):
        self.fasta_path = fasta_path
        self._fa = None

    def __enter__(self):
        try:
            from pyfaidx import Fasta
            self._fa = Fasta(self.fasta_path, one_based_attributes=False)
        except ImportError:
            raise ImportError(
                "Please install pyfaidx: pip install pyfaidx. "
                "If the genome FASTA has no .fai, an index will be built automatically."
            )
        return self

    def __exit__(self, *args):
        if self._fa is not None:
            try:
                self._fa.close()
            except Exception:
                pass
            self._fa = None

    def fetch(self, contig: str, start: int, end: int) -> str:
        """Fetch the [start, end) 0-based interval as an uppercase string."""
        seq = self._fa[contig][start:end]
        s = str(seq).replace("\n", "").replace(" ", "").upper()
        return s


def ensure_fai(fasta_path: str) -> None:
    """If .fai is missing, try to build an index with faidx or samtools."""
    fai_path = fasta_path + ".fai"
    if os.path.isfile(fai_path):
        return
    try:
        import subprocess
        subprocess.run(["faidx", fasta_path], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        try:
            subprocess.run(["samtools", "faidx", fasta_path], check=True, capture_output=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.warning(
                "faidx/samtools not found; pyfaidx will build the index automatically (may be slow)."
            )


def write_labeled_csv(
    rows: List[dict],
    out_path: str,
    regression_column: str = "regression_label",
) -> None:
    """
    Write a CSV compatible with merged_dataset.csv.
    Each row needs: unique_id, chromosome, sequence, classification_label, tss_position, regression_label.
    regression_label is a comma-separated float string of the same length as sequence.
    """
    for r in rows:
        lab = r.get(regression_column)
        if isinstance(lab, np.ndarray):
            r["regression_label"] = ",".join(f"{x:.6f}" for x in lab.tolist())
        elif isinstance(lab, (list, tuple)):
            r["regression_label"] = ",".join(f"{float(x):.6f}" for x in lab)
        else:
            r["regression_label"] = str(lab or "")
    df = pd.DataFrame(rows)
    cols = ["unique_id", "chromosome", "sequence", "classification_label", "tss_position", "regression_label"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df[cols].to_csv(out_path, index=False, encoding="utf-8")
