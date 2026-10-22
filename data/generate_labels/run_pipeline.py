# -*- coding: utf-8 -*-
"""
Promoter sequence extraction and labeling — main entry
Pipeline: config → BED → extract → annotate → write CSV.
"""

import argparse
import logging
import sys
import os
import random
from typing import Dict, List, Tuple

import numpy as np

# Ensure same-directory imports work when run as a script
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
_root = os.path.dirname(os.path.dirname(_here))
if _root not in sys.path:
    sys.path.insert(0, _root)

# TSS mask windows for negative sampling (values aligned with GenerateNegative.py)
# Forward (+):  [tss - UPSTREAM_FORWARD, tss + DOWNSTREAM_FORWARD)
# Reverse (-):  [tss - UPSTREAM_REVERSE, tss + DOWNSTREAM_REVERSE)
UPSTREAM_FORWARD = 250
DOWNSTREAM_FORWARD = 50
UPSTREAM_REVERSE = 51
DOWNSTREAM_REVERSE = 249

try:
    from . import config
    from .io_utils import load_bed, write_labeled_csv, GenomeFetcher
    from .extract import extract_all
    from .annotate import make_labels
except ImportError:
    import config
    from io_utils import load_bed, write_labeled_csv, GenomeFetcher
    from extract import extract_all
    from annotate import make_labels

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge half-open intervals [s, e)."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [[intervals[0][0], intervals[0][1]]]
    for s, e in intervals[1:]:
        if s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(s, e) for s, e in merged]


def _build_allowed_segments_for_chr(
    disallowed_intervals: List[Tuple[int, int]],
    chrom_len: int,
) -> List[Tuple[int, int]]:
    """
    Return half-open intervals allowed for sampling (all disallowed intervals subtracted from [0, chrom_len)).
    """
    # Clip to valid range and drop empty intervals
    clipped: List[Tuple[int, int]] = []
    for s, e in disallowed_intervals:
        if e <= 0 or s >= chrom_len:
            continue
        s2 = max(0, s)
        e2 = min(chrom_len, e)
        if s2 < e2:
            clipped.append((s2, e2))

    merged_disallowed = _merge_intervals(clipped)

    allowed: List[Tuple[int, int]] = []
    cur = 0
    for s, e in merged_disallowed:
        if s > cur:
            allowed.append((cur, s))
        cur = max(cur, e)
        if cur >= chrom_len:
            break
    if cur < chrom_len:
        allowed.append((cur, chrom_len))
    return allowed


def _sample_negative_sequence_from_allowed(
    fetcher,
    contig: str,
    allowed_segments: List[Tuple[int, int]],
    seq_len: int,
    rng: random.Random,
    max_tries: int = 5000,
) -> str:
    """
    Sample a negative sequence of length seq_len from allowed intervals:
    1) Sampled interval does not overlap TSS mask windows (guaranteed by allowed_segments)
    2) Sampled sequence contains no N (same filter as positives)
    """
    candidates = [(s, e) for (s, e) in allowed_segments if (e - s) >= seq_len]
    if not candidates:
        raise ValueError(f"No segment with length >= {seq_len} in allowed_segments (contig={contig})")

    for _ in range(max_tries):
        seg_start, seg_end = rng.choice(candidates)
        start = rng.randint(seg_start, seg_end - seq_len)
        end = start + seq_len
        seq = fetcher.fetch(contig, start, end)
        if "N" in seq:
            continue
        return seq

    raise ValueError(f"Failed to sample an N-free negative after many tries (contig={contig}, seq_len={seq_len})")


def run(
    bed_path: str = None,
    genome_path: str = None,
    output_path: str = None,
    extract_length: int = None,
    length_list: list = None,
    strategy: str = "gaussian",
    max_records: int = None,
    seed: int = None,
    annotation_length: int = None,
) -> int:
    """
    Run the full pipeline: extract → annotate → write CSV.
    strategy: "gaussian" | "triangle" | "impulse" | "poisson"
    annotation_length: label annotation window length (bp), e.g. 41 for TSS ± upstream/downstream totaling 41 bp;
        defaults to config 41 if unspecified.
    Returns the number of records written successfully (including negatives).
    """
    bed_path = bed_path or config.BED_PATH
    genome_path = genome_path or config.GENOME_FASTA_PATH
    output_path = output_path or config.OUTPUT_CSV_PATH
    seed = seed if seed is not None else config.RANDOM_SEED

    if not os.path.isfile(bed_path):
        logger.error("BED file not found: %s", bed_path)
        return 0
    if not os.path.isfile(genome_path):
        logger.error("Genome FASTA not found: %s", genome_path)
        return 0

    annotation_len = annotation_length if annotation_length is not None else config.ANNOTATION_LEN
    logger.info("Extracting promoters from BED: %s (annotation window %d bp)", bed_path, annotation_len)
    records = extract_all(
        bed_path=bed_path,
        genome_path=genome_path,
        extract_length=extract_length,
        length_list=length_list or config.EXTRACT_LENGTHS,
        max_records=max_records,
        seed=seed,
        annotation_length=annotation_length,
    )
    logger.info("Successfully extracted %d sequences", len(records))

    if not records:
        logger.warning("No valid records; skip writing")
        return 0

    # Drop sequences containing ambiguous bases (e.g. 'N')
    filtered_records = []
    dropped = 0
    for rec in records:
        seq = str(rec.get("sequence", "")).upper()
        if "N" in seq:
            dropped += 1
            continue
        filtered_records.append(rec)

    if dropped > 0:
        logger.info("Dropped sequences containing ambiguous base 'N': %d", dropped)

    records = filtered_records
    if not records:
        logger.warning("No valid sequences after filtering; skip writing")
        return 0

    # Build labels per record (compatible with merged_dataset)
    for rec in records:
        L = rec["L"]
        a_start = rec["annotation_start"]
        a_end = rec["annotation_end"]
        tss_in_seq = rec["tss_in_seq"]
        rec["regression_label"] = make_labels(L, a_start, a_end, tss_in_seq, strategy, annotation_length=annotation_length)
        rec["classification_label"] = 1
        rec["tss_position"] = tss_in_seq

    # Also generate negatives: same count as positives, each matching the corresponding positive length
    disallowed_by_chr: Dict[str, List[Tuple[int, int]]] = {}
    bed_df = load_bed(bed_path)
    for _, row in bed_df.iterrows():
        chrom = str(row["chr"]).strip()
        tss = int(row["start"])
        strand = str(row["strand"]).strip().lower()
        is_forward = strand in ("forward", "+", "1")
        if is_forward:
            s = tss - UPSTREAM_FORWARD
            e = tss + DOWNSTREAM_FORWARD
        else:
            s = tss - UPSTREAM_REVERSE
            e = tss + DOWNSTREAM_REVERSE
        disallowed_by_chr.setdefault(chrom, []).append((s, e))

    rng = random.Random(seed)

    # Precompute allowed sampling segments per chromosome needed
    # Note: only interval subtraction here — avoid loading the whole genome into memory like GenerateNegative.py.
    try:
        from pyfaidx import Fasta
    except ImportError:
        raise ImportError(
            "Please install pyfaidx: pip install pyfaidx. Negative sampling needs contig lengths from the FASTA."
        )

    unique_pos_chroms = sorted({str(r.get("chromosome", "")).strip() for r in records if r.get("chromosome")})
    allowed_segments_by_chr: Dict[str, List[Tuple[int, int]]] = {}
    contig_by_chr: Dict[str, str] = {}

    fa = None
    try:
        fa = Fasta(genome_path, one_based_attributes=False)
        for chr_name in unique_pos_chroms:
            contig = config.get_contig(chr_name)
            contig_by_chr[chr_name] = contig
            contig_len = len(fa[contig])
            disallowed = disallowed_by_chr.get(chr_name, [])
            allowed_segments_by_chr[chr_name] = _build_allowed_segments_for_chr(disallowed, contig_len)
    finally:
        if fa is not None:
            try:
                fa.close()
            except Exception:
                pass

    negatives: List[dict] = []
    with GenomeFetcher(genome_path) as fetcher:
        # Cache candidate segments to avoid re-filtering every time
        candidates_cache: Dict[Tuple[str, int], List[Tuple[int, int]]] = {}

        for idx, pos_rec in enumerate(records):
            L = int(pos_rec["L"])
            preferred_chr = str(pos_rec["chromosome"]).strip()

            # Prefer the same chromosome as the positive; fall back to others if too short
            target_chr_list = [preferred_chr] + [c for c in allowed_segments_by_chr.keys() if c != preferred_chr]
            chosen_seq = None
            chosen_chr = None

            for chr_name in target_chr_list:
                allowed_segments = allowed_segments_by_chr.get(chr_name, [])
                if not allowed_segments:
                    continue
                cache_key = (chr_name, L)
                if cache_key not in candidates_cache:
                    candidates_cache[cache_key] = [(s, e) for (s, e) in allowed_segments if (e - s) >= L]
                candidates = candidates_cache[cache_key]
                if not candidates:
                    continue

                contig = contig_by_chr.get(chr_name)
                if not contig:
                    continue

                # Resample until N-free via _sample_negative_sequence_from_allowed
                chosen_seq = _sample_negative_sequence_from_allowed(
                    fetcher=fetcher,
                    contig=contig,
                    allowed_segments=allowed_segments,
                    seq_len=L,
                    rng=rng,
                )
                chosen_chr = chr_name
                break

            if chosen_seq is None:
                raise ValueError(
                    f"Failed to generate a negative of length={L} for a positive "
                    f"(positive unique_id={pos_rec.get('unique_id')})"
                )

            negatives.append(
                {
                    "unique_id": f"neg_{idx}",
                    "chromosome": chosen_chr or preferred_chr,
                    "sequence": chosen_seq,
                    "classification_label": 0,
                    # tss_position is unused by current training code; placeholder integer
                    "tss_position": 0,
                    # All-zero regression labels so loss_config threshold can separate pos/neg
                    "regression_label": np.zeros(L, dtype=np.float32),
                }
            )

    all_records = records + negatives
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    write_labeled_csv(all_records, output_path)
    logger.info(
        "Wrote CSV: %s (strategy=%s, annotation window %d bp; positives=%d, negatives=%d)",
        output_path,
        strategy,
        annotation_len,
        len(records),
        len(negatives),
    )
    return len(all_records)


def main():
    parser = argparse.ArgumentParser(description="Promoter sequence extraction and labeling")
    parser.add_argument("--bed", default=None, help="BED file path")
    parser.add_argument("--genome", default=None, help="Genome FASTA path")
    parser.add_argument("--output", "-o", default=None, help="Output CSV path")
    parser.add_argument("--length", "-L", type=int, default=None,
                        help="Extract length (if omitted, sampled from 100/200/400/600)")
    parser.add_argument(
        "--strategy",
        "-s",
        choices=["gaussian", "triangle", "impulse", "poisson"],
        default="gaussian",
        help="Annotation strategy: gaussian=Gaussian, triangle=triangle, impulse=impulse, poisson=Poisson tail",
    )
    parser.add_argument("--max-records", "-n", type=int, default=None, help="Max records to process (for testing)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--annotation-length", "-a", type=int, default=None,
                        help="Label annotation window length (bp), e.g. 41 for TSS ± totaling 41 bp; default 41")
    args = parser.parse_args()

    run(
        bed_path=args.bed,
        genome_path=args.genome,
        output_path=args.output,
        extract_length=args.length,
        strategy=args.strategy,
        max_records=args.max_records,
        seed=args.seed,
        annotation_length=args.annotation_length,
    )


if __name__ == "__main__":
    main()
