#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSS base-level contribution analysis (biological interpretability)

For all test-set samples from the training split, use per-site occlusion:
replace one base with N and measure the drop in predicted probability at the TSS site
as that position's contribution to the TSS.

Positives: reference = true TSS; contribution = P(TSS|original) - P(TSS|masked)
Negatives: reference = predicted peak; contribution = P(peak|original) - P(peak|masked)

Outputs:
  - Mean contribution curve relative to TSS
  - Positive/negative group curves
  - Several single-sample example plots
  - NumPy arrays and summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model import DNABERTForTSSLocalization  # noqa: E402
from scripts.infer_trained_model import _build_model, _resolve_abs_path  # noqa: E402
from utils import load_training_config, set_seed  # noqa: E402

# Known core promoter elements (relative to TSS, bp)
ELEMENT_REGIONS = {
    "TATA box": (-34, -27),
    "Inr": (-2, 4),
    "DPE": (28, 32),
}

ELEMENT_COLORS = {
    "TATA box": "#f59e0b",  # amber
    "Inr": "#10b981",       # emerald
    "DPE": "#8b5cf6",       # violet
}


def _resolve_checkpoint_path(checkpoint_arg: Optional[str], merged_csv_path: str) -> str:
    """Resolve checkpoint path; compatible with training best_*_unet.pth naming."""
    if checkpoint_arg:
        path = _resolve_abs_path(checkpoint_arg)
        if os.path.isfile(path):
            return path
        raise SystemExit(f"checkpoint not found: {path}")

    csv_stem = os.path.splitext(os.path.basename(merged_csv_path))[0]
    if csv_stem.startswith("promoter_labeled_"):
        ckpt_id = csv_stem[len("promoter_labeled_") :]
    else:
        ckpt_id = csv_stem
    ckpt_id = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in ckpt_id)

    candidates = [
        os.path.join(PROJECT_ROOT, "checkpoints", f"best_{ckpt_id}_unet.pth"),
        os.path.join(PROJECT_ROOT, "checkpoints", f"best_{ckpt_id}.pth"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise SystemExit(f"checkpoint not found; tried: {candidates}")


def load_records(
    merged_csv_path: str,
    use_negative_samples: bool,
    max_seq_length: Optional[int],
    allowed_chroms: set,
) -> List[Dict[str, Any]]:
    """Load sample records consistent with load_data (including tss_position and other metadata)."""
    df = pd.read_csv(merged_csv_path)
    if not use_negative_samples:
        df = df[df["classification_label"] == 1]

    records: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        chrom = str(row.get("chromosome", ""))
        if chrom not in allowed_chroms:
            continue

        sequence = str(row["sequence"]).upper()
        if not sequence:
            continue

        if max_seq_length is not None and len(sequence) > max_seq_length:
            sequence = sequence[:max_seq_length]
            seq_len = max_seq_length
        else:
            seq_len = len(sequence)

        regression_label_str = str(row.get("regression_label", ""))
        if regression_label_str and regression_label_str != "nan":
            if "," in regression_label_str:
                label_values = [float(x.strip()) for x in regression_label_str.split(",")]
            else:
                label_values = [float(regression_label_str)]
            if len(label_values) != seq_len:
                if len(label_values) > seq_len:
                    label_values = label_values[:seq_len]
                else:
                    label_values.extend([0.0] * (seq_len - len(label_values)))
            label_array = np.array(label_values, dtype=np.float32)
        else:
            label_array = np.zeros(seq_len, dtype=np.float32)

        records.append(
            {
                "unique_id": str(row.get("unique_id", idx)),
                "chromosome": chrom,
                "sequence": sequence,
                "label": label_array,
                "seq_len": seq_len,
                "tss_position": int(row.get("tss_position", 0)),
                "classification_label": int(row.get("classification_label", 0)),
            }
        )

    if not records:
        raise ValueError("No valid samples")
    return records


def split_test_records(
    records: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> List[Dict[str, Any]]:
    """Use the same random seed and split ratios as split_data."""
    set_seed(random_seed)
    n = len(records)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    indices = np.random.permutation(n)
    test_indices = indices[val_end:]
    return [records[i] for i in test_indices]


def mask_sequence(sequence: str, position: int, mask_char: str = "N") -> str:
    """0-based single-base occlusion."""
    chars = list(sequence)
    if 0 <= position < len(chars):
        chars[position] = mask_char
    return "".join(chars)


def _collate_sequences(
    sequences: List[str],
    tokenizer,
    max_length: int,
) -> Dict[str, Any]:
    """Batch sequences in the same format as TSSDataset collate_fn."""
    batch_items = []
    for sequence in sequences:
        encoded = tokenizer(
            sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        label_len = len(sequence)
        batch_items.append(
            {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "offset_mapping": encoded["offset_mapping"].squeeze(0),
                "original_seq_length": label_len,
                "original_sequence": sequence,
            }
        )

    max_token_len = max(item["input_ids"].shape[0] for item in batch_items)
    max_base_len = max(item["original_seq_length"] for item in batch_items)
    pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") else 0

    padded_input_ids = []
    padded_attention_masks = []
    padded_offset_mappings = []
    original_seq_lengths = []
    original_sequences = []

    for item in batch_items:
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        offset_mapping = item["offset_mapping"]
        pad_token_length = max_token_len - input_ids.shape[0]

        if pad_token_length > 0:
            padded_input_id = torch.cat(
                [
                    input_ids,
                    torch.full((pad_token_length,), pad_token_id, dtype=input_ids.dtype),
                ]
            )
            padded_mask = torch.cat(
                [attention_mask, torch.zeros(pad_token_length, dtype=attention_mask.dtype)]
            )
            padded_offset = torch.cat(
                [offset_mapping, torch.zeros(pad_token_length, 2, dtype=offset_mapping.dtype)],
                dim=0,
            )
        else:
            padded_input_id = input_ids
            padded_mask = attention_mask
            padded_offset = offset_mapping

        padded_input_ids.append(padded_input_id)
        padded_attention_masks.append(padded_mask)
        padded_offset_mappings.append(padded_offset)
        original_seq_lengths.append(item["original_seq_length"])
        original_sequences.append(item["original_sequence"])

    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_masks),
        "offset_mapping": padded_offset_mappings,
        "original_seq_length": original_seq_lengths,
        "original_sequence": original_sequences,
        "max_base_len": max_base_len,
    }


@torch.inference_mode()
def predict_batch(
    model: torch.nn.Module,
    tokenizer,
    sequences: List[str],
    device: str,
    max_length: int,
) -> np.ndarray:
    """Return (B, L) base-level prediction probabilities."""
    if not sequences:
        return np.empty((0, 0), dtype=np.float32)

    batch = _collate_sequences(sequences, tokenizer, max_length)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    offset_mapping = [om.to(device) for om in batch["offset_mapping"]]

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        offset_mapping=offset_mapping,
        original_seq_length=batch["original_seq_length"],
        original_sequence=batch["original_sequence"],
    )
    probs = outputs[0] if isinstance(outputs, tuple) else outputs
    probs = probs.squeeze(1).detach().cpu().numpy()

    max_base_len = batch["max_base_len"]
    if probs.shape[1] > max_base_len:
        probs = probs[:, :max_base_len]
    return probs.astype(np.float32)


def get_reference_index(
    record: Dict[str, Any],
    baseline_curve: np.ndarray,
) -> Tuple[int, str]:
    """
    Return the alignment reference site (0-based) and its type.
    Positives: true TSS; negatives: predicted peak.
    """
    if record["classification_label"] == 1 and record["tss_position"] > 0:
        idx = int(record["tss_position"]) - 1
        idx = min(max(idx, 0), record["seq_len"] - 1)
        return idx, "true_tss"
    peak_idx = int(np.argmax(baseline_curve[: record["seq_len"]]))
    return peak_idx, "pred_peak"


def compute_baseline_scores(
    baseline_curves: np.ndarray,
    records: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    """Compute baseline probability at the reference site and the reference index for each sample."""
    n = len(records)
    baseline_ref = np.zeros(n, dtype=np.float32)
    baseline_peak = np.zeros(n, dtype=np.float32)
    ref_indices: List[int] = []
    ref_types: List[str] = []

    for i, rec in enumerate(records):
        curve = baseline_curves[i, : rec["seq_len"]]
        ref_idx, ref_type = get_reference_index(rec, curve)
        ref_indices.append(ref_idx)
        ref_types.append(ref_type)
        baseline_ref[i] = curve[ref_idx]
        baseline_peak[i] = float(curve.max())

    return baseline_ref, baseline_peak, ref_indices, ref_types


def aggregate_relative_contribution(
    all_contrib: np.ndarray,
    records: List[Dict[str, Any]],
    ref_indices: List[int],
    rel_min: int,
    rel_max: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align absolute-position contributions to TSS/peak-relative coordinates, then average.
    Returns relative_positions, mean_contrib, std_contrib
    """
    rel_positions = np.arange(rel_min, rel_max + 1, dtype=np.int32)
    n_rel = len(rel_positions)
    sums = np.zeros(n_rel, dtype=np.float64)
    sums_sq = np.zeros(n_rel, dtype=np.float64)
    counts = np.zeros(n_rel, dtype=np.int64)

    offset = -rel_min
    for i, rec in enumerate(records):
        seq_len = rec["seq_len"]
        ref_idx = ref_indices[i]
        for pos in range(seq_len):
            rel = pos - ref_idx
            if rel < rel_min or rel > rel_max:
                continue
            j = rel + offset
            val = float(all_contrib[i, pos])
            sums[j] += val
            sums_sq[j] += val * val
            counts[j] += 1

    mean = np.zeros(n_rel, dtype=np.float32)
    std = np.zeros(n_rel, dtype=np.float32)
    valid = counts > 0
    mean[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    var = np.zeros(n_rel, dtype=np.float64)
    var[valid] = sums_sq[valid] / counts[valid] - (sums[valid] / counts[valid]) ** 2
    var = np.clip(var, 0.0, None)
    std[valid] = np.sqrt(var[valid]).astype(np.float32)
    return rel_positions, mean, std


def plot_relative_curve(
    rel_positions: np.ndarray,
    mean_contrib: np.ndarray,
    std_contrib: np.ndarray,
    output_path: str,
    title: str,
    n_samples: int,
) -> None:
    """Plot mean contribution curve relative to TSS."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rel_positions, mean_contrib, color="#2563eb", linewidth=1.8, label="Mean contribution")
    ax.fill_between(
        rel_positions,
        mean_contrib - std_contrib,
        mean_contrib + std_contrib,
        color="#2563eb",
        alpha=0.2,
        label="±1 std",
    )
    ax.axvline(0, color="#dc2626", linestyle="--", linewidth=1.2, label="TSS")

    for name, (start, end) in ELEMENT_REGIONS.items():
        ax.axvspan(
            start,
            end,
            color=ELEMENT_COLORS.get(name, "#9ca3af"),
            alpha=0.18,
            label=f"{name} ({start}~{end} bp)",
        )

    ax.set_xlabel("Relative position to TSS (bp)")
    ax.set_ylabel("Contribution ΔP(TSS)\n[baseline probability − masked probability]")
    ax.set_title(f"{title}\n(n={n_samples})")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_example_sample(
    record: Dict[str, Any],
    baseline_curve: np.ndarray,
    contrib: np.ndarray,
    ref_idx: int,
    output_path: str,
) -> None:
    """Single-sample view: prediction curve + contribution curve."""
    seq_len = record["seq_len"]
    rel_x = np.arange(-ref_idx, seq_len - ref_idx)
    contrib_valid = contrib[:seq_len]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

    abs_x = np.arange(seq_len)
    axes[0].plot(abs_x, baseline_curve[:seq_len], color="#059669", linewidth=1.5, label="Predicted probability")
    if record["classification_label"] == 1 and record["tss_position"] > 0:
        tss_idx = int(record["tss_position"]) - 1
        axes[0].axvline(tss_idx, color="#dc2626", linestyle="--", label=f"True TSS ({record['tss_position']})")
    axes[0].set_ylabel("TSS probability")
    axes[0].set_title(f"Sample {record['unique_id']} - prediction curve")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(rel_x, contrib_valid, width=1.0, color="#2563eb", alpha=0.75)
    axes[1].axvline(0, color="#dc2626", linestyle="--", linewidth=1.2)
    for name, (start, end) in ELEMENT_REGIONS.items():
        axes[1].axvspan(
            start,
            end,
            color=ELEMENT_COLORS.get(name, "#9ca3af"),
            alpha=0.18,
        )
    axes[1].set_xlabel("Relative position to TSS (bp)")
    axes[1].set_ylabel("Contribution")
    axes[1].set_title("Single-base occlusion contribution (relative to TSS)")
    axes[1].grid(True, alpha=0.3)
    legend_handles = [
        Line2D([0], [0], color="#2563eb", linewidth=2.0, label="Contribution"),
        Line2D([0], [0], color="#dc2626", linestyle="--", linewidth=1.2, label="TSS (0 bp)"),
        Patch(facecolor=ELEMENT_COLORS["TATA box"], alpha=0.18, edgecolor="none", label="TATA box"),
        Patch(facecolor=ELEMENT_COLORS["Inr"], alpha=0.18, edgecolor="none", label="Inr"),
        Patch(facecolor=ELEMENT_COLORS["DPE"], alpha=0.18, edgecolor="none", label="DPE"),
    ]
    axes[1].legend(handles=legend_handles, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_occlusion_analysis(
    model: torch.nn.Module,
    tokenizer,
    records: List[Dict[str, Any]],
    device: str,
    max_length: int,
    batch_size: int,
    mask_char: str,
    rel_min: int,
    rel_max: int,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    """Per-site occlusion; compute the contribution matrix (N, max_seq_len) for all samples.

    Order of operations:
      1. Per-site masked inference → store predicted probability at each site per sample
      2. Baseline inference (original sequences)
      3. Difference: contribution = baseline - masked
    """
    n = len(records)
    max_seq_len = max(rec["seq_len"] for rec in records)

    # masked_at_ref[i, pos] would store sample i's probability at its reference site after masking pos
    # The reference site is known only after baseline inference, so storing full curves is tempting
    # Memory-wise, only the value at **that site** would be needed once the reference is known
    # But the reference is not known yet, so full curves would be required → too much memory
    # Writing every masked curve to disk is also impractical
    # Most efficient: a quick baseline pass to fix references, then occlusion — despite preferring occlusion first
    #
    # In practice: baseline is fast (only n sequences, seconds); run it first, fix references,
    # then enter the occlusion loop. Label the baseline progress as "Initializing" rather than a separate stage.

    sequences = [rec["sequence"] for rec in records]

    # Quick baseline inference (fix reference sites; only n sequences, finishes in seconds)
    print("Initializing: baseline inference (fixing reference sites)...")
    baseline_curves = []
    for start in range(0, n, batch_size):
        batch_seqs = sequences[start : start + batch_size]
        baseline_curves.append(predict_batch(model, tokenizer, batch_seqs, device, max_length))
    baseline_curves = np.concatenate(baseline_curves, axis=0)

    baseline_ref, _baseline_peak, ref_indices, ref_types = compute_baseline_scores(
        baseline_curves, records
    )
    print(f"Initialization done: {n} samples, reference sites fixed")

    # Per-site occlusion (main cost; progress bar updates per batch)
    all_contrib = np.zeros((n, max_seq_len), dtype=np.float32)
    n_batches_per_pos = (n + batch_size - 1) // batch_size
    total_batches = max_seq_len * n_batches_per_pos
    print(f"Starting per-site occlusion: {max_seq_len} sites × {n} samples, "
          f"~{total_batches:,} batches")

    pbar = tqdm(total=total_batches, desc="Per-site occlusion",
                unit="batch", dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches "
                           "[{elapsed}<{remaining}, {rate_fmt}] {postfix}")

    for pos in range(max_seq_len):
        pbar.set_postfix_str(f"site {pos+1}/{max_seq_len}")
        masked_seqs = []
        active_indices = []
        for i, rec in enumerate(records):
            if pos >= rec["seq_len"]:
                continue
            masked_seqs.append(mask_sequence(rec["sequence"], pos, mask_char))
            active_indices.append(i)

        if not masked_seqs:
            pbar.update(n_batches_per_pos)
            continue

        masked_curves_chunks = []
        for start in range(0, len(masked_seqs), batch_size):
            batch_seqs = masked_seqs[start : start + batch_size]
            masked_curves_chunks.append(
                predict_batch(model, tokenizer, batch_seqs, device, max_length)
            )
            pbar.update(1)
        remaining_batches = n_batches_per_pos - ((len(masked_seqs) + batch_size - 1) // batch_size)
        if remaining_batches > 0:
            pbar.update(remaining_batches)
        masked_curves = np.concatenate(masked_curves_chunks, axis=0)

        for j, sample_idx in enumerate(active_indices):
            ref_idx = ref_indices[sample_idx]
            delta = baseline_ref[sample_idx] - masked_curves[j, ref_idx]
            all_contrib[sample_idx, pos] = max(delta, 0.0)

    pbar.close()

    return all_contrib, baseline_curves, ref_indices, ref_types


@torch.inference_mode()
def filter_predicted_promoter_records(
    model: torch.nn.Module,
    tokenizer,
    records: List[Dict[str, Any]],
    device: str,
    max_length: int,
    batch_size: int,
    pred_threshold: float,
) -> List[Dict[str, Any]]:
    """Keep only samples that are both model-predicted promoters and true-label promoters."""
    if not records:
        return []

    sequences = [rec["sequence"] for rec in records]
    keep_records: List[Dict[str, Any]] = []
    n_batches = (len(records) + batch_size - 1) // batch_size
    for start in tqdm(
        range(0, len(records), batch_size),
        total=n_batches,
        desc="Filter predicted-positive and true-positive samples",
        unit="batch",
        dynamic_ncols=True,
    ):
        batch_records = records[start : start + batch_size]
        batch_seqs = sequences[start : start + batch_size]
        batch_curves = predict_batch(model, tokenizer, batch_seqs, device, max_length)
        for i, rec in enumerate(batch_records):
            seq_len = rec["seq_len"]
            peak_prob = float(batch_curves[i, :seq_len].max())
            if peak_prob >= pred_threshold and int(rec.get("classification_label", 0)) == 1:
                keep_records.append(rec)

    return keep_records


def main() -> None:
    ap = argparse.ArgumentParser(description="TSS base-level contribution analysis (per-site occlusion on the test set)")
    ap.add_argument("--config", type=str, default="config/training_config.json")
    ap.add_argument(
        "--data-path",
        type=str,
        default="data\\proceed_promoters\\promoter_labeled_gaussian_600_11.csv",
        help="Data CSV path (overrides data.merged_csv_path in the config)",
    )
    ap.add_argument("--checkpoint", type=str, default="checkpoints/best_gaussian_600_11.pth", help="Model checkpoint; inferred from the dataset name by default")
    ap.add_argument(
        "--output-dir",
        type=str,
        default="results/interpretability/tss_contribution",
    )
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="Inference batch size; defaults to the training config")
    ap.add_argument("--mask-char", type=str, default="N", help="Mask base character")
    ap.add_argument("--rel-min", type=int, default=None, help="Lower bound of TSS-relative window (bp); full length by default")
    ap.add_argument("--rel-max", type=int, default=None, help="Upper bound of TSS-relative window (bp); full length by default")
    ap.add_argument("--n-examples", type=int, default=10, help="Number of single-sample example plots to save")
    ap.add_argument(
        "--pred-threshold",
        type=float,
        default=None,
        help="Threshold for 'predicted promoter' (defaults to config.default_threshold)",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max test samples to use; if omitted, use all filtered test samples",
    )
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    config = load_training_config(_resolve_abs_path(args.config))
    data_cfg = config.get("data", {})
    merged_csv_path = _resolve_abs_path(args.data_path)
    train_ratio = float(data_cfg.get("train_ratio", 0.8))
    val_ratio = float(data_cfg.get("val_ratio", 0.1))
    test_ratio = float(data_cfg.get("test_ratio", 0.1))
    random_seed = int(data_cfg.get("random_seed", 42))
    use_negative_samples = bool(config.get("use_negative_samples", True))

    dnabert_cfg = config.get("dnabert", {})
    max_seq_length = int(dnabert_cfg.get("max_seq_length", 600))
    batch_size = int(args.batch_size or config.get("training", {}).get("batch_size", 16))
    pred_threshold = float(
        config.get("default_threshold", 0.2) if args.pred_threshold is None else args.pred_threshold
    )

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint, merged_csv_path)

    output_dir = _resolve_abs_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    allowed_chroms = {f"chr{i}" for i in range(1, 22)}

    records = load_records(
        merged_csv_path,
        use_negative_samples=use_negative_samples,
        max_seq_length=max_seq_length,
        allowed_chroms=allowed_chroms,
    )
    test_records = split_test_records(
        records, train_ratio, val_ratio, test_ratio, random_seed
    )
    if not test_records:
        raise SystemExit("Test set is empty; cannot run contribution analysis")

    n_pos = sum(r["classification_label"] == 1 for r in test_records)
    n_neg = len(test_records) - n_pos
    original_test_size = len(test_records)
    print(f"Test samples: {len(test_records)} (true labels: positives {n_pos}, negatives {n_neg})")
    print(f"Data path: {merged_csv_path}")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Filter: predicted promoter (peak_prob >= {pred_threshold}) and true-label promoter")

    model = _build_model(checkpoint_path, config, device)
    tokenizer = model.tokenizer

    try:
        test_records = filter_predicted_promoter_records(
            model=model,
            tokenizer=tokenizer,
            records=test_records,
            device=device,
            max_length=max_seq_length,
            batch_size=batch_size,
            pred_threshold=pred_threshold,
        )
        if not test_records:
            raise SystemExit(
                f"No samples satisfy: predicted promoter (peak_prob >= {pred_threshold}) and true-label promoter."
            )

        # max_samples applies to the post-filter sample count
        if args.max_samples is not None and args.max_samples > 0:
            test_records = test_records[: min(len(test_records), int(args.max_samples))]

        # Default to full length (full relative range), from final analysis samples
        test_max_seq_len = max(rec["seq_len"] for rec in test_records)
        rel_min = args.rel_min if args.rel_min is not None else -(test_max_seq_len - 1)
        rel_max = args.rel_max if args.rel_max is not None else (test_max_seq_len - 1)
        if rel_min > rel_max:
            raise SystemExit(f"Invalid window: rel_min({rel_min}) > rel_max({rel_max})")

        n_pos = sum(r["classification_label"] == 1 for r in test_records)
        n_neg = len(test_records) - n_pos
        print(
            f"Post-filter samples for interpretability: {len(test_records)} "
            f"(true labels: positives {n_pos}, negatives {n_neg})"
        )

        all_contrib, baseline_curves, ref_indices, ref_types = run_occlusion_analysis(
            model=model,
            tokenizer=tokenizer,
            records=test_records,
            device=device,
            max_length=max_seq_length,
            batch_size=batch_size,
            mask_char=args.mask_char,
            rel_min=rel_min,
            rel_max=rel_max,
        )
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Full test set
    rel_pos, mean_all, std_all = aggregate_relative_contribution(
        all_contrib, test_records, ref_indices, rel_min, rel_max
    )
    plot_relative_curve(
        rel_pos,
        mean_all,
        std_all,
        os.path.join(output_dir, "contribution_relative_all.png"),
        title="Mean base-level contribution on test set (relative to TSS/predicted peak)",
        n_samples=len(test_records),
    )

    # Positive / negative groups
    pos_idx = [i for i, r in enumerate(test_records) if r["classification_label"] == 1]
    neg_idx = [i for i, r in enumerate(test_records) if r["classification_label"] == 0]

    summary: Dict[str, Any] = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": os.path.abspath(checkpoint_path),
        "config": os.path.abspath(_resolve_abs_path(args.config)),
        "merged_csv_path": os.path.abspath(merged_csv_path),
        "random_seed": random_seed,
        "original_test_size": original_test_size,
        "predicted_promoter_threshold": pred_threshold,
        "test_size": len(test_records),
        "positive_size": n_pos,
        "negative_size": n_neg,
        "mask_char": args.mask_char,
        "rel_window": [rel_min, rel_max],
        "method": "occlusion (single-base mask, delta P at reference site)",
    }

    if pos_idx:
        pos_records = [test_records[i] for i in pos_idx]
        pos_contrib = all_contrib[pos_idx]
        pos_ref = [ref_indices[i] for i in pos_idx]
        rel_pos_p, mean_p, std_p = aggregate_relative_contribution(
            pos_contrib, pos_records, pos_ref, rel_min, rel_max
        )
        plot_relative_curve(
            rel_pos_p,
            mean_p,
            std_p,
            os.path.join(output_dir, "contribution_relative_positive.png"),
            title="Positive samples - mean base-level contribution (relative to true TSS)",
            n_samples=len(pos_idx),
        )
        summary["positive_tss_window_mean"] = float(
            mean_p[(rel_pos_p >= -5) & (rel_pos_p <= 5)].mean()
        )
        summary["positive_random_upstream_mean"] = float(
            mean_p[(rel_pos_p >= -80) & (rel_pos_p <= -60)].mean()
        )

    if neg_idx:
        neg_records = [test_records[i] for i in neg_idx]
        neg_contrib = all_contrib[neg_idx]
        neg_ref = [ref_indices[i] for i in neg_idx]
        _, mean_n, std_n = aggregate_relative_contribution(
            neg_contrib, neg_records, neg_ref, rel_min, rel_max
        )
        plot_relative_curve(
            rel_pos,
            mean_n,
            std_n,
            os.path.join(output_dir, "contribution_relative_negative.png"),
            title="Negative samples - mean base-level contribution (relative to predicted peak)",
            n_samples=len(neg_idx),
        )

    # Mean contribution in element regions
    element_stats = {}
    for name, (start, end) in ELEMENT_REGIONS.items():
        mask = (rel_pos >= start) & (rel_pos <= end)
        if mask.any():
            element_stats[name] = {
                "region_bp": [start, end],
                "mean_contribution": float(mean_all[mask].mean()),
            }
    summary["element_region_contribution"] = element_stats

    # Save arrays
    np.save(os.path.join(output_dir, "relative_positions.npy"), rel_pos)
    np.save(os.path.join(output_dir, "mean_contribution_all.npy"), mean_all)
    np.save(os.path.join(output_dir, "std_contribution_all.npy"), std_all)
    np.save(os.path.join(output_dir, "per_sample_contribution.npy"), all_contrib)

    # Single-sample examples (rank by TSS-neighborhood contribution sum; take top + random)
    tss_window_score = []
    for i, rec in enumerate(test_records):
        if rec["classification_label"] != 1:
            continue
        ref = ref_indices[i]
        lo = max(0, ref - 5)
        hi = min(rec["seq_len"], ref + 6)
        tss_window_score.append((all_contrib[i, lo:hi].sum(), i))
    tss_window_score.sort(reverse=True)
    example_indices = [idx for _, idx in tss_window_score[: args.n_examples]]
    if len(example_indices) < args.n_examples:
        example_indices.extend(range(min(args.n_examples - len(example_indices), len(test_records))))

    examples_dir = os.path.join(output_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)
    for rank, idx in enumerate(example_indices):
        plot_example_sample(
            test_records[idx],
            baseline_curves[idx],
            all_contrib[idx],
            ref_indices[idx],
            os.path.join(examples_dir, f"example_{rank + 1}_{test_records[idx]['unique_id']}.png"),
        )

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nAnalysis done; results saved to: {output_dir}")
    print(f"  - contribution_relative_all.png")
    print(f"  - contribution_relative_positive.png")
    print(f"  - summary.json")
    if element_stats:
        print("\nMean contribution in element regions:")
        for name, stats in element_stats.items():
            print(f"  {name}: {stats['mean_contribution']:.6f}")


if __name__ == "__main__":
    main()
