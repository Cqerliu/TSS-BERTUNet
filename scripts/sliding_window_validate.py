#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sliding-window single-model inference: base-level predictions are overlap-averaged into a region curve;
predicted peaks are extracted from the curve and scored against BED ground-truth TSS at tolerances τ=0..τ_max bp:
peak accuracy, miss rate, and false-alarm rate (see docs/sliding_peak_tolerance_metrics.md).

Default behavior: choose the matching analysis-region JSON under results/ from `--chromosome`.

Validation metrics are based on those analysis regions: within each region, overlap-averaged base-level
predictions are **min-max normalized to [0,1]**, then peaks are taken at thresholds **0.1~0.9 (step 0.1)**
and tolerance curves are computed. The default PNG plots the full-length normalized region curve;
`--peak-threshold` only controls the peak-picking threshold used in the summary and plot annotations
(interpret on [0,1]).

Example:
  python scripts/sliding_window_validate.py \\
    --chromosome chr22 \\
    --checkpoints checkpoints/best_gaussian_600_11.pth \\
    --stride 100

  Default PNG / *_metrics.json / *_summary.json stems are derived from data.merged_csv_path in the config
  (e.g. promoter_labeled_gaussian_50_11.csv → validate_gaussian_50); override with --merged-csv.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

for _font_name in ("Microsoft YaHei", "SimHei", "NSimSun", "DengXian", "PingFang SC", "Noto Sans CJK SC"):
    if any(_font_name == f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [
            _font_name,
        ] + [x for x in plt.rcParams.get("font.sans-serif", []) if x != _font_name]
        break
plt.rcParams.setdefault("axes.unicode_minus", False)

from data import create_dataloader  # noqa: E402
from data.generate_labels.config import BED_PATH, CHR_TO_CONTIG, GENOME_FASTA_PATH  # noqa: E402
from data.generate_labels.io_utils import ensure_fai, load_bed  # noqa: E402
from utils import load_training_config  # noqa: E402


def _resolve_abs_path(project_root: str, path_str: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(project_root, path_str)


def _run_inference(
    model: torch.nn.Module,
    val_loader,
    device: str,
    *,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_predictions: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    iterator = tqdm(val_loader, desc="Validating") if show_progress else val_loader
    with torch.inference_mode():
        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            offset_mapping = batch.get("offset_mapping", None)
            original_seq_length = batch.get("original_seq_length", None)
            original_sequence = batch.get("original_sequence", None)
            if offset_mapping is not None:
                offset_mapping = [
                    om.to(device) if isinstance(om, torch.Tensor) else om
                    for om in offset_mapping
                ]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target=labels,
                offset_mapping=offset_mapping,
                original_seq_length=original_seq_length,
                original_sequence=original_sequence,
            )
            predictions = outputs[0] if isinstance(outputs, tuple) else outputs
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(labels.detach().cpu())

    if not all_predictions:
        return torch.empty(0), torch.empty(0)

    max_len = max(
        max(t.shape[-1] for t in all_predictions),
        max(t.shape[-1] for t in all_targets),
    )
    padded_predictions: List[torch.Tensor] = []
    padded_targets: List[torch.Tensor] = []
    for pred, tgt in zip(all_predictions, all_targets):
        if pred.ndim == 3 and pred.shape[-1] < max_len:
            pad = torch.zeros(pred.shape[0], pred.shape[1], max_len - pred.shape[-1], dtype=pred.dtype)
            pred = torch.cat([pred, pad], dim=2)
        if tgt.ndim == 3 and tgt.shape[-1] < max_len:
            pad = torch.zeros(tgt.shape[0], tgt.shape[1], max_len - tgt.shape[-1], dtype=tgt.dtype)
            tgt = torch.cat([tgt, pad], dim=2)
        padded_predictions.append(pred)
        padded_targets.append(tgt)

    predictions = torch.cat(padded_predictions, dim=0)
    targets = torch.cat(padded_targets, dim=0)
    return predictions, targets


def _build_model(
    checkpoint_path: str,
    config: Dict[str, Any],
    project_root: str,
    device: str,
) -> torch.nn.Module:
    from model import DNABERTForTSSLocalization

    dnabert_cfg = config.get("dnabert", {})
    reg_cfg = config.get("regression_head", {})
    use_only_unet = config.get("use_only_unet", False)
    dnabert_model_path = _resolve_abs_path(project_root, dnabert_cfg.get("model_path", "DNABERT-2-117M"))

    model = DNABERTForTSSLocalization(
        dnabert_path=dnabert_model_path,
        hidden_size=dnabert_cfg.get("hidden_size", 768),
        max_seq_length=dnabert_cfg.get("max_seq_length", 512),
        dropout=dnabert_cfg.get("dropout", 0.1),
        freeze_dnabert=dnabert_cfg.get("freeze_dnabert", False),
        regression_hidden_dim=reg_cfg.get("hidden_dim", dnabert_cfg.get("hidden_size", 768) // 2),
        regression_dropout=reg_cfg.get("dropout", dnabert_cfg.get("dropout", 0.1)),
        regression_num_layers=reg_cfg.get("num_layers", 2),
        use_unet=reg_cfg.get("use_unet", True),
        use_only_unet=use_only_unet,
    )

    ckpt_obj = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        state_dict = ckpt_obj["model_state_dict"]
    elif isinstance(ckpt_obj, dict):
        state_dict = ckpt_obj
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    return model


def _normalize_chr(c: str) -> str:
    s = str(c).strip()
    if not s:
        return ""
    sl = s.lower()
    raw = sl[3:] if sl.startswith("chr") else sl
    if raw == "x":
        return "chrX"
    if raw == "y":
        return "chrY"
    if raw == "m":
        return "chrM"
    if raw == "mt":
        return "chrMT"
    return "chr" + raw


def _infer_window_bp_from_checkpoint_path(ckpt_path: str) -> Optional[int]:
    base = os.path.basename(ckpt_path)
    m = re.search(r"_(\d+)(?:_\d+)?\.pth\s*$", base, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _stem_validate_from_merged_csv(merged_csv_path: Optional[str]) -> str:
    """
    Build the result-file stem from the training CSV, e.g.:
    promoter_labeled_gaussian_50_11.csv -> validate_gaussian_50
    (strip the promoter_labeled_ prefix and the trailing _<fold> numeric segment).
    """
    if not merged_csv_path or not str(merged_csv_path).strip():
        return "validate"
    base = os.path.splitext(os.path.basename(str(merged_csv_path).strip()))[0]
    base = re.sub(r"^promoter_labeled_", "", base, count=1, flags=re.IGNORECASE)
    parts = base.split("_")
    if len(parts) >= 3 and parts[-1].isdigit():
        base = "_".join(parts[:-1])
    return f"validate_{base}"


def _window_starts(full_len: int, win: int, stride: int) -> List[int]:
    if win <= 0 or stride <= 0:
        raise ValueError("win and stride must be positive integers")
    if full_len < win:
        raise ValueError(f"region length {full_len} is smaller than window {win}")
    last = full_len - win
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _fetch_region_sequence(
    genome_path: str,
    contig: str,
    g_start: int,
    g_end: int,
) -> str:
    ensure_fai(genome_path)
    from pyfaidx import Fasta

    fa = Fasta(genome_path, one_based_attributes=False)
    try:
        return str(fa[contig][g_start:g_end]).replace("\n", "").replace(" ", "").upper()
    finally:
        fa.close()


def _load_bed_tss_relative(
    bed_path: str,
    chrom: str,
    g_start: int,
    g_end: int,
) -> np.ndarray:
    df = load_bed(bed_path)
    df = df[df["chr"].map(_normalize_chr) == _normalize_chr(chrom)]
    pos = df["start"].astype(np.int64).values
    pos = pos[(pos >= g_start) & (pos < g_end)] - int(g_start)
    pos = np.unique(pos)
    return pos.astype(np.int64, copy=False)


def _read_region_start_end(obj: Dict[str, Any]) -> Tuple[int, int]:
    """Accept legacy chr22_genomic_* fields and newer chromosome_genomic_* fields."""
    if "chromosome_genomic_start_0based" in obj and "chromosome_genomic_end_0based_exclusive" in obj:
        return int(obj["chromosome_genomic_start_0based"]), int(obj["chromosome_genomic_end_0based_exclusive"])
    if "chr22_genomic_start_0based" in obj and "chr22_genomic_end_0based_exclusive" in obj:
        return int(obj["chr22_genomic_start_0based"]), int(obj["chr22_genomic_end_0based_exclusive"])
    raise KeyError("region-json missing interval fields: need chromosome_genomic_* or chr22_genomic_*")


def _load_regions_from_json(path: str) -> Tuple[str, List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    chrom = str(j.get("chromosome", "")).strip()
    regions: List[Dict[str, Any]] = []
    if isinstance(j.get("windows"), list):
        for i, w in enumerate(j["windows"]):
            g0, g1 = _read_region_start_end(w)
            tss_rel = [int(x) for x in w.get("tss_relative_positions_0based", [])]
            regions.append(
                {
                    "region_index": int(w.get("window_index", i)),
                    "g_start": g0,
                    "g_end": g1,
                    "tss_rel": tss_rel,
                }
            )
    else:
        g0, g1 = _read_region_start_end(j)
        tss_rel = [int(x) for x in j.get("tss_relative_positions_0based", [])]
        regions.append({"region_index": 0, "g_start": g0, "g_end": g1, "tss_rel": tss_rel})
    return chrom, regions


def _resolve_region_json_by_chromosome(project_root: str, chrom: str) -> str:
    """Auto-select region-json under results/ by chromosome."""
    results_dir = os.path.join(project_root, "results")
    c = _normalize_chr(chrom)
    exact = os.path.join(results_dir, f"{c}_tss_cover_windows.json")
    if os.path.isfile(exact):
        return exact
    pattern = os.path.join(results_dir, f"*{c}*tss*cover*windows*.json")
    cand = [p for p in glob.glob(pattern) if os.path.isfile(p)]
    if cand:
        return sorted(cand)[0]
    raise SystemExit(
        f"No region-json found for {c}. First generate "
        f"`{c}_tss_cover_windows.json`。"
    )


def _parse_target_chromosomes(chromosomes_arg: Optional[str], chromosome_arg: str) -> List[str]:
    if chromosomes_arg and str(chromosomes_arg).strip():
        raw = [x.strip() for x in str(chromosomes_arg).split(",") if x.strip()]
    else:
        raw = [str(chromosome_arg).strip()]
    out: List[str] = []
    seen = set()
    for c in raw:
        cn = _normalize_chr(c)
        if cn and cn not in seen:
            out.append(cn)
            seen.add(cn)
    if not out:
        raise SystemExit("Specify at least one chromosome (--chromosome or --chromosomes)")
    return out


def _sample_regions_windows(
    regions: List[Dict[str, Any]],
    n: int,
    seed: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Sample n windows without replacement from the region-json windows list.
    Returns (sampled regions list, 0-based indices into the original list).
    """
    m = len(regions)
    if n > m:
        raise ValueError(f"sample size {n} exceeds number of windows in JSON {m}")
    if n < 1:
        raise ValueError("sample size must be a positive integer")
    rng = np.random.default_rng(seed)
    pick = rng.choice(m, size=n, replace=False)
    pick = np.sort(pick.astype(np.int64))
    out = [regions[int(i)] for i in pick]
    return out, pick.tolist()


def _stage_config(base: Dict[str, Any], max_seq_length: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg.setdefault("dnabert", {})["max_seq_length"] = int(max_seq_length)
    return cfg


def _sliding_average_predictions_chunked(
    sequence: str,
    model: torch.nn.Module,
    device: str,
    window_bp: int,
    stride: int,
    batch_size: int,
    chunk_windows: int,
) -> np.ndarray:
    """Build DataLoaders in chunks to avoid holding millions of substrings for a whole chromosome."""
    L = len(sequence)
    starts = _window_starts(L, window_bp, stride)
    accum = np.zeros(L, dtype=np.float64)
    counts = np.zeros(L, dtype=np.float64)

    n_chunks = (len(starts) + chunk_windows - 1) // chunk_windows
    for chunk_lo in tqdm(range(0, len(starts), chunk_windows), desc="Sliding-window inference", total=n_chunks):
        chunk = starts[chunk_lo : chunk_lo + chunk_windows]
        windows = [sequence[s : s + window_bp] for s in chunk]
        labels = [np.zeros(window_bp, dtype=np.float32) for _ in chunk]
        loader = create_dataloader(
            windows,
            labels,
            model.tokenizer,
            batch_size=batch_size,
            shuffle=False,
            max_length=window_bp,
        )
        predictions, _ = _run_inference(model, loader, device=device)
        if predictions.ndim == 3:
            predictions = predictions.squeeze(1)
        pred_np = predictions.detach().cpu().float().numpy()

        for j, s in enumerate(chunk):
            slice_pred = pred_np[j]
            wl = min(window_bp, slice_pred.shape[-1], L - s)
            if wl <= 0:
                continue
            accum[s : s + wl] += slice_pred[:wl]
            counts[s : s + wl] += 1.0

    denom = np.maximum(counts, 1e-12)
    return (accum / denom).astype(np.float32)


# Metric scan: peak-height thresholds on the normalized curve (independent of raw model logits)
PEAK_THRESHOLD_SCAN: Tuple[float, ...] = tuple(round(i / 10.0, 1) for i in range(1, 10))


def _minmax_normalize_predictions(
    y: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Min-max scale a single region to [0,1]. Near-flat curves become all zeros to avoid divide-by-zero artifacts.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    lo = float(np.min(y)) if y.size else 0.0
    hi = float(np.max(y)) if y.size else 0.0
    span = hi - lo
    if span <= eps:
        out = np.zeros(y.shape, dtype=np.float32)
        return out, {
            "method": "min_max",
            "min": lo,
            "max": hi,
            "span": span,
            "constant_or_near_constant": True,
        }
    out = ((y - lo) / span).astype(np.float32)
    return out, {
        "method": "min_max",
        "min": lo,
        "max": hi,
        "span": span,
        "constant_or_near_constant": False,
    }


def _find_predicted_peaks_1d(
    y: np.ndarray,
    height: float,
    min_distance: int,
) -> np.ndarray:
    """
    Strict local maxima with y >= height; greedily keep by score descending with centers at least
    min_distance bp apart (mark [i - min_distance + 1, i + min_distance) occupied at index i).
    """
    y = np.asarray(y, dtype=np.float64)
    L = int(y.shape[0])
    if L == 0:
        return np.array([], dtype=np.int64)
    cand = np.zeros(L, dtype=bool)
    if L >= 3:
        cand[1:-1] = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
    elif L == 2:
        cand[0] = y[0] >= y[1]
        cand[1] = y[1] >= y[0]
    else:
        cand[0] = True
    cand &= y >= float(height)
    idx = np.flatnonzero(cand)
    if idx.size == 0:
        return np.array([], dtype=np.int64)

    d = max(1, int(min_distance))
    order = idx[np.argsort(-y[idx])]
    blocked = np.zeros(L, dtype=bool)
    kept: List[int] = []
    for i in order:
        ii = int(i)
        if blocked[ii]:
            continue
        kept.append(ii)
        lo = max(0, ii - d + 1)
        hi = min(L, ii + d)
        blocked[lo:hi] = True
    return np.array(sorted(kept), dtype=np.int64)


def _build_tolerance_metrics(
    tss_idx: np.ndarray,
    peak_idx: np.ndarray,
    tolerance_max: int,
    tolerance_step: int,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """
    For each tolerance τ, compute TSS-side peak_accuracy / miss_rate, peak-side false_alarm_rate,
    and predicted_tss_accuracy (fraction of predicted peaks within ≤τ of the nearest true TSS;
    complementary to false_alarm_rate).
    Returns (table rows, d_tss, d_peak) for debugging; d_tss[i]=distance from the i-th TSS to the nearest peak,
    and likewise for d_peak.
    """
    tss_idx = np.asarray(tss_idx, dtype=np.int64).reshape(-1)
    peak_idx = np.asarray(peak_idx, dtype=np.int64).reshape(-1)
    nt = int(tss_idx.shape[0])
    npk = int(peak_idx.shape[0])

    if nt == 0:
        d_tss = np.array([], dtype=np.float64)
    elif npk == 0:
        d_tss = np.full(nt, np.inf, dtype=np.float64)
    else:
        d_tss = np.min(np.abs(tss_idx[:, None] - peak_idx[None, :]), axis=1).astype(np.float64)

    if npk == 0:
        d_peak = np.array([], dtype=np.float64)
    elif nt == 0:
        d_peak = np.full(npk, np.inf, dtype=np.float64)
    else:
        d_peak = np.min(np.abs(peak_idx[:, None] - tss_idx[None, :]), axis=1).astype(np.float64)

    rows: List[Dict[str, Any]] = []
    step = max(1, int(tolerance_step))
    taumax = max(0, int(tolerance_max))
    for tau in range(0, taumax + 1, step):
        if nt > 0:
            acc = float(np.mean(d_tss <= tau))
            miss = float(1.0 - acc)
        else:
            acc = None
            miss = None
        if npk > 0:
            far = float(np.mean(d_peak > tau))
            pred_tss_acc = float(np.mean(d_peak <= tau))
        else:
            far = None
            pred_tss_acc = None
        rows.append(
            {
                "tolerance_bp": int(tau),
                "peak_accuracy": acc,
                "miss_rate": miss,
                "false_alarm_rate": far,
                "predicted_tss_accuracy": pred_tss_acc,
            }
        )
    return rows, d_tss, d_peak


def _aggregate_tolerance_curve_from_distance_parts(
    d_tss_parts: List[np.ndarray],
    d_peak_parts: List[np.ndarray],
    *,
    tolerance_max: int,
    tolerance_step: int,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """Concatenate multi-region d_tss / d_peak and build the tolerance curve for one threshold."""
    d_tss_all = np.concatenate(d_tss_parts) if d_tss_parts else np.array([], dtype=np.float64)
    d_peak_all = np.concatenate(d_peak_parts) if d_peak_parts else np.array([], dtype=np.float64)
    nt_l = int(d_tss_all.shape[0])
    npk_l = int(d_peak_all.shape[0])
    agg_rows: List[Dict[str, Any]] = []
    for tau in range(0, int(tolerance_max) + 1, max(1, int(tolerance_step))):
        acc = float(np.mean(d_tss_all <= tau)) if nt_l > 0 else None
        miss = (1.0 - acc) if acc is not None else None
        if npk_l > 0:
            far = float(np.mean(d_peak_all > tau))
            pred_tss_acc = float(np.mean(d_peak_all <= tau))
        else:
            far = None
            pred_tss_acc = None
        agg_rows.append(
            {
                "tolerance_bp": int(tau),
                "peak_accuracy": acc,
                "miss_rate": miss,
                "false_alarm_rate": far,
                "predicted_tss_accuracy": pred_tss_acc,
            }
        )
    return nt_l, npk_l, agg_rows


def _compute_peak_tolerance_metrics(
    curve: np.ndarray,
    tss_idx: np.ndarray,
    *,
    peak_threshold: float,
    min_peak_distance_bp: int,
    tolerance_max: int,
    tolerance_step: int,
) -> Dict[str, Any]:
    L = int(curve.shape[0])
    tss_idx = np.unique(tss_idx[(tss_idx >= 0) & (tss_idx < L)])
    peak_idx = _find_predicted_peaks_1d(curve, peak_threshold, min_peak_distance_bp)
    rows, _d_tss, _d_peak = _build_tolerance_metrics(
        tss_idx,
        peak_idx,
        tolerance_max=tolerance_max,
        tolerance_step=tolerance_step,
    )
    return {
        "n_true_tss": int(tss_idx.shape[0]),
        "n_predicted_peaks": int(peak_idx.shape[0]),
        "predicted_tss_relative_positions_0based": peak_idx.astype(np.int64).tolist(),
        "peak_threshold": float(peak_threshold),
        "min_peak_distance_bp": int(min_peak_distance_bp),
        "tolerance_max_bp": int(tolerance_max),
        "tolerance_step_bp": int(max(1, tolerance_step)),
        "tolerance_curve": rows,
        "metrics_doc": "docs/sliding_peak_tolerance_metrics.md",
    }


def _plot_random_window_validation(
    curve: np.ndarray,
    tss_idx: np.ndarray,
    pred_tss_idx: np.ndarray,
    *,
    chrom: str,
    contig: str,
    L: int,
    win_len: int,
    stride: int,
    out_path: str,
) -> Dict[str, Any]:
    """
    Plot the prediction curve over the full interval [0, L) and mark true/predicted TSS in the region.
    Returns summary metadata (window start, length).
    """
    seg_len = int(L)
    s0 = 0
    seg = curve[s0 : s0 + seg_len]
    xv = np.arange(seg_len, dtype=np.int32)

    tss_in = tss_idx[(tss_idx >= s0) & (tss_idx < s0 + seg_len)]
    rel = (tss_in - s0).astype(np.int64)
    y_at = seg[rel].astype(np.float64) if rel.size else np.array([], dtype=np.float64)
    pred_in = pred_tss_idx[(pred_tss_idx >= s0) & (pred_tss_idx < s0 + seg_len)]
    pred_rel = (pred_in - s0).astype(np.int64)
    pred_y = seg[pred_rel].astype(np.float64) if pred_rel.size else np.array([], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(16, 4.5))
    ax.plot(
        xv,
        seg,
        color="#1f77b4",
        label=f"{win_len} bp model · stride {stride}",
        linewidth=0.9,
        alpha=0.95,
        zorder=1,
    )

    if rel.size > 0:
        for t in rel:
            ax.axvline(
                int(t),
                color="#c0392b",
                linestyle="--",
                linewidth=0.9,
                alpha=0.65,
                zorder=2,
            )
        ax.scatter(
            rel,
            y_at,
            s=52,
            c="#e74c3c",
            marker="v",
            edgecolors="#7b241c",
            linewidths=0.8,
            zorder=5,
            label=f"True TSS (in window n={rel.size})",
        )
        if rel.size <= 40:
            for i_t, t in enumerate(rel):
                ax.annotate(
                    f"true {int(t)}",
                    xy=(int(t), y_at[i_t]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=5,
                    color="#641e16",
                    rotation=90,
                    zorder=6,
                )
    else:
        ax.text(
            0.5,
            0.02,
            "No BED TSS in this region",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
            color="#666",
        )

    if pred_rel.size > 0:
        ax.scatter(
            pred_rel,
            pred_y,
            s=18,
            c="#2ca02c",
            marker="o",
            edgecolors="none",
            alpha=0.9,
            zorder=4,
            label=f"Predicted TSS (in window n={pred_rel.size})",
        )
        if pred_rel.size <= 50:
            for i_p, p in enumerate(pred_rel):
                ax.annotate(
                    f"pred {int(p)}",
                    xy=(int(p), pred_y[i_p]),
                    xytext=(0, -10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=5,
                    color="#145a32",
                    rotation=90,
                    zorder=6,
                )

    ax.set_xlim(0, max(seg_len - 1, 0))
    ax.set_xlabel("Relative position within window (bp)")
    ax.set_ylabel("Overlap-averaged prediction score")
    ax.set_title(
        f"{chrom} full-region {seg_len:,} bp visualization · relative coords · {contig}"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "viz_window_length_bp": int(seg_len),
        "viz_window_start_rel_0based": int(s0),
        "viz_window_end_rel_0based_exclusive": int(s0 + seg_len),
        "viz_window_tss_count": int(rel.size),
        "viz_window_predicted_tss_relative_positions_0based": pred_rel.astype(np.int64).tolist(),
        "viz_output_png": os.path.abspath(out_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-select region-json by chromosome: sliding-window single model + TSS metrics")
    ap.add_argument("--genome", type=str, default=GENOME_FASTA_PATH, help="Reference genome FASTA")
    ap.add_argument("--chromosome", type=str, default="chr22", help="Chromosome name (for BED filtering)")
    ap.add_argument(
        "--chromosomes",
        type=str,
        default="chr22,chrX,chrY",
        help="Multiple chromosomes, comma-separated (e.g. chr22,chrX); overrides --chromosome when set",
    )
    ap.add_argument(
        "--checkpoints",
        "-C",
        type=str,
        default="checkpoints/best_gaussian_600_11.pth",
        dest="checkpoints",
        metavar="PATH",
        help="Single checkpoint path (.pth)",
    )
    ap.add_argument(
        "--length",
        type=int,
        default=None,
        help="Model input window length (bp); inferred from filename if omitted",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=600,
        help="Sliding stride (bp)",
    )
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument(
        "--chunk-windows",
        type=int,
        default=512,
        help="Number of windows merged per inference round (memory only; does not change math)",
    )
    ap.add_argument("--config", type=str, default="config/sliding_window_validate_config.json")
    ap.add_argument(
        "--merged-csv",
        type=str,
        default=None,
        help="Training merged CSV (for result naming; defaults to data.merged_csv_path in config)",
    )
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu")
    ap.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Visualization PNG path; default results/<stem derived from data CSV>.png",
    )
    ap.add_argument(
        "--viz-region-index",
        type=int,
        default=4,
        help="region_index to visualize (window index from auto-selected region-json); defaults to the first window",
    )
    ap.add_argument("--save-npz", type=str, default=None, help="Save pred / positions (npz can be large for long regions)")
    ap.add_argument("--bed", type=str, default=BED_PATH, help="TSS BED such as EPDnew")
    ap.add_argument(
        "--peak-threshold",
        type=float,
        default=0.5,
        help="Peak-picking threshold on the normalized [0,1] curve for PNG annotations and the default peak list in summary; metrics JSON also writes the full 0.1~0.9 scan",
    )
    ap.add_argument(
        "--min-peak-distance",
        type=int,
        default=None,
        help="Minimum spacing between predicted peaks (bp), greedy NMS; default max(50, model_window//2)",
    )
    ap.add_argument(
        "--tolerance-max",
        type=int,
        default=300,
        help="Tolerance upper bound τ (bp), inclusive; with step yields 0,step,...,τ",
    )
    ap.add_argument(
        "--tolerance-step",
        type=int,
        default=1,
        help="Tolerance scan step (bp); use 5 or 10 to shrink metrics JSON size",
    )
    ap.add_argument(
        "--sample-windows",
        type=int,
        default=None,
        metavar="N",
        help="Randomly sample N windows from the auto-selected region-json for validation; use all if unset",
    )
    ap.add_argument(
        "--sample-windows-seed",
        type=int,
        default=42,
        help="With --sample-windows, fix the random sample; different each run if unset",
    )
    args = ap.parse_args()

    base_cfg = load_training_config(_resolve_abs_path(PROJECT_ROOT, args.config))
    merged_csv_resolved = args.merged_csv or base_cfg.get("data", {}).get("merged_csv_path")
    if merged_csv_resolved:
        merged_csv_resolved = _resolve_abs_path(PROJECT_ROOT, str(merged_csv_resolved))
    out_stem = _stem_validate_from_merged_csv(merged_csv_resolved)
    if args.output is None:
        args.output = os.path.join(PROJECT_ROOT, "results", f"{out_stem}.png")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    target_chroms = _parse_target_chromosomes(args.chromosomes, args.chromosome)
    genome_path = _resolve_abs_path(PROJECT_ROOT, args.genome)

    regions: List[Dict[str, Any]] = []
    selected_region_json_by_chrom: Dict[str, str] = {}
    n_windows_in_json = 0
    for chrom_req in target_chroms:
        region_json_path = _resolve_region_json_by_chromosome(PROJECT_ROOT, chrom_req)
        print(f"Auto-selected region-json: {region_json_path}")
        chrom_json, regions_one = _load_regions_from_json(region_json_path)
        chrom_eff = _normalize_chr(chrom_json) if chrom_json else chrom_req
        contig_eff = CHR_TO_CONTIG.get(chrom_eff, chrom_eff)
        selected_region_json_by_chrom[chrom_eff] = os.path.abspath(region_json_path)
        for rr in regions_one:
            regions.append(
                {
                    **rr,
                    "chromosome": chrom_eff,
                    "contig": contig_eff,
                    "source_region_json": os.path.abspath(region_json_path),
                }
            )
        n_windows_in_json += len(regions_one)

    if not regions:
        raise SystemExit("Auto-selected region-json contains no valid regions")
    sample_pick_idx: Optional[List[int]] = None
    if args.sample_windows is not None:
        n_s = int(args.sample_windows)
        if n_s < 1:
            raise SystemExit("--sample-windows must be a positive integer")
        if n_s > n_windows_in_json:
            raise SystemExit(f"--sample-windows={n_s} exceeds number of windows in JSON {n_windows_in_json}")
        try:
            regions, sample_pick_idx = _sample_regions_windows(regions, n_s, args.sample_windows_seed)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        print(
            f"Randomly sampled {len(regions)}/{n_windows_in_json} windows from region-json "
            f"(original indices {sample_pick_idx})"
        )

    ckpt_resolved = _resolve_abs_path(PROJECT_ROOT, args.checkpoints)
    if not os.path.isfile(ckpt_resolved):
        raise SystemExit(f"checkpoint not found: {ckpt_resolved}")

    win_len = args.length if args.length is not None else _infer_window_bp_from_checkpoint_path(ckpt_resolved)
    if win_len is None:
        raise SystemExit("Cannot determine window length; pass --length (bp) explicitly")
    for r in regions:
        r_len = int(r["g_end"]) - int(r["g_start"])
        if r_len <= 0:
            raise SystemExit("region-json has invalid interval: end must be greater than start")
        if win_len > r_len:
            raise SystemExit(f"model window {win_len} bp exceeds region length {r_len} bp (region_index={r['region_index']})")

    peak_th = (
        float(args.peak_threshold)
        if args.peak_threshold is not None
        else float(base_cfg.get("default_threshold", 0.2))
    )
    min_peak_dist = args.min_peak_distance
    if min_peak_dist is None:
        min_peak_dist = max(50, int(win_len) // 2)

    meta: Dict[str, Any] = {
        "chromosomes": target_chroms,
        "refseq_contigs": {c: CHR_TO_CONTIG.get(c, c) for c in target_chroms},
        "n_regions": int(len(regions)),
        "n_windows_in_json": int(n_windows_in_json),
        "stride": int(args.stride),
        "checkpoint": ckpt_resolved,
        "window_length_bp": int(win_len),
        "chunk_windows": int(args.chunk_windows),
        "region_json_files": selected_region_json_by_chrom,
        "merged_csv_path": merged_csv_resolved,
        "result_name_stem": out_stem,
        "bed_file": os.path.abspath(_resolve_abs_path(PROJECT_ROOT, args.bed)),
        "sample_windows": int(args.sample_windows) if args.sample_windows is not None else None,
        "sample_windows_seed": args.sample_windows_seed,
        "sampled_indices_in_json_windows": sample_pick_idx,
        "prediction_normalization": {
            "method": "min_max_per_region",
            "range": "[0, 1]",
            "note": "Independent min-max per evaluation sub-region; then pick peaks per threshold and concatenate distances for aggregated metrics",
        },
        "peak_threshold_scan": list(PEAK_THRESHOLD_SCAN),
        "summary_peak_threshold": float(peak_th),
    }

    stage_cfg = _stage_config(base_cfg, win_len)
    model = _build_model(ckpt_resolved, stage_cfg, PROJECT_ROOT, device)
    try:
        per_region_metrics: List[Dict[str, Any]] = []
        all_d_by_th: Dict[float, List[np.ndarray]] = {float(t): [] for t in PEAK_THRESHOLD_SCAN}
        all_d_peak_by_th: Dict[float, List[np.ndarray]] = {float(t): [] for t in PEAK_THRESHOLD_SCAN}
        first_curve: Optional[np.ndarray] = None
        first_tss: Optional[np.ndarray] = None
        first_L: Optional[int] = None
        first_region_index: Optional[int] = None
        first_region_chrom: Optional[str] = None
        first_region_contig: Optional[str] = None

        for r in tqdm(regions, desc="Region evaluation", total=len(regions)):
            chrom_r = str(r.get("chromosome", "")).strip() or target_chroms[0]
            contig_r = str(r.get("contig", "")).strip() or CHR_TO_CONTIG.get(chrom_r, chrom_r)
            g_start = int(r["g_start"])
            g_end = int(r["g_end"])
            tss_rel_json = [int(x) for x in r.get("tss_rel", [])]
            tss_from_bed = _load_bed_tss_relative(
                _resolve_abs_path(PROJECT_ROOT, args.bed), chrom_r, g_start, g_end
            )
            tss_rel = tss_rel_json if tss_rel_json else tss_from_bed.tolist()
            tss_arr = np.array(sorted(set(tss_rel)), dtype=np.int64)
            sequence = _fetch_region_sequence(genome_path, contig_r, g_start, g_end)
            L = len(sequence)
            if L != g_end - g_start:
                raise RuntimeError(f"sequence length {L} inconsistent with [{g_start},{g_end})")
            tss_arr = np.unique(tss_arr[(tss_arr >= 0) & (tss_arr < L)])
            n_windows = len(_window_starts(L, win_len, int(args.stride)))
            print(
                f"{chrom_r} region_index={int(r['region_index'])} · [{g_start:,}, {g_end:,}) · L={L:,} bp · "
                f"W={win_len} · stride {int(args.stride)} → {n_windows:,} windows"
            )
            curve_raw = _sliding_average_predictions_chunked(
                sequence,
                model,
                device,
                window_bp=win_len,
                stride=int(args.stride),
                batch_size=int(args.batch_size),
                chunk_windows=int(args.chunk_windows),
            )
            curve, norm_meta = _minmax_normalize_predictions(curve_raw)
            per_th_rows: List[Dict[str, Any]] = []
            for th in PEAK_THRESHOLD_SCAN:
                peak_idx_t = _find_predicted_peaks_1d(curve, float(th), int(min_peak_dist))
                rows_t, d_tss_t, d_peak_t = _build_tolerance_metrics(
                    tss_arr,
                    peak_idx_t,
                    tolerance_max=int(args.tolerance_max),
                    tolerance_step=int(args.tolerance_step),
                )
                per_th_rows.append(
                    {
                        "peak_threshold": float(th),
                        "n_predicted_peaks": int(peak_idx_t.shape[0]),
                        "predicted_tss_relative_positions_0based": peak_idx_t.astype(np.int64).tolist(),
                        "tolerance_curve": rows_t,
                    }
                )
                all_d_by_th[float(th)].append(d_tss_t)
                all_d_peak_by_th[float(th)].append(d_peak_t)

            peak_idx_summary = _find_predicted_peaks_1d(curve, float(peak_th), int(min_peak_dist))
            per_region_metrics.append(
                {
                    "chromosome": chrom_r,
                    "refseq_contig": contig_r,
                    "source_region_json": str(r.get("source_region_json", "")),
                    "region_index": int(r["region_index"]),
                    "genomic_start_0based": g_start,
                    "genomic_end_0based_exclusive": g_end,
                    "sequence_length_bp": int(L),
                    "n_true_tss": int(tss_arr.shape[0]),
                    "true_tss_relative_positions_0based": tss_arr.astype(np.int64).tolist(),
                    "normalization": norm_meta,
                    "per_peak_threshold": per_th_rows,
                    "n_predicted_peaks": int(peak_idx_summary.shape[0]),
                    "predicted_tss_relative_positions_0based": peak_idx_summary.astype(np.int64).tolist(),
                    "summary_peak_threshold": float(peak_th),
                }
            )
            if first_curve is None:
                first_curve = curve
                first_tss = tss_arr
                first_L = int(L)
                first_region_index = int(r["region_index"])
                first_region_chrom = chrom_r
                first_region_contig = contig_r

            if (
                args.viz_region_index is not None
                and int(r["region_index"]) == int(args.viz_region_index)
            ):
                first_curve = curve
                first_tss = tss_arr
                first_L = int(L)
                first_region_index = int(r["region_index"])
                first_region_chrom = chrom_r
                first_region_contig = contig_r
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    first_pred: Optional[np.ndarray] = None
    if first_curve is not None:
        first_pred = _find_predicted_peaks_1d(first_curve, float(peak_th), int(min_peak_dist))

    per_peak_threshold_global: List[Dict[str, Any]] = []
    nt_aggregate: Optional[int] = None
    for th in PEAK_THRESHOLD_SCAN:
        d_parts = all_d_by_th[float(th)]
        dp_parts = all_d_peak_by_th[float(th)]
        nti, npki, agg_rows = _aggregate_tolerance_curve_from_distance_parts(
            d_parts,
            dp_parts,
            tolerance_max=int(args.tolerance_max),
            tolerance_step=int(args.tolerance_step),
        )
        if nt_aggregate is None:
            nt_aggregate = int(nti)
        per_peak_threshold_global.append(
            {
                "peak_threshold": float(th),
                "n_true_tss": int(nti),
                "n_predicted_peaks": int(npki),
                "min_peak_distance_bp": int(min_peak_dist),
                "tolerance_max_bp": int(args.tolerance_max),
                "tolerance_step_bp": int(max(1, int(args.tolerance_step))),
                "tolerance_curve": agg_rows,
            }
        )

    metrics: Dict[str, Any] = {
        "n_true_tss": int(nt_aggregate if nt_aggregate is not None else 0),
        "prediction_normalization_note": "See meta.prediction_normalization; metrics use per-region min-max curves; predicted_tss_accuracy in tolerance_curve is the predicted-peak hit rate (complementary to false_alarm_rate)",
        "min_peak_distance_bp": int(min_peak_dist),
        "tolerance_max_bp": int(args.tolerance_max),
        "tolerance_step_bp": int(max(1, int(args.tolerance_step))),
        "metrics_doc": "docs/sliding_peak_tolerance_metrics.md",
        "summary_peak_threshold": float(peak_th),
        "n_predicted_peaks_at_summary_threshold": int(first_pred.shape[0]) if first_pred is not None else 0,
        "per_peak_threshold": per_peak_threshold_global,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    viz_info: Dict[str, Any] = {}
    if args.viz_region_index is not None and first_region_index != int(args.viz_region_index):
        raise SystemExit(
            f"No window matching --viz-region-index={int(args.viz_region_index)}; "
            f"see per_window_results in *_summary.json for evaluated region_index values"
        )
    if first_curve is not None and first_tss is not None and first_pred is not None and first_L is not None:
        viz_info = _plot_random_window_validation(
            first_curve,
            first_tss,
            first_pred,
            chrom=first_region_chrom or target_chroms[0],
            contig=first_region_contig or CHR_TO_CONTIG.get(target_chroms[0], target_chroms[0]),
            L=first_L,
            win_len=win_len,
            stride=int(args.stride),
            out_path=args.output,
        )
        viz_info["region_index"] = first_region_index
        if args.viz_region_index is not None:
            viz_info["note"] = f"Visualization region_index={int(args.viz_region_index)} as specified"
            print(f"Saved window visualization (region_index={int(args.viz_region_index)}): {args.output}")
        else:
            viz_info["note"] = "In multi-window mode, defaults to visualizing the first window; set --viz-region-index to choose"
            print(f"Saved window visualization (default first window, region_index={first_region_index}): {args.output}")

    if args.save_npz:
        if len(regions) > 1:
            print("Note: multi-window mode does not save --save-npz (window lengths may differ; files are large)")
        else:
            npz_path = _resolve_abs_path(PROJECT_ROOT, args.save_npz)
            os.makedirs(os.path.dirname(os.path.abspath(npz_path)) or ".", exist_ok=True)
            x = np.arange(first_L, dtype=np.int32)
            np.savez_compressed(
                npz_path,
                pred=first_curve,
                positions=x,
                window_length_bp=np.array([win_len], dtype=np.int32),
                genomic_start_0based=np.array([int(regions[0]["g_start"])], dtype=np.int64),
            )
            print(f"Saved npz: {npz_path}")

    metrics_path = os.path.splitext(args.output)[0] + "_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({**meta, **metrics}, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics: {metrics_path}")

    summary_path = os.path.splitext(args.output)[0] + "_summary.json"
    summary_windows: List[Dict[str, Any]] = []
    for m in per_region_metrics:
        scan_rows = m.get("per_peak_threshold", [])
        per_th_peaks: List[Dict[str, Any]] = [
            {
                "peak_threshold": float(row["peak_threshold"]),
                "n_predicted_peaks": int(row["n_predicted_peaks"]),
                "predicted_tss_relative_positions_0based": [
                    int(x) for x in row.get("predicted_tss_relative_positions_0based", [])
                ],
            }
            for row in scan_rows
        ]
        summary_windows.append(
            {
                "chromosome": str(m.get("chromosome", "")),
                "refseq_contig": str(m.get("refseq_contig", "")),
                "region_index": int(m["region_index"]),
                "genomic_start_0based": int(m["genomic_start_0based"]),
                "genomic_end_0based_exclusive": int(m["genomic_end_0based_exclusive"]),
                "n_true_tss": int(m["n_true_tss"]),
                "true_tss_relative_positions_0based": [
                    int(x) for x in m.get("true_tss_relative_positions_0based", [])
                ],
                "predicted_peaks_by_peak_threshold_0p1_to_0p9": per_th_peaks,
            }
        )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **meta,
                "output_png": os.path.abspath(args.output),
                "metrics_file": os.path.abspath(metrics_path),
                "visualization": viz_info if viz_info else None,
                "n_regions_evaluated": int(len(per_region_metrics)),
                "per_window_results": summary_windows,
                "first_region_tss_relative_0based_first10": []
                if first_tss is None
                else [int(t) for t in first_tss[:10].tolist()],
                "first_region_predicted_tss_relative_0based_first10": []
                if first_pred is None
                else [int(t) for t in first_pred[:10].tolist()],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
