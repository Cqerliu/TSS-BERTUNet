#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load a trained model and reproduce the test-set split with the same data/random seed
as training for inference evaluation.

Output metrics:
- classification_accuracy
- classification_precision
- classification_recall
- classification_f1
- classification_auc_roc
- classification_tp
- classification_fp
- classification_tn
- classification_fn
- positive_mae
- positive_pearson
- peak_accuracy_1bp
- peak_accuracy_50bp
- peak_accuracy_100bp
- peak_accuracy_150bp
- peak_accuracy_200bp
- peak_accuracy_250bp
- peak_accuracy_300bp
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data import create_dataloader, load_data, split_data  # noqa: E402
from model import DNABERTForTSSLocalization  # noqa: E402
from training.metrics import ClassificationMetrics, SeparatedRegressionMetrics  # noqa: E402
from utils import load_training_config, set_seed  # noqa: E402


def _resolve_abs_path(path_str: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(PROJECT_ROOT, path_str)


def _build_model(checkpoint_path: str, config: Dict[str, Any], device: str) -> torch.nn.Module:
    dnabert_cfg = config.get("dnabert", {})
    reg_cfg = config.get("regression_head", {})
    use_only_unet = config.get("use_only_unet", False)

    model_path = _resolve_abs_path(dnabert_cfg.get("model_path", "DNABERT-2-117M"))
    model = DNABERTForTSSLocalization(
        dnabert_path=model_path,
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
    model.eval()
    return model


def _run_inference(
    model: torch.nn.Module,
    dataloader,
    device: str,
    *,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_predictions: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    iterator = tqdm(dataloader, desc="Test infer") if show_progress else dataloader
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


def _peak_accuracy_at_tolerance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    tolerance: int,
) -> float:
    if prediction.ndim == 2:
        prediction = prediction.unsqueeze(1)
    if target.ndim == 2:
        target = target.unsqueeze(1)

    target_max = target.max(dim=-1)[0]
    if target_max.ndim > 1:
        target_max = target_max.squeeze(1)
    is_positive = target_max >= threshold
    if is_positive.sum().item() == 0:
        return 0.0

    pos_pred = prediction[is_positive]
    pos_target = target[is_positive]

    if pos_pred.ndim == 3:
        pred_peak = pos_pred.argmax(dim=-1).squeeze(1)
    else:
        pred_peak = pos_pred.argmax(dim=-1)
    if pos_target.ndim == 3:
        target_peak = pos_target.argmax(dim=-1).squeeze(1)
    else:
        target_peak = pos_target.argmax(dim=-1)

    err = torch.abs(pred_peak - target_peak)
    return float((err <= int(tolerance)).float().mean().item())


def _build_default_checkpoint_path(merged_csv_path: str) -> str:
    csv_stem = os.path.splitext(os.path.basename(merged_csv_path))[0] if merged_csv_path else "training"
    if csv_stem.startswith("promoter_labeled_"):
        ckpt_id = csv_stem[len("promoter_labeled_") :]
    else:
        ckpt_id = csv_stem
    ckpt_id = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in ckpt_id)
    return os.path.join(PROJECT_ROOT, "checkpoints", f"best_{ckpt_id}.pth")


def _build_result_key(checkpoint_path: str, merged_csv_path: str) -> str:
    # Prefer extracting gaussian_xxx from the checkpoint name, e.g. best_gaussian_200_11 -> gaussian_200
    ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    m = re.search(r"(gaussian_\d+)", ckpt_stem)
    if m:
        return m.group(1)

    # Fallback: from data filename promoter_labeled_xxx_11 -> xxx
    csv_stem = os.path.splitext(os.path.basename(merged_csv_path))[0]
    if csv_stem.startswith("promoter_labeled_"):
        tail = csv_stem[len("promoter_labeled_") :]
        parts = tail.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            tail = "_".join(parts[:-1])
        return tail
    return ckpt_stem


def _append_payload_to_json(output_path: str, payload: Dict[str, Any], result_key: str) -> str:
    """
    Append in key-value form following the validation_results.json structure:
    - File missing: write {result_key: payload}
    - File is a dict: append result_key at the end (auto timestamp suffix on collision)
    - File is a list (legacy format): convert to dict then append
    - File corrupt / non-JSON: keep original text as .bak, then rebuild as dict
    """
    append_item = dict(payload)
    append_item["inference_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.exists(output_path):
        data_to_save: Dict[str, Any] = {result_key: append_item}
    else:
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            if isinstance(old_data, dict):
                data_to_save = old_data
            elif isinstance(old_data, list):
                # Compat with prior list-append format; keep old data when possible
                data_to_save = {}
                for idx, item in enumerate(old_data, 1):
                    if isinstance(item, dict):
                        data_to_save[f"legacy_{idx}"] = item
                    else:
                        data_to_save[f"legacy_{idx}"] = {"value": item}
            else:
                data_to_save = {}
        except Exception:
            # File may not be valid JSON; back up then rebuild
            backup_path = f"{output_path}.bak"
            with open(output_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            with open(backup_path, "w", encoding="utf-8") as bf:
                bf.write(raw_text)
            data_to_save = {}

    final_key = result_key
    if final_key in data_to_save:
        final_key = f"{result_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data_to_save[final_key] = append_item

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    return final_key


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Load a trained model and evaluate inference on the same test split as training"
    )
    ap.add_argument("--config", type=str, default="config/training_config.json", help="Training config file path")
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_triangle_600_11.pth",
        help="Model checkpoint; if unset, inferred from the training naming rule",
    )
    ap.add_argument("--output", type=str, default="validation_results.json", help="Output metrics JSON path")
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu")
    ap.add_argument("--no-progress", action="store_true", help="Disable inference progress bar")
    args = ap.parse_args()

    config = load_training_config(_resolve_abs_path(args.config))
    default_threshold = float(config["default_threshold"])
    use_negative_samples = bool(config.get("use_negative_samples", True))

    data_cfg = config.get("data", {})
    merged_csv_path = _resolve_abs_path(data_cfg.get("merged_csv_path", "data/dataset/merged_dataset.csv"))
    train_ratio = float(data_cfg.get("train_ratio", 0.8))
    val_ratio = float(data_cfg.get("val_ratio", 0.1))
    test_ratio = float(data_cfg.get("test_ratio", 0.1))
    random_seed = int(data_cfg.get("random_seed", 42))

    dnabert_cfg = config.get("dnabert", {})
    max_seq_length = int(dnabert_cfg.get("max_seq_length", 512))
    batch_size = int(config.get("training", {}).get("batch_size", 16))

    checkpoint_path = _resolve_abs_path(args.checkpoint) if args.checkpoint else _build_default_checkpoint_path(merged_csv_path)
    if not os.path.isfile(checkpoint_path):
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")

    if args.output:
        output_path = _resolve_abs_path(args.output)
    else:
        ckpt_stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
        output_path = os.path.join(PROJECT_ROOT, "results", f"infer_{ckpt_stem}_test_metrics.json")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    sequences, labels, chromosomes = load_data(
        merged_csv_path=merged_csv_path,
        use_negative_samples=use_negative_samples,
        max_seq_length=max_seq_length,
    )

    # Same as train_dnabert.py: keep only chr1-chr21
    allowed_chroms = {f"chr{i}" for i in range(1, 22)}
    filtered_sequences: List[str] = []
    filtered_labels: List[np.ndarray] = []
    for seq, lab, chrom in zip(sequences, labels, chromosomes):
        if str(chrom) in allowed_chroms:
            filtered_sequences.append(seq)
            filtered_labels.append(lab)

    set_seed(random_seed)
    _train_data, _val_data, test_data = split_data(
        filtered_sequences,
        filtered_labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    model = _build_model(checkpoint_path, config, device)
    try:
        test_loader = create_dataloader(
            test_data[0],
            test_data[1],
            model.tokenizer,
            batch_size=batch_size,
            shuffle=False,
            max_length=max_seq_length,
        )
        predictions, targets = _run_inference(
            model,
            test_loader,
            device=device,
            show_progress=(not args.no_progress),
        )
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    cls_metrics = ClassificationMetrics.compute(predictions, targets, threshold=default_threshold)
    reg_metrics = SeparatedRegressionMetrics.compute(
        predictions,
        targets,
        threshold=default_threshold,
        compute_negative_regression=False,
    )

    out_metrics: Dict[str, Any] = {
        "classification_accuracy": float(cls_metrics.get("classification_accuracy", 0.0)),
        "classification_precision": float(cls_metrics.get("classification_precision", 0.0)),
        "classification_recall": float(cls_metrics.get("classification_recall", 0.0)),
        "classification_f1": float(cls_metrics.get("classification_f1", 0.0)),
        "classification_auc_roc": float(cls_metrics.get("classification_auc_roc", 0.0)),
        "classification_tp": int(cls_metrics.get("classification_tp", 0)),
        "classification_fp": int(cls_metrics.get("classification_fp", 0)),
        "classification_tn": int(cls_metrics.get("classification_tn", 0)),
        "classification_fn": int(cls_metrics.get("classification_fn", 0)),
        "positive_mae": float(reg_metrics.get("positive_mae", 0.0)),
        "positive_pearson": float(reg_metrics.get("positive_pearson", 0.0)),
    }

    for tol in (1, 50, 100, 150, 200, 250, 300):
        out_metrics[f"peak_accuracy_{tol}bp"] = _peak_accuracy_at_tolerance(
            predictions,
            targets,
            threshold=default_threshold,
            tolerance=tol,
        )

    payload: Dict[str, Any] = {
        "config_file": os.path.abspath(_resolve_abs_path(args.config)),
        "checkpoint": os.path.abspath(checkpoint_path),
        "merged_csv_path": os.path.abspath(merged_csv_path),
        "random_seed": random_seed,
        "data_split_ratio": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "test_size": len(test_data[0]),
        "threshold": default_threshold,
        "metrics": out_metrics,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    result_key = _build_result_key(checkpoint_path, merged_csv_path)
    saved_key = _append_payload_to_json(output_path, payload, result_key)

    print(f"Inference done; metrics appended to: {output_path}")
    print(f"Saved key: {saved_key}")
    print(json.dumps(out_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
