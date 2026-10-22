#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation metrics module
"""

import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from typing import Dict, Tuple, Optional
import torch.nn.functional as F

# Loss functions are no longer imported here


class MAE:
    """Mean Absolute Error."""
    
    @staticmethod
    def compute(prediction: torch.Tensor, target: torch.Tensor) -> float:
        """
        Args:
            prediction: Predictions
            target: Ground truth
        Returns:
            Mean absolute error
        """
        return F.l1_loss(prediction, target).item()


class MSE:
    """Mean Squared Error."""
    
    @staticmethod
    def compute(prediction: torch.Tensor, target: torch.Tensor) -> float:
        """
        Args:
            prediction: Predictions
            target: Ground truth
        Returns:
            Mean squared error
        """
        return F.mse_loss(prediction, target).item()


class PearsonCorrelation:
    """Pearson correlation coefficient."""
    
    @staticmethod
    def compute(prediction: torch.Tensor, target: torch.Tensor) -> float:
        """
        Args:
            prediction: Predictions
            target: Ground truth
        Returns:
            Pearson correlation coefficient
        """
        # Flatten tensors
        pred_flat = prediction.detach().cpu().numpy().flatten()
        target_flat = target.detach().cpu().numpy().flatten()
        
        # Compute correlation
        corr, _ = pearsonr(pred_flat, target_flat)
        
        return float(corr)


class SpearmanCorrelation:
    """Spearman correlation coefficient."""
    
    @staticmethod
    def compute(prediction: torch.Tensor, target: torch.Tensor) -> float:
        """
        Args:
            prediction: Predictions
            target: Ground truth
        Returns:
            Spearman correlation coefficient
        """
        # Flatten tensors
        pred_flat = prediction.detach().cpu().numpy().flatten()
        target_flat = target.detach().cpu().numpy().flatten()
        
        # Compute correlation
        corr, _ = spearmanr(pred_flat, target_flat)
        
        return float(corr)


class ClassificationMetrics:
    """Classification metrics: whether a sequence contains a TSS."""
    
    @staticmethod
    def compute(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.1) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            prediction: Predictions (batch, 1, length) or (batch, length)
            target: Ground truth (batch, 1, length) or (batch, length)
            threshold: Threshold for deciding whether a TSS is present
        
        Returns:
            Classification metrics dictionary
        """
        # Handle different input shapes
        if prediction.ndim == 2:
            prediction = prediction.unsqueeze(1)
        if target.ndim == 2:
            target = target.unsqueeze(1)
        
        # Max value (presence of TSS)
        pred_max = prediction.max(dim=-1)[0]  # (batch, 1) or (batch,)
        target_max = target.max(dim=-1)[0]  # (batch, 1) or (batch,)
        
        # Ensure 1D tensors
        if pred_max.ndim > 1:
            pred_max = pred_max.squeeze(1)  # (batch,)
        if target_max.ndim > 1:
            target_max = target_max.squeeze(1)  # (batch,)
        
        # Binary labels
        pred_binary = (pred_max >= threshold).float()
        target_binary = (target_max >= threshold).float()
        
        # TP, FP, TN, FN
        TP = (pred_binary * target_binary).sum().item()
        FP = (pred_binary * (1 - target_binary)).sum().item()
        TN = ((1 - pred_binary) * (1 - target_binary)).sum().item()
        FN = ((1 - pred_binary) * target_binary).sum().item()
        
        # Metrics
        total = TP + FP + TN + FN
        accuracy = (TP + TN) / total if total > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # AUC-ROC
        try:
            pred_max_np = pred_max.detach().cpu().numpy()
            target_binary_np = target_binary.detach().cpu().numpy()
            if len(np.unique(target_binary_np)) > 1:  # Need at least two classes
                auc_roc = roc_auc_score(target_binary_np, pred_max_np)
            else:
                auc_roc = 0.0
        except:
            auc_roc = 0.0
        
        return {
            'classification_accuracy': accuracy,
            'classification_precision': precision,
            'classification_recall': recall,
            'classification_f1': f1,
            'classification_auc_roc': auc_roc,
            'classification_tp': TP,
            'classification_fp': FP,
            'classification_tn': TN,
            'classification_fn': FN
        }


class SeparatedRegressionMetrics:
    """Class-separated regression metrics: positives and negatives computed separately."""
    
    @staticmethod
    def compute(
        prediction: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.1,
        compute_negative_regression: bool = False
    ) -> Dict[str, float]:
        """
        Compute class-separated regression metrics.
        
        Args:
            prediction: Predictions (batch, 1, length) or (batch, length)
            target: Ground truth (batch, 1, length) or (batch, length)
            threshold: Threshold for deciding whether a TSS is present
            compute_negative_regression: Whether to compute negative-sample regression metrics (mae/mse, etc.)
        
        Returns:
            Class-separated regression metrics dictionary
        """
        # Handle different input shapes
        if prediction.ndim == 2:
            prediction = prediction.unsqueeze(1)
        if target.ndim == 2:
            target = target.unsqueeze(1)
        
        target_max = target.max(dim=-1)[0]  # (batch, 1) or (batch,)
        if target_max.ndim > 1:
            target_max = target_max.squeeze(1)  # (batch,)
        is_positive = (target_max >= threshold)
        
        metrics = {}
        
        # Positive metrics
        if is_positive.sum() > 0:
            pos_pred = prediction[is_positive]
            pos_target = target[is_positive]
            
            metrics['positive_mae'] = MAE.compute(pos_pred, pos_target)
            metrics['positive_mse'] = MSE.compute(pos_pred, pos_target)
            
            try:
                metrics['positive_pearson'] = PearsonCorrelation.compute(pos_pred, pos_target)
            except:
                metrics['positive_pearson'] = 0.0
            
            try:
                metrics['positive_spearman'] = SpearmanCorrelation.compute(pos_pred, pos_target)
            except:
                metrics['positive_spearman'] = 0.0
            
            metrics['positive_count'] = is_positive.sum().item()
        else:
            metrics['positive_mae'] = 0.0
            metrics['positive_mse'] = 0.0
            metrics['positive_pearson'] = 0.0
            metrics['positive_spearman'] = 0.0
            metrics['positive_count'] = 0
        
        # Negative metrics
        is_negative = ~is_positive
        metrics['negative_count'] = is_negative.sum().item()
        if compute_negative_regression and is_negative.sum() > 0:
            neg_pred = prediction[is_negative]
            neg_target = target[is_negative]
            metrics['negative_mae'] = MAE.compute(neg_pred, neg_target)
            metrics['negative_mse'] = MSE.compute(neg_pred, neg_target)
        
        return metrics


class PeakPositionCount:
    """Peak position accuracy counts (positives only)."""
    
    def __init__(self, tolerance: int = 1, threshold: float = 0.1):
        """
        Args:
            tolerance (int): Allowed error in bp
            threshold (float): Threshold for deciding whether a TSS is present
        """
        self.tolerance = tolerance
        self.threshold = threshold
    
    def compute(self, prediction: torch.Tensor, target: torch.Tensor) -> Tuple[int, int]:
        """
        Compute peak position accuracy (positives only).
        
        Args:
            prediction: Predictions (batch, 1, length) or (batch, length)
            target: Ground truth (batch, 1, length) or (batch, length)
        Returns:
            (correct_count, total_count)
        """
        # Handle different input shapes
        if prediction.ndim == 2:
            # (batch, length) -> (batch, 1, length)
            prediction = prediction.unsqueeze(1)
        if target.ndim == 2:
            # (batch, length) -> (batch, 1, length)
            target = target.unsqueeze(1)
        
        # Peak accuracy only on positives
        target_max = target.max(dim=-1)[0]  # (batch, 1)
        if target_max.ndim > 1:
            target_max = target_max.squeeze(1)  # (batch,)
        is_positive = (target_max >= self.threshold)
        
        if is_positive.sum() == 0:
            return 0, 0
        
        # Positives only
        pos_pred = prediction[is_positive]  # (n_positive, 1, length) or (n_positive, length)
        pos_target = target[is_positive]  # (n_positive, 1, length) or (n_positive, length)
        
        # Find peak positions
        # If pos_pred is (n_positive, 1, length), squeeze the middle dim or argmax on the last dim
        # If pos_pred is (n_positive, length), argmax on the last dim directly
        if pos_pred.ndim == 3:
            # (n_positive, 1, length) -> argmax on last dim
            pred_peak_pos = pos_pred.argmax(dim=-1)  # (n_positive, 1)
            pred_peak_pos = pred_peak_pos.squeeze(1)  # (n_positive,)
        else:
            # (n_positive, length) -> argmax on last dim
            pred_peak_pos = pos_pred.argmax(dim=-1)  # (n_positive,)
        
        if pos_target.ndim == 3:
            target_peak_pos = pos_target.argmax(dim=-1)  # (n_positive, 1)
            target_peak_pos = target_peak_pos.squeeze(1)  # (n_positive,)
        else:
            target_peak_pos = pos_target.argmax(dim=-1)  # (n_positive,)
        
        # Errors
        errors = torch.abs(pred_peak_pos - target_peak_pos)
        
        # Count within tolerance
        correct = (errors <= self.tolerance).sum().item()
        total = pos_pred.shape[0]
        
        return correct, total


def compute_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.1,
    use_negative_samples: bool = True,
    peak_tolerance_bps: Optional[list] = None,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        prediction: Predictions (batch, 1, length) or (batch, length)
        target: Ground truth (batch, 1, length) or (batch, length)
        threshold: Threshold for deciding whether a TSS is present
        use_negative_samples: Whether negative samples are used
    Returns:
        Metrics dictionary
    """
    use_negative = use_negative_samples
    
    metrics = {}
    
    if use_negative:
        # ========== Classification metrics (only when negatives are present) ==========
        classification_metrics = ClassificationMetrics.compute(prediction, target, threshold)
        metrics.update(classification_metrics)
        
        # ========== Class-separated regression: positives only ==========
        # No overall regression metrics; no negative regression errors either.
        separated_metrics = SeparatedRegressionMetrics.compute(
            prediction,
            target,
            threshold,
            compute_negative_regression=False
        )
        metrics.update(separated_metrics)
        
        # ========== Data statistics ==========
        total_samples = prediction.shape[0]
        metrics['total_samples'] = total_samples
        metrics['positive_samples'] = separated_metrics.get('positive_count', 0)
        metrics['negative_samples'] = separated_metrics.get('negative_count', 0)
        metrics['positive_ratio'] = (
            metrics['positive_samples'] / total_samples if total_samples > 0 else 0.0
        )
        # Drop redundant count fields (keep positive_samples / negative_samples only)
        metrics.pop('positive_count', None)
        metrics.pop('negative_count', None)
    else:
        # ========== No negatives: all samples are positives ==========
        total_samples = prediction.shape[0]
        metrics['total_samples'] = total_samples
        metrics['positive_samples'] = total_samples
        metrics['negative_samples'] = 0
        metrics['positive_ratio'] = 1.0
        
        # ========== Regression metrics: positives only ==========
        # Even when all samples are positives, do not emit overall mae/mse/pearson/spearman
        # to avoid duplicating positive_* fields.
        metrics['positive_mae'] = MAE.compute(prediction, target)
        metrics['positive_mse'] = MSE.compute(prediction, target)
        
        try:
            metrics['positive_pearson'] = PearsonCorrelation.compute(prediction, target)
        except:
            metrics['positive_pearson'] = 0.0
        
        try:
            metrics['positive_spearman'] = SpearmanCorrelation.compute(prediction, target)
        except:
            metrics['positive_spearman'] = 0.0
        
        # Without negatives, do not emit negative_count / positive_count
        # (use positive_samples / negative_samples only)

    # ========== Peak position accuracy (positives only; always computed) ==========
    # Note: peak position uses argmax of raw predictions, not thresholded binaries
    if peak_tolerance_bps is None:
        peak_tolerance_bps = [1, 5, 10, 20, 40, 80, 100]
    peak_tolerance_bps = sorted({int(t) for t in peak_tolerance_bps if int(t) >= 0})

    peak_total = 0
    for i, tolerance in enumerate(peak_tolerance_bps):
        peak_counter = PeakPositionCount(tolerance=tolerance, threshold=threshold)
        correct, total = peak_counter.compute(prediction, target)
        if i == 0:
            peak_total = total
            metrics['peak_total'] = peak_total
        metrics[f'peak_count_{tolerance}bp'] = correct
        metrics[f'peak_accuracy_{tolerance}bp'] = correct / total if total > 0 else 0.0
    
    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Print evaluation metrics.
    
    Args:
        metrics: Metrics dictionary
        prefix: Prefix string
    """
    print(f"\n{prefix}Evaluation metrics:")
    
    # Data statistics
    print(f"\n[Data statistics]")
    print(f"  Total samples: {metrics.get('total_samples', 0)}")
    print(f"  Positive samples: {metrics.get('positive_samples', 0)}")
    print(f"  Negative samples: {metrics.get('negative_samples', 0)}")
    print(f"  Positive ratio: {metrics.get('positive_ratio', 0.0):.2%}")
    
    # Classification metrics
    print(f"\n[Classification metrics]")
    print(f"  Accuracy: {metrics.get('classification_accuracy', 0.0):.4f}")
    print(f"  Precision: {metrics.get('classification_precision', 0.0):.4f}")
    print(f"  Recall: {metrics.get('classification_recall', 0.0):.4f}")
    print(f"  F1 score: {metrics.get('classification_f1', 0.0):.4f}")
    print(f"  AUC-ROC: {metrics.get('classification_auc_roc', 0.0):.4f}")
    print(f"  TP: {metrics.get('classification_tp', 0)}, FP: {metrics.get('classification_fp', 0)}, "
          f"TN: {metrics.get('classification_tn', 0)}, FN: {metrics.get('classification_fn', 0)}")
    
    # Positive regression metrics
    print(f"\n[Positive regression metrics]")
    print(f"  MAE: {metrics.get('positive_mae', 0.0):.4f}")
    print(f"  MSE: {metrics.get('positive_mse', 0.0):.4f}")
    print(f"  Pearson correlation: {metrics.get('positive_pearson', 0.0):.4f}")
    print(f"  Spearman correlation: {metrics.get('positive_spearman', 0.0):.4f}")
    
    # Negative regression metrics (if present)
    if 'negative_mae' in metrics or 'negative_mse' in metrics:
        print(f"\n[Negative regression metrics]")
        print(f"  MAE: {metrics.get('negative_mae', 0.0):.4f}")
        print(f"  MSE: {metrics.get('negative_mse', 0.0):.4f}")
    
    # Peak position accuracy (positives only)
    peak_total = metrics.get('peak_total', 0)
    if peak_total > 0:
        print(f"\n[Peak position accuracy (positives only, n={peak_total})]")
        tolerance_bps = sorted(
            int(k.replace('peak_accuracy_', '').replace('bp', ''))
            for k in metrics.keys()
            if isinstance(k, str) and k.startswith('peak_accuracy_') and k.endswith('bp')
        )
        for tol in tolerance_bps:
            print(f"  ±{tol}bp: {metrics.get(f'peak_count_{tol}bp', 0)}/{peak_total} "
                  f"({metrics.get(f'peak_accuracy_{tol}bp', 0.0):.2%})")
    else:
        print(f"\n[Peak position accuracy]")
        print(f"  No positive samples; peak position accuracy cannot be computed")
