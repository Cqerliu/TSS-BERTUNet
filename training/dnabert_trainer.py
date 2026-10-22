#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DNABERT trainer.
Custom trainer adapted to the DNABERT model and data format.
"""

import os
import re
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
from .metrics import compute_metrics
from .focal_loss import FocalLoss, FocalLossWithRegression


class DNABERTTrainer:
    """DNABERT trainer (adapted to the custom data format)."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer: torch.optim.Optimizer,
        loss_config: dict,
        device: str = 'cpu',
        max_grad_norm: float = 1.0,
        dataset_name: str = None
    ):
        """
        Args:
            model: Model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            loss_config: Loss config dict with threshold, use_negative_samples,
                         focal_alpha, focal_gamma, use_regression_in_focal, etc.
            device: Device
            max_grad_norm: Max gradient norm for clipping
            dataset_name: Dataset name used for visualization subfolder naming
                          (e.g. promoter_labeled_gaussian_100)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_config = loss_config
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        # Initialize loss: see training/focal_loss.py
        threshold = loss_config.get('threshold', 0.1)
        focal_alpha = loss_config.get('focal_alpha', 0.25)
        focal_gamma = loss_config.get('focal_gamma', 2.0)
        use_regression_in_focal = loss_config.get('use_regression_in_focal', True)
        regression_weight = loss_config.get('regression_weight', 0.5)
        use_l1 = loss_config.get('use_l1', True)
        if use_regression_in_focal:
            self.loss_fn = FocalLossWithRegression(
                alpha=focal_alpha,
                gamma=focal_gamma,
                threshold=threshold,
                regression_weight=regression_weight,
                use_l1=use_l1,
                reduction='mean',
            )
        else:
            self.loss_fn = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                threshold=threshold,
                reduction='mean',
            )
        
        # Visualization: save under dataset-named subfolders
        name = (dataset_name or 'training').strip()
        name = re.sub(r'[<>:"/\\|?*]', '_', name)  # Strip illegal path characters
        if not name:
            name = 'training'
        self.visualization_dir = os.path.join('visualizations', 'training', name)
        os.makedirs(self.visualization_dir, exist_ok=True)
    
    def compute_masked_loss(self, predictions, targets, attention_mask=None):
        """
        Compute masked loss, ignoring padding positions.
        
        Args:
            predictions: (B, 1, L_base) predictions (base level)
            targets: (B, 1, L_base) targets (base level)
            attention_mask: (B, L_token) or (B, L_base) attention mask;
                           1 = valid, 0 = padding.
                           If None, use non-zero target positions as the mask.
        
        Returns:
            (total_loss, loss_dict)
        """
        # Unify prediction/target dimensions
        if predictions.ndim == 3:
            predictions = predictions.squeeze(1)  # (B, L_base)
        if targets.ndim == 3:
            targets = targets.squeeze(1)  # (B, L_base)
        
        # Ensure same length
        min_len = min(predictions.shape[1], targets.shape[1])
        predictions = predictions[:, :min_len]
        targets = targets[:, :min_len]
        
        # Build mask: use attention_mask if provided, otherwise non-zero targets
        if attention_mask is not None:
            # If attention_mask is token-level, it would need expansion to base level;
            # here we assume the model already handled masking, so match shapes when possible
            if attention_mask.shape[1] == predictions.shape[1]:
                # attention_mask is already base-level
                mask = attention_mask.float()  # (B, L_base)
            else:
                # attention_mask is token-level while predictions are base-level;
                # fall back to non-zero target positions
                mask = (targets.sum(dim=0) > 0).float().unsqueeze(0).expand(targets.shape[0], -1)
        else:
            # Use all positions as valid (simple approach)
            mask = torch.ones_like(targets)
        
        total_loss = self.loss_fn(predictions, targets, mask=mask)
        loss_dict = {'total_loss': float(total_loss.detach().item())}
        return total_loss, loss_dict
    
    def train_epoch(self):
        """
        Train for one epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Get offset_mapping, original_seq_length, and original_sequence if present
            offset_mapping = batch.get('offset_mapping', None)
            original_seq_length = batch.get('original_seq_length', None)
            original_sequence = batch.get('original_sequence', None)
            
            # If offset_mapping is a list, move tensors to device
            if offset_mapping is not None:
                offset_mapping = [om.to(self.device) if isinstance(om, torch.Tensor) else om for om in offset_mapping]
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target=labels,
                offset_mapping=offset_mapping,
                original_seq_length=original_seq_length,
                original_sequence=original_sequence
            )
            
            # Handle multi-output
            if isinstance(outputs, tuple):
                predictions = outputs[0]
            else:
                predictions = outputs
            
            # Compute loss
            # Note: model outputs and labels are both base-level
            loss, loss_dict = self.compute_masked_loss(
                predictions, labels, 
                attention_mask=None
            )
            
            loss.backward()
            
            # Gradient clipping (prevent explosion)
            if hasattr(self, 'max_grad_norm') and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self):
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        all_attention_masks = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Get offset_mapping, original_seq_length, and original_sequence if present
                offset_mapping = batch.get('offset_mapping', None)
                original_seq_length = batch.get('original_seq_length', None)
                original_sequence = batch.get('original_sequence', None)
                
                # If offset_mapping is a list, move tensors to device
                if offset_mapping is not None:
                    offset_mapping = [om.to(self.device) if isinstance(om, torch.Tensor) else om for om in offset_mapping]
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target=labels,
                    offset_mapping=offset_mapping,
                    original_seq_length=original_seq_length,
                    original_sequence=original_sequence
                )
                
                # Handle multi-output
                if isinstance(outputs, tuple):
                    predictions = outputs[0]
                else:
                    predictions = outputs
                
                # Masked loss
                # Note: model outputs and labels are both base-level
                loss, _ = self.compute_masked_loss(
                    predictions, labels, 
                    attention_mask=None
                )
                total_loss += loss.item()
                num_batches += 1
                
                # Build base-level mask for visualization
                # Model output is already base-level; create a matching mask
                if offset_mapping is not None and original_seq_length is not None:
                    # Use the model's _expand_mask_to_bases_with_offset helper
                    base_mask = self.model._expand_mask_to_bases_with_offset(
                        attention_mask, offset_mapping, original_seq_length
                    )
                else:
                    # Without offset_mapping, assume attention_mask is already base-level
                    base_mask = attention_mask
                
                all_predictions.append(predictions.cpu())
                all_targets.append(labels.cpu())
                all_attention_masks.append(base_mask.cpu())
        
        avg_loss = total_loss / num_batches
        
        # Handle inconsistent batch lengths: pad to a common length before concat
        if len(all_predictions) > 0:
            # Max length across batches
            max_pred_len = max(pred.shape[-1] for pred in all_predictions)
            max_target_len = max(tgt.shape[-1] for tgt in all_targets)
            max_mask_len = max(mask.shape[-1] for mask in all_attention_masks)
            max_len = max(max_pred_len, max_target_len, max_mask_len)
            
            # Pad all tensors to the same length
            padded_predictions = []
            padded_targets = []
            padded_masks = []
            
            for pred, tgt, mask in zip(all_predictions, all_targets, all_attention_masks):
                # Predictions: (B, 1, L) or (B, L)
                if pred.ndim == 3:
                    B, C, L = pred.shape
                    if L < max_len:
                        padding = torch.zeros(B, C, max_len - L, dtype=pred.dtype)
                        pred = torch.cat([pred, padding], dim=2)
                    padded_predictions.append(pred)
                else:
                    B, L = pred.shape
                    if L < max_len:
                        padding = torch.zeros(B, max_len - L, dtype=pred.dtype)
                        pred = torch.cat([pred, padding], dim=1)
                    padded_predictions.append(pred)
                
                # Targets: (B, 1, L) or (B, L)
                if tgt.ndim == 3:
                    B, C, L = tgt.shape
                    if L < max_len:
                        padding = torch.zeros(B, C, max_len - L, dtype=tgt.dtype)
                        tgt = torch.cat([tgt, padding], dim=2)
                    padded_targets.append(tgt)
                else:
                    B, L = tgt.shape
                    if L < max_len:
                        padding = torch.zeros(B, max_len - L, dtype=tgt.dtype)
                        tgt = torch.cat([tgt, padding], dim=1)
                    padded_targets.append(tgt)
                
                # Attention masks: (B, L)
                B, L = mask.shape
                if L < max_len:
                    padding = torch.zeros(B, max_len - L, dtype=mask.dtype)
                    mask = torch.cat([mask, padding], dim=1)
                padded_masks.append(mask)
            
            predictions = torch.cat(padded_predictions, dim=0)
            targets = torch.cat(padded_targets, dim=0)
            attention_masks = torch.cat(padded_masks, dim=0)
        else:
            # No predictions; return empty tensors
            predictions = torch.empty(0)
            targets = torch.empty(0)
            attention_masks = torch.empty(0)
        
        # Get threshold and use_negative_samples
        threshold = self.loss_config.get('threshold', 0.1)
        use_negative_samples = self.loss_config.get('use_negative_samples', True)
        
        peak_tolerance_bps = self.loss_config.get('peak_tolerance_bps')
        metrics = compute_metrics(
            predictions,
            targets,
            threshold=threshold,
            use_negative_samples=use_negative_samples,
            peak_tolerance_bps=peak_tolerance_bps,
        )
        
        return avg_loss, metrics, predictions, targets, attention_masks
    
    def visualize_predictions(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        attention_masks: torch.Tensor = None,
        epoch: int = 0,
        n_samples: int = 10
    ):
        """
        Visualize predictions vs ground-truth labels.
        
        Args:
            predictions: Predictions, shape=(batch, 1, length) or (batch, length)
            targets: Targets, shape=(batch, 1, length) or (batch, length)
            attention_masks: Attention masks, shape=(batch, length), used to hide padding
            epoch: Current epoch index
            n_samples: Number of samples to visualize
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if attention_masks is not None and isinstance(attention_masks, torch.Tensor):
            attention_masks = attention_masks.detach().cpu().numpy()
        
        # Unify dimensions
        if predictions.ndim == 3:
            predictions = predictions.squeeze(1)  # (B, L_base)
        if targets.ndim == 3:
            targets = targets.squeeze(1)  # (B, L_base)
        
        # Align lengths (take the minimum)
        min_len = min(predictions.shape[1], targets.shape[1])
        predictions = predictions[:, :min_len]
        targets = targets[:, :min_len]
        
        # If attention_mask is present, set padding predictions to 0 (not 0.5)
        # Note: the model already applies the mask, so padding should already be 0;
        # re-applying here ensures visualization shows only valid positions
        if attention_masks is not None:
            # Ensure attention_masks and predictions shapes match
            if attention_masks.shape[1] >= min_len:
                # attention_masks: (batch, length) - base level
                # Zero out padding positions (mask=0)
                predictions = predictions * attention_masks[:, :min_len]
            else:
                # Shape mismatch: attention_masks may be token-level;
                # the model already handled masking, so use predictions as-is
                print(f"  Warning: attention_masks shape ({attention_masks.shape}) does not match "
                      f"predictions shape ({predictions.shape}); skipping mask application")
        
        # Debug: check if all predictions are zero
        if np.all(predictions == 0):
            print(f"  Warning: all predictions are 0!")
            print(f"  predictions shape: {predictions.shape}")
            print(f"  predictions max: {np.max(predictions)}")
            print(f"  predictions min: {np.min(predictions)}")
            print(f"  targets max: {np.max(targets)}")
            print(f"  targets min: {np.min(targets)}")
        
        batch_size = predictions.shape[0]
        # Fixed grid: 3 rows x 3 cols = 9 sample subplots
        n_cols = 3
        n_rows = 3
        max_samples = n_rows * n_cols
        n_samples = min(n_samples, batch_size, max_samples)
        
        # Randomly select samples
        indices = np.random.choice(batch_size, size=n_samples, replace=False)
        
        # Create grid figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
        
        # Handle single-row case
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        # Get threshold
        threshold = self.loss_config.get('threshold', 0.1)
        
        for i, idx in enumerate(indices):
            ax = axes[i]
            pred_signal = predictions[idx]
            target_signal = targets[idx]
            # Fix y-axis to [0, 1] without clipping the data itself
            x = np.arange(len(pred_signal))
            
            # Ground-truth labels (red)
            ax.plot(x, target_signal, 'r-', label='True Label', linewidth=1.5, alpha=0.7)
            
            # Predicted labels (blue)
            ax.plot(x, pred_signal, 'b-', label='Predicted Label', linewidth=1.5, alpha=0.7)
            
            # Basic metrics
            mae = np.mean(np.abs(pred_signal - target_signal))
            mse = np.mean((pred_signal - target_signal) ** 2)

            # Positive/negative annotation: based on whether max true label exceeds threshold
            # (consistent with metrics threshold logic)
            is_positive_true = float(np.max(target_signal)) >= float(threshold)
            is_positive_pred = float(np.max(pred_signal)) >= float(threshold)
            true_pos_neg_text = "Positive" if is_positive_true else "Negative"
            pred_pos_neg_text = "Positive" if is_positive_pred else "Negative"
            
            # Peak positions (argmax of each signal)
            target_peak = np.argmax(target_signal)
            pred_peak = np.argmax(pred_signal)
            
            # Position accuracy at several tolerances
            accuracy_info = []
            for tolerance in [1, 5, 10, 20, 40, 80, 100]:
                if abs(pred_peak - target_peak) <= tolerance:
                    acc = 1.0
                else:
                    acc = 0.0
                accuracy_info.append(f'±{tolerance}bp: {acc:.2f}')
            
            # Peak error
            avg_peak_error = abs(pred_peak - target_peak)
            
            # Info text box
            info_text = f'Sample #{idx}\n'
            info_text += f'True Label: {true_pos_neg_text}\n'
            info_text += f'Pred Label: {pred_pos_neg_text}\n'
            info_text += f'MAE: {mae:.4f}\n'
            info_text += f'MSE: {mse:.4f}\n'
            info_text += f'Peak Error: {avg_peak_error:.1f}bp\n'
            info_text += f'True Peak: {target_peak}\n'
            info_text += f'Pred Peak: {pred_peak}\n'
            if accuracy_info:
                info_text += '\n'.join(accuracy_info)
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Mark true TSS peak with a red downward triangle
            target_peak_value = target_signal[target_peak]
            ax.plot(target_peak, target_peak_value, 'rv', markersize=10, 
                   markerfacecolor='red', markeredgecolor='darkred', 
                   markeredgewidth=1.5, label='True TSS', 
                   zorder=5)
            
            # Mark predicted TSS peak with an upward triangle
            pred_peak_value = pred_signal[pred_peak]
            
            # Within true-peak tolerance (±10bp)?
            is_correct = abs(pred_peak - target_peak) <= 10
            
            # Color by correctness: green if correct, blue otherwise
            if is_correct:
                ax.plot(pred_peak, pred_peak_value, '^g', markersize=10,
                       markerfacecolor='green', markeredgecolor='darkgreen',
                       markeredgewidth=1.5, label='Pred TSS (Correct)',
                       zorder=5)
            else:
                ax.plot(pred_peak, pred_peak_value, '^b', markersize=10,
                       markerfacecolor='blue', markeredgecolor='darkblue',
                       markeredgewidth=1.5, label='Pred TSS (Incorrect)',
                       zorder=5)
            
            # Labels and title
            ax.set_xlabel('Position (bp)', fontsize=10)
            ax.set_ylabel('TSS Probability', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, len(pred_signal))
            ax.set_ylim(0.0, 1.0)
        
        # Hide unused subplots
        for i in range(len(indices), len(axes)):
            axes[i].axis('off')
        
        # Do not show a global figure title (e.g. Epoch info)
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.visualization_dir, f'epoch_{epoch+1:03d}_dnabert_predictions.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Visualization saved: {save_path}")
    
    def fit(
        self,
        epochs,
        early_stopping=None,
        checkpoint=None,
        lr_scheduler=None,
        early_stopping_monitor='val_loss',
    ):
        """Train the model.

        Args:
            early_stopping_monitor: Metric used for early stopping. 'val_loss' uses
                validation loss; otherwise a key from the metrics dict returned by
                validate() (e.g. 'classification_accuracy').
        """
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate (masked loss is already used inside validate)
            val_loss, metrics, val_predictions, val_targets, val_attention_masks = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(metrics)
            
            print(f"  Train loss: {train_loss:.4f}")
            print(f"  Val loss: {val_loss:.4f}")
            print(f"  Val metrics: {metrics}")
            
            # Visualize predictions after each epoch
            print(f"\nGenerating visualizations...")
            self.visualize_predictions(val_predictions, val_targets, val_attention_masks, epoch, n_samples=9)
            
            if checkpoint:
                ckpt_monitor = getattr(checkpoint, 'monitor', 'val_loss')
                if ckpt_monitor in (None, 'val_loss'):
                    ckpt_score = val_loss
                else:
                    if ckpt_monitor not in metrics:
                        raise KeyError(
                            f"checkpoint.monitor={ckpt_monitor!r} not in validation metrics; "
                            f"available keys: {list(metrics.keys())}"
                        )
                    ckpt_score = float(metrics[ckpt_monitor])
                checkpoint(self.model, ckpt_score, epoch)

            # Early stopping: decoupled from checkpoint / lr_scheduler by default;
            # can monitor a specific metrics key
            if early_stopping:
                if early_stopping_monitor in (None, 'val_loss'):
                    es_score = val_loss
                    es_name = 'val_loss'
                else:
                    if early_stopping_monitor not in metrics:
                        raise KeyError(
                            f"early_stopping_monitor={early_stopping_monitor!r} not in validation metrics; "
                            f"available keys: {list(metrics.keys())}"
                        )
                    es_score = float(metrics[early_stopping_monitor])
                    es_name = early_stopping_monitor
                if early_stopping(es_score):
                    print(f"  Early stopping triggered after {epoch+1} epochs ({es_name} no longer improving)")
                    break
            
            # Learning Rate Scheduler
            if lr_scheduler:
                lr_scheduler(val_loss)
            
            print("-" * 60)
        
        return self.history
