#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Focal Loss implementation.
Used for class imbalance; assigns higher weight to hard examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    where:
    - p_t is the predicted probability
    - alpha is the class-balance factor
    - gamma is the focusing parameter
    
    For regression, convert predictions and targets to binary labels:
    - positions above the threshold are 1 (positive)
    - positions at or below the threshold are 0 (negative)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, threshold: float = 0.1, 
                 reduction: str = 'mean'):
        """
        Args:
            alpha: Class-balance factor for positives vs negatives (default 0.25)
            gamma: Focusing parameter to down-weight easy examples (default 2.0)
            threshold: Threshold to convert regression values to binary labels (default 0.1)
            reduction: Loss reduction, 'mean' or 'sum' (default 'mean')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.threshold = threshold
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            predictions: Predictions of shape (B, L) or (B, 1, L), typically in [0, 1]
            targets: Targets of shape (B, L) or (B, 1, L), typically in [0, 1]
            mask: Mask of shape (B, L); 1 = valid, 0 = padding (optional)
        
        Returns:
            loss: Focal Loss value
        """
        # Unify dimensions
        if predictions.ndim == 3:
            predictions = predictions.squeeze(1)  # (B, L)
        if targets.ndim == 3:
            targets = targets.squeeze(1)  # (B, L)
        
        # Ensure predictions and targets have the same length
        min_len = min(predictions.shape[1], targets.shape[1])
        predictions = predictions[:, :min_len]
        targets = targets[:, :min_len]
        
        # Convert regression values to binary labels
        # Above threshold -> 1 (positive), otherwise 0 (negative)
        pred_binary = (predictions > self.threshold).float()  # (B, L)
        target_binary = (targets > self.threshold).float()  # (B, L)
        
        # Predicted probabilities (sigmoid to keep in [0, 1] if needed)
        # If predictions are already in [0, 1], use them directly
        if predictions.min() < 0 or predictions.max() > 1:
            pred_probs = torch.sigmoid(predictions)  # (B, L)
        else:
            pred_probs = predictions  # (B, L)
        
        # p_t: for positives p_t = p; for negatives p_t = 1 - p
        p_t = pred_probs * target_binary + (1 - pred_probs) * (1 - target_binary)  # (B, L)
        
        # alpha_t: for positives alpha_t = alpha; for negatives alpha_t = 1 - alpha
        alpha_t = self.alpha * target_binary + (1 - self.alpha) * (1 - target_binary)  # (B, L)
        
        # Focal Loss
        # FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        # Add a small epsilon to avoid log(0)
        epsilon = 1e-8
        focal_weight = alpha_t * torch.pow(1 - p_t, self.gamma)  # (B, L)
        focal_loss = -focal_weight * torch.log(p_t + epsilon)  # (B, L)
        
        # Apply mask if provided
        if mask is not None:
            # Ensure mask length matches
            if mask.shape[1] >= min_len:
                mask = mask[:, :min_len]
            else:
                # Mask length mismatch; use an all-ones mask
                mask = torch.ones_like(predictions)
            focal_loss = focal_loss * mask
        
        # Reduce loss
        if self.reduction == 'mean':
            if mask is not None:
                loss = focal_loss.sum() / (mask.sum() + epsilon)
            else:
                loss = focal_loss.mean()
        elif self.reduction == 'sum':
            loss = focal_loss.sum()
        else:
            loss = focal_loss
        
        return loss


class FocalLossWithRegression(nn.Module):
    """
    Combined Focal Loss + regression loss.
    
    For TSS localization we jointly consider:
    1. Binary classification loss (Focal Loss): whether each position is a TSS
    2. Regression loss (L1/MSE): predicted TSS probability values
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, threshold: float = 0.1,
                 regression_weight: float = 0.5, use_l1: bool = True, reduction: str = 'mean'):
        """
        Args:
            alpha: Focal Loss class-balance factor
            gamma: Focal Loss focusing parameter
            threshold: Binary classification threshold
            regression_weight: Weight of the regression term (default 0.5)
            use_l1: Use L1 loss if True, otherwise MSE (default True)
            reduction: Loss reduction mode
        """
        super(FocalLossWithRegression, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, threshold=threshold, reduction=reduction)
        self.regression_weight = regression_weight
        self.use_l1 = use_l1
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the combined loss.
        
        Args:
            predictions: Predictions of shape (B, L) or (B, 1, L)
            targets: Targets of shape (B, L) or (B, 1, L)
            mask: Mask of shape (B, L) (optional)
        
        Returns:
            total_loss: Combined loss value
        """
        # Focal Loss (binary classification)
        focal_loss = self.focal_loss(predictions, targets, mask)
        
        # Regression loss
        if predictions.ndim == 3:
            predictions = predictions.squeeze(1)
        if targets.ndim == 3:
            targets = targets.squeeze(1)
        
        min_len = min(predictions.shape[1], targets.shape[1])
        predictions = predictions[:, :min_len]
        targets = targets[:, :min_len]
        
        if self.use_l1:
            regression_loss = F.l1_loss(predictions, targets, reduction='none')  # (B, L)
        else:
            regression_loss = F.mse_loss(predictions, targets, reduction='none')  # (B, L)
        
        # Apply mask
        if mask is not None:
            if mask.shape[1] >= min_len:
                mask = mask[:, :min_len]
            else:
                mask = torch.ones_like(predictions)
            regression_loss = regression_loss * mask
        
        # Reduce regression loss
        epsilon = 1e-8
        if self.reduction == 'mean':
            if mask is not None:
                regression_loss = regression_loss.sum() / (mask.sum() + epsilon)
            else:
                regression_loss = regression_loss.mean()
        elif self.reduction == 'sum':
            regression_loss = regression_loss.sum()
        
        # Combined loss
        total_loss = (1 - self.regression_weight) * focal_loss + self.regression_weight * regression_loss
        
        return total_loss
