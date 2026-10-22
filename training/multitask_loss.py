#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-branch joint loss (classification + regression + consistency).

Implements the dual-branch joint loss design (see docs/):
    L_total = lambda_cls * L_cls + lambda_reg * L_reg + lambda_cons * L_cons
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """
    Joint loss for sample-level classification + base-level regression.

    Expected inputs:
    - cls_logits: classification logits of shape (B,) or (B, 1)
    - reg_predictions: regression predictions of shape (B, L) or (B, 1, L)
    - reg_targets: regression targets of shape (B, L) or (B, 1, L)
    - mask: valid-position mask of shape (B, L) (optional)
    """

    def __init__(
        self,
        *,
        threshold: float = 0.2,
        alpha: float = 0.5,
        gamma: float = 2.0,
        peak_radius: int = 10,
        w_peak: float = 3.0,
        w_bg: float = 1.0,
        lambda_cls: float = 0.4,
        lambda_reg: float = 0.5,
        lambda_cons: float = 0.1,
        consistency_k: float = 10.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction only supports 'mean' or 'sum'")
        if peak_radius < 0:
            raise ValueError("peak_radius cannot be negative")

        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.peak_radius = int(peak_radius)
        self.w_peak = float(w_peak)
        self.w_bg = float(w_bg)
        self.lambda_cls = float(lambda_cls)
        self.lambda_reg = float(lambda_reg)
        self.lambda_cons = float(lambda_cons)
        self.consistency_k = float(consistency_k)
        self.reduction = reduction
        self.eps = 1e-8

    @staticmethod
    def _to_2d_sequence(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return x.squeeze(1)
        if x.ndim == 2:
            return x
        raise ValueError(f"Sequence tensor ndim should be 2 or 3, got {x.ndim}")

    @staticmethod
    def _to_1d_logits(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2 and x.shape[1] == 1:
            return x.squeeze(1)
        if x.ndim == 1:
            return x
        raise ValueError(f"Classification logits should be (B,) or (B,1), got {tuple(x.shape)}")

    def _reduce(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.reduction == "sum":
            if mask is None:
                return x.sum()
            return (x * mask).sum()
        if mask is None:
            return x.mean()
        return (x * mask).sum() / (mask.sum() + self.eps)

    def _focal_bce_with_logits(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * torch.pow(1.0 - p_t, self.gamma)
        focal_loss = -focal_weight * torch.log(p_t)
        return self._reduce(focal_loss)

    def _build_peak_weight_map(self, targets: torch.Tensor, y_cls: torch.Tensor) -> torch.Tensor:
        # Default background weight
        w = torch.full_like(targets, self.w_bg)
        if self.peak_radius == 0:
            return w

        peak_idx = targets.argmax(dim=1)  # (B,)
        bsz, seq_len = targets.shape
        for b in range(bsz):
            if y_cls[b] <= 0:
                continue
            c = int(peak_idx[b].item())
            lo = max(0, c - self.peak_radius)
            hi = min(seq_len, c + self.peak_radius + 1)
            w[b, lo:hi] = self.w_peak
        return w

    def forward(
        self,
        cls_logits: torch.Tensor,
        reg_predictions: torch.Tensor,
        reg_targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cls_logits = self._to_1d_logits(cls_logits)
        reg_predictions = self._to_2d_sequence(reg_predictions)
        reg_targets = self._to_2d_sequence(reg_targets)

        min_len = min(reg_predictions.shape[1], reg_targets.shape[1])
        reg_predictions = reg_predictions[:, :min_len]
        reg_targets = reg_targets[:, :min_len]

        if mask is not None:
            if mask.ndim != 2:
                raise ValueError(f"mask should be (B,L), got {tuple(mask.shape)}")
            if mask.shape[1] >= min_len:
                mask = mask[:, :min_len].float()
            else:
                mask = torch.ones_like(reg_predictions, dtype=reg_predictions.dtype)

        # 1) Sample-level label: whether a TSS exists in the window
        y_cls = (reg_targets.max(dim=1).values >= self.threshold).float()

        # 2) Classification loss: Focal BCE
        loss_cls = self._focal_bce_with_logits(cls_logits, y_cls)

        # 3) Regression loss: peak-weighted SmoothL1
        w = self._build_peak_weight_map(reg_targets, y_cls)
        reg_elem = F.smooth_l1_loss(reg_predictions, reg_targets, reduction="none")
        weighted_reg = reg_elem * w
        if mask is not None:
            weighted_reg = weighted_reg * mask
            reg_den = (w * mask).sum() + self.eps
        else:
            reg_den = w.sum() + self.eps
        if self.reduction == "sum":
            loss_reg = weighted_reg.sum()
        else:
            loss_reg = weighted_reg.sum() / reg_den

        # 4) Consistency loss: classification prob vs regression peak prob
        p_cls = torch.sigmoid(cls_logits).clamp(self.eps, 1.0 - self.eps)
        p_reg = torch.sigmoid(self.consistency_k * (reg_predictions.max(dim=1).values - self.threshold))
        loss_cons = F.binary_cross_entropy(p_cls, p_reg.detach(), reduction=self.reduction)

        total = self.lambda_cls * loss_cls + self.lambda_reg * loss_reg + self.lambda_cons * loss_cons
        detail = {
            "loss_cls": loss_cls.detach(),
            "loss_reg": loss_reg.detach(),
            "loss_cons": loss_cons.detach(),
            "positive_ratio": y_cls.mean().detach(),
        }
        return total, detail
