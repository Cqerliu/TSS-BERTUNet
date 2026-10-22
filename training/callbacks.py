#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training callbacks
"""

import torch
from typing import Optional
import os


class EarlyStopping:
    """Early stopping."""
    
    def __init__(
        self, 
        patience: int = 10, 
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Args:
            patience (int): Number of epochs with no improvement to wait
            min_delta (float): Minimum change to qualify as an improvement
            mode (str): 'min' means lower is better; 'max' means higher is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Args:
            score (float): Current score
        Returns:
            Whether training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Return whether the current score is better than the best so far."""
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta


class ModelCheckpoint:
    """Model checkpoint saver."""
    
    def __init__(
        self, 
        filepath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True
    ):
        """
        Args:
            filepath (str): Save path
            monitor (str): Metric to monitor
            mode (str): 'min' means lower is better; 'max' means higher is better
            save_best_only (bool): Whether to save only the best model
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None
    
    def __call__(
        self, 
        model: torch.nn.Module,
        score: float,
        epoch: int
    ):
        """
        Save the model.
        
        Args:
            model: Model
            score: Current score
            epoch: Current epoch
        """
        should_save = False
        
        if self.best_score is None:
            self.best_score = score
            should_save = True
        elif self.mode == 'min':
            if score < self.best_score:
                self.best_score = score
                should_save = True
        else:
            if score > self.best_score:
                self.best_score = score
                should_save = True
        
        if should_save or not self.save_best_only:
            # Create directory
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'score': score,
            }, self.filepath)
            
            if should_save:
                print(f"Model saved: {self.filepath} (score: {score:.4f})")


class LearningRateScheduler:
    """Learning rate scheduler."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = 'min',
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-6
    ):
        """
        Args:
            optimizer: Optimizer
            mode (str): 'min' monitors loss; 'max' monitors a metric
            factor (float): LR decay factor
            patience (int): Epochs with no improvement before reducing LR
            min_lr (float): Minimum learning rate
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.counter = 0
        self.best_score = None
        self.current_lr = optimizer.param_groups[0]['lr']
    
    def __call__(self, score: float):
        """
        Args:
            score (float): Current score
        """
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self._reduce_lr()
                self.counter = 0
    
    def _is_better(self, current: float, best: float) -> bool:
        """Return whether the current score is better than the best so far."""
        if self.mode == 'min':
            return current < best
        else:
            return current > best
    
    def _reduce_lr(self):
        """Reduce the learning rate."""
        old_lr = self.current_lr
        new_lr = max(old_lr * self.factor, self.min_lr)
        
        if new_lr < old_lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            self.current_lr = new_lr
            print(f"Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
