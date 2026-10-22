#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train DNABERT-2-117M for TSS (transcription start site) localization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add DNABERT-2-117M directory to path
dnabert_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'DNABERT-2-117M')
sys.path.insert(0, dnabert_path)

import argparse
import torch
import torch.nn as nn
import numpy as np

from model import DNABERTForTSSLocalization
from training import (
    DNABERTTrainer,
    EarlyStopping, 
    ModelCheckpoint,
    LearningRateScheduler
)
from data import load_data, create_dataloader, split_data
from utils import load_training_config, set_seed
from scripts.save_training_results import save_training_result


def main(config_path: str = 'config/dnabert_config.json'):
    """Main entry point."""
    print("=" * 60)
    print("DNABERT-2-117M TSS localization training")
    print("=" * 60)
    
    # 0. Load config
    config = load_training_config(config_path)
    DEFAULT_THRESHOLD = config['default_threshold']
    USE_NEGATIVE_SAMPLES = config['use_negative_samples']
    
    # DNABERT config
    DNABERT_CONFIG = config.get('dnabert', {})
    DNABERT_MODEL_PATH = DNABERT_CONFIG.get('model_path', 'DNABERT-2-117M')
    FREEZE_DNABERT = DNABERT_CONFIG.get('freeze_dnabert', False)
    MAX_SEQ_LENGTH = DNABERT_CONFIG.get('max_seq_length', 512)
    HIDDEN_SIZE = DNABERT_CONFIG.get('hidden_size', 768)
    DROPOUT = DNABERT_CONFIG.get('dropout', 0.1)
    USE_ONLY_UNET = config.get('use_only_unet', False)  # Ablation: U-Net only
    
    # Regression head config
    REGRESSION_HEAD_CONFIG = config.get('regression_head', {})
    USE_UNET = REGRESSION_HEAD_CONFIG.get('use_unet', True)  # Use U-Net by default
    REGRESSION_HIDDEN_DIM = REGRESSION_HEAD_CONFIG.get('hidden_dim', HIDDEN_SIZE // 2)
    REGRESSION_DROPOUT = REGRESSION_HEAD_CONFIG.get('dropout', 0.1)
    REGRESSION_NUM_LAYERS = REGRESSION_HEAD_CONFIG.get('num_layers', 2)
    
    # Data config
    DATA_CONFIG = config.get('data', {})
    MERGED_CSV_PATH = DATA_CONFIG.get('merged_csv_path', 'data/dataset/merged_dataset.csv')
    TRAIN_RATIO = DATA_CONFIG.get('train_ratio', 0.8)
    VAL_RATIO = DATA_CONFIG.get('val_ratio', 0.1)
    TEST_RATIO = DATA_CONFIG.get('test_ratio', 0.1)
    RANDOM_SEED = DATA_CONFIG.get('random_seed', 42)
    
    # Training hyperparameters
    TRAINING_CONFIG = config['training']
    BATCH_SIZE = TRAINING_CONFIG['batch_size']
    LEARNING_RATE = TRAINING_CONFIG['learning_rate']
    EPOCHS = TRAINING_CONFIG['epochs']
    
    # Loss hyperparameters (see training/focal_loss.py)
    LOSS_CONFIG = config.get('loss') or {}
    # Callback hyperparameters
    CALLBACKS_CONFIG = config['callbacks']
    
    # 1. Load data
    sequences, labels, chromosomes = load_data(
        merged_csv_path=MERGED_CSV_PATH,
        use_negative_samples=USE_NEGATIVE_SAMPLES,
        max_seq_length=MAX_SEQ_LENGTH
    )

    print(f"\nTotal sequences (all chromosomes): {len(sequences)}")

    # 1.1 Chromosome filter: keep chr1-chr21 only; drop chr22, chrX, chrY, and others
    allowed_chroms = {f"chr{i}" for i in range(1, 22)}
    filtered_sequences = []
    filtered_labels = []
    kept = 0
    dropped = 0
    for seq, lab, chrom in zip(sequences, labels, chromosomes):
        chrom = str(chrom)
        if chrom in allowed_chroms:
            filtered_sequences.append(seq)
            filtered_labels.append(lab)
            kept += 1
        else:
            dropped += 1

    sequences = filtered_sequences
    labels = filtered_labels

    print(f"After chromosome filter (chr1-chr21 only), kept samples: {kept}")
    print(f"Dropped samples (non chr1-chr21, e.g. chrX/chrY/chr22): {dropped}")
    print(f"Mean sequence length: {np.mean([len(s) for s in sequences]):.1f}")
    
    # 2. Split data
    print("\nSplitting dataset...")
    # Set all random seeds (including PyTorch)
    set_seed(RANDOM_SEED)
    print(f"Random seed set to: {RANDOM_SEED} (for reproducibility)")
    train_data, val_data, test_data = split_data(
        sequences, labels,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO
    )
    
    print(f"Train set: {len(train_data[0])}")
    print(f"Validation set: {len(val_data[0])}")
    print(f"Test set: {len(test_data[0])}")
    
    # 3. Create model and tokenizer
    if USE_ONLY_UNET:
        print("\nCreating model (ablation: U-Net only)...")
    else:
        print("\nCreating DNABERT-2 model...")
    
    # Resolve DNABERT model path (relative -> absolute)
    if not os.path.isabs(DNABERT_MODEL_PATH):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Project root is the directory containing train_dnabert.py
        project_root = script_dir
        dnabert_path = os.path.join(project_root, DNABERT_MODEL_PATH)
    else:
        dnabert_path = DNABERT_MODEL_PATH
    
    model = DNABERTForTSSLocalization(
        dnabert_path=dnabert_path,
        hidden_size=HIDDEN_SIZE,
        max_seq_length=MAX_SEQ_LENGTH,
        dropout=DROPOUT,
        freeze_dnabert=FREEZE_DNABERT,
        regression_hidden_dim=REGRESSION_HIDDEN_DIM,
        regression_dropout=REGRESSION_DROPOUT,
        regression_num_layers=REGRESSION_NUM_LAYERS,
        use_unet=USE_UNET,
        use_only_unet=USE_ONLY_UNET
    )
    
    # Get tokenizer (loaded during model init; also loaded in ablation mode for data loading)
    tokenizer = model.tokenizer
    
    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 4. Create data loaders
    print("\nCreating data loaders...")
    train_loader = create_dataloader(
        train_data[0], train_data[1], tokenizer,
        batch_size=BATCH_SIZE, shuffle=True, max_length=MAX_SEQ_LENGTH
    )
    val_loader = create_dataloader(
        val_data[0], val_data[1], tokenizer,
        batch_size=BATCH_SIZE, shuffle=False, max_length=MAX_SEQ_LENGTH
    )
    
    # 5. Create optimizer
    if USE_ONLY_UNET:
        # Ablation: no DNABERT; base-level head (U-Net or MLP) is trained from scratch.
        # Fine-tuning LRs like 2e-5 keep logits near 0, so sigmoid outputs stick at ~0.5.
        # Optimize only the regression head that participates in the forward pass.
        _alr = TRAINING_CONFIG.get('ablation_learning_rate', None)
        if _alr is not None:
            unet_lr = float(_alr)
        else:
            unet_lr = float(LEARNING_RATE) * float(
                TRAINING_CONFIG.get('ablation_lr_multiplier', 10.0)
            )
        optimizer = torch.optim.Adam(model.regression_head.parameters(), lr=unet_lr)
        print(
            f"Optimizer: Adam (ablation, base-level regression head only), lr={unet_lr} "
            f"(override via training.ablation_learning_rate; "
            f"otherwise learning_rate x training.ablation_lr_multiplier, default x50)"
        )
    elif FREEZE_DNABERT:
        # If DNABERT is frozen, optimize only the regression head (larger LR is fine)
        regression_params = list(model.regression_head.parameters())
        if hasattr(model, 'token_regression_head'):
            regression_params += list(model.token_regression_head.parameters())
        optimizer = torch.optim.Adam(regression_params, lr=LEARNING_RATE)
        print(f"Optimizer: Adam (regression head only), lr={LEARNING_RATE}")
    else:
        # When DNABERT is not frozen, use layered learning rates:
        # smaller LR for DNABERT (fine-tuning), larger LR for the regression head (from scratch)
        regression_params = list(model.regression_head.parameters())
        if hasattr(model, 'token_regression_head'):
            regression_params += list(model.token_regression_head.parameters())
        optimizer = torch.optim.Adam([
            {'params': model.dnabert.parameters(), 'lr': LEARNING_RATE},  # DNABERT fine-tuning
            {'params': regression_params, 'lr': LEARNING_RATE * 10}  # regression head from scratch
        ])
        print(f"Optimizer: Adam (layered learning rates)")
        print(f"  DNABERT learning rate: {LEARNING_RATE}")
        print(f"  Regression head learning rate: {LEARNING_RATE * 10}")
    
    # 6. Create loss function
    label_max_values = np.array([label.max() for label in labels])
    has_negative_samples = np.any(label_max_values < DEFAULT_THRESHOLD)
    use_negative = USE_NEGATIVE_SAMPLES and has_negative_samples
    
    print(f"\nDataset analysis:")
    print(f"  Total samples: {len(labels)}")
    print(f"  Positive samples: {np.sum(label_max_values >= DEFAULT_THRESHOLD)}")
    print(f"  Negative samples: {np.sum(label_max_values < DEFAULT_THRESHOLD)}")
    print(f"  Contains negatives: {has_negative_samples}")
    
    print(f"\nConfiguring loss function...")
    print("Using Focal-family losses (training/focal_loss.py)")
    if not use_negative:
        print("Warning: almost no negatives in current data; classification branch may be weakly discriminative")

    # focal_loss.py params: config['loss'] can override the defaults below
    _loss_defaults = {
        'focal_alpha': 0.5,
        'focal_gamma': 2.0,
        'use_regression_in_focal': True,
        'regression_weight': 0.5,
        'use_l1': True,
    }
    _merged_loss = {**_loss_defaults, **LOSS_CONFIG}
    # Only record peak-localization accuracy at these tolerances
    peak_tolerance_bps = [1, 50, 100, 150, 200, 250, 300]
    loss_config = {
        'use_negative_samples': use_negative,
        'threshold': DEFAULT_THRESHOLD,
        'peak_tolerance_bps': peak_tolerance_bps,
        **_merged_loss,
    }
    print(
        "  Loss params (from config.loss; missing keys use defaults): "
        f"focal_alpha={loss_config['focal_alpha']}, "
        f"focal_gamma={loss_config['focal_gamma']}, "
        f"use_regression_in_focal={loss_config['use_regression_in_focal']}, "
        f"regression_weight={loss_config['regression_weight']}, "
        f"use_l1={loss_config['use_l1']}, "
        f"peak_tolerance_bps={loss_config['peak_tolerance_bps']}"
    )
    
    # 7. Create callbacks (checkpoint / early stopping use val classification_accuracy; higher is better)
    METRIC_MONITOR = 'classification_accuracy'
    METRIC_MODE = 'max'

    early_stopping_config = CALLBACKS_CONFIG['early_stopping']
    early_stopping = EarlyStopping(
        patience=early_stopping_config['patience'],
        min_delta=early_stopping_config.get('min_delta', 0.001),
        mode=METRIC_MODE,
    )
    
    checkpoint_config = dict(CALLBACKS_CONFIG['model_checkpoint'])
    checkpoint_config['monitor'] = METRIC_MONITOR
    # Differentiate checkpoint filenames by data file, e.g. promoter_labeled_gaussian_100.csv -> best_gaussian_100.pth
    _csv_stem = os.path.splitext(os.path.basename(MERGED_CSV_PATH))[0] if MERGED_CSV_PATH else 'training'
    if _csv_stem.startswith('promoter_labeled_'):
        _ckpt_id = _csv_stem[len('promoter_labeled_'):]
    else:
        _ckpt_id = _csv_stem
    _ckpt_id = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in _ckpt_id)
    _ckpt_dir = os.path.dirname(checkpoint_config['path'])
    if not _ckpt_dir:
        _ckpt_dir = 'checkpoints'
    checkpoint_path = os.path.join(_ckpt_dir, f'best_{_ckpt_id}_unet.pth')
    checkpoint_config['path'] = checkpoint_path
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor=METRIC_MONITOR,
        mode=METRIC_MODE,
        save_best_only=checkpoint_config.get('save_best_only', True)
    )
    
    lr_scheduler_config = CALLBACKS_CONFIG['lr_scheduler']
    lr_scheduler = LearningRateScheduler(
        optimizer,
        mode=lr_scheduler_config['mode'],
        factor=lr_scheduler_config['factor'],
        patience=lr_scheduler_config['patience']
    )
    
    # 8. Create trainer (visualizations saved under dataset-named subfolders)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    MAX_GRAD_NORM = TRAINING_CONFIG.get('max_grad_norm', 1.0)
    # Dataset name from data path (no directory/extension), e.g. promoter_labeled_gaussian_100
    dataset_name = os.path.splitext(os.path.basename(MERGED_CSV_PATH))[0] if MERGED_CSV_PATH else 'training'

    trainer = DNABERTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_config=loss_config,
        device=device,
        max_grad_norm=MAX_GRAD_NORM,
        dataset_name=dataset_name
    )
    
    # 9. Start training
    print("\nStarting training...")
    history = trainer.fit(
        epochs=EPOCHS,
        early_stopping=early_stopping,
        checkpoint=checkpoint,
        lr_scheduler=lr_scheduler,
        early_stopping_monitor=METRIC_MONITOR,
    )
    
    print(f"\n{'='*60}")
    print("Training finished")
    print(f"{'='*60}")
    # Early stopping criterion: validation classification_accuracy (maximize)
    vm = history.get('val_metrics') or []
    if vm:
        accs = [m.get('classification_accuracy', float('nan')) for m in vm]
        best_acc = max(accs)
        best_epoch_acc = 1 + max(range(len(accs)), key=lambda i: accs[i])
        print(f"Best classification_accuracy: {best_acc:.4f} (Epoch {best_epoch_acc})")
    if history.get('val_loss'):
        best_val_loss = min(history['val_loss'])
        best_epoch_loss = 1 + min(range(len(history['val_loss'])), key=lambda i: history['val_loss'][i])
        print(f"Best val_loss: {best_val_loss:.4f} (Epoch {best_epoch_loss})")
    print("Checkpoint policy: higher classification_accuracy is better")
    print(f"Model saved to: {checkpoint_path}")
    
    # Save training results (callbacks write the actual checkpoint path)
    _callbacks_save = {**CALLBACKS_CONFIG, 'model_checkpoint': checkpoint_config}
    save_training_result(
        model_name='DNABERT-2-117M',
        history=history,
        model_path=checkpoint_path,
        best_metric_key=METRIC_MONITOR,
        training_config={
            'loss': loss_config,
            'data': DATA_CONFIG,
            'training': TRAINING_CONFIG,
            'callbacks': _callbacks_save
        },
        regression_head_config={
            'hidden_dim': REGRESSION_HIDDEN_DIM,
            'dropout': REGRESSION_DROPOUT,
            'num_layers': REGRESSION_NUM_LAYERS
        },
        use_unet=USE_UNET,
        model_config={
            'hidden_size': HIDDEN_SIZE,
            'max_seq_length': MAX_SEQ_LENGTH,
            'dropout': DROPOUT,
            'freeze_dnabert': FREEZE_DNABERT
        },
        use_only_unet=USE_ONLY_UNET
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train DNABERT-2-117M for TSS localization")
    parser.add_argument(
        '--config',
        type=str,
        default='config/training_config.json',
        help='Path to training config file (default: config/training_config.json)'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(config_path=args.config)
