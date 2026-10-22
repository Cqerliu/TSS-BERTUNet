#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration utility functions
"""

import os
import json
import numpy as np
import torch


def load_training_config(config_path: str = 'config/dnabert_config.json'):
    """
    Load training configuration file.
    
    Prefer dnabert_config.json; fall back to training_config.json if missing.
    
    Args:
        config_path: Path to config file, default 'config/dnabert_config.json'
    
    Returns:
        dict: Configuration parameter dictionary
    """
    # If using the default path and the file is missing, try training_config.json
    if config_path == 'config/dnabert_config.json' and not os.path.exists(config_path):
        fallback_path = 'config/training_config.json'
        if os.path.exists(fallback_path):
            print(f"Warning: {config_path} not found, using {fallback_path}")
            config_path = fallback_path
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Validate required keys
    required_keys = ['default_threshold', 'use_negative_samples']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Config file missing required parameter: {key}")
    
    # Set training hyperparameter defaults
    if 'training' not in config:
        config['training'] = {}
    training_defaults = {
        'batch_size': 16,  # DNABERT typically uses a smaller batch size
        'learning_rate': 2e-5,  # Pretrained models typically use a smaller learning rate
        'epochs': 100
    }
    for key, default_value in training_defaults.items():
        if key not in config['training']:
            config['training'][key] = default_value
    
    # Set loss hyperparameter defaults
    if 'loss' not in config:
        config['loss'] = {}
    
    # Set callback hyperparameter defaults
    if 'callbacks' not in config:
        config['callbacks'] = {}
    
    if 'early_stopping' not in config['callbacks']:
        config['callbacks']['early_stopping'] = {}
    if 'patience' not in config['callbacks']['early_stopping']:
        config['callbacks']['early_stopping']['patience'] = 15
    if 'min_delta' not in config['callbacks']['early_stopping']:
        config['callbacks']['early_stopping']['min_delta'] = 0.001
    
    if 'model_checkpoint' not in config['callbacks']:
        config['callbacks']['model_checkpoint'] = {}
    if 'path' not in config['callbacks']['model_checkpoint']:
        config['callbacks']['model_checkpoint']['path'] = 'models/best_dnabert.pth'
    if 'monitor' not in config['callbacks']['model_checkpoint']:
        config['callbacks']['model_checkpoint']['monitor'] = 'classification_accuracy'
    
    if 'lr_scheduler' not in config['callbacks']:
        config['callbacks']['lr_scheduler'] = {}
    if 'mode' not in config['callbacks']['lr_scheduler']:
        config['callbacks']['lr_scheduler']['mode'] = 'min'
    if 'factor' not in config['callbacks']['lr_scheduler']:
        config['callbacks']['lr_scheduler']['factor'] = 0.5
    if 'patience' not in config['callbacks']['lr_scheduler']:
        config['callbacks']['lr_scheduler']['patience'] = 5
    
    # DNABERT-specific config
    if 'dnabert' not in config:
        config['dnabert'] = {}
    dnabert_defaults = {
        'model_path': 'DNABERT-2-117M',  # DNABERT model path
        'freeze_dnabert': False,  # Whether to freeze DNABERT parameters
        'max_seq_length': 512,  # DNABERT max sequence length
        'hidden_size': 768,  # DNABERT hidden size
        'dropout': 0.1  # Dropout rate
    }
    for key, default_value in dnabert_defaults.items():
        if key not in config['dnabert']:
            config['dnabert'][key] = default_value
    
    # Regression head config
    if 'regression_head' not in config:
        config['regression_head'] = {}
    regression_head_defaults = {
        'hidden_dim': 384,  # Regression head hidden dim (hidden_size // 2)
        'dropout': 0.1,
        'num_layers': 2
    }
    for key, default_value in regression_head_defaults.items():
        if key not in config['regression_head']:
            config['regression_head'][key] = default_value
    
    # Data config
    if 'data' not in config:
        config['data'] = {}
    data_defaults = {
        'merged_csv_path': 'data/dataset/merged_dataset.csv',
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'random_seed': 42
    }
    for key, default_value in data_defaults.items():
        if key not in config['data']:
            config['data'][key] = default_value
    
    print(f"Loaded config file: {config_path}")
    print(f"  Default threshold: {config['default_threshold']}")
    print(f"  Use negative samples: {config['use_negative_samples']}")
    print(f"  DNABERT config:")
    print(f"    Model path: {config['dnabert']['model_path']}")
    print(f"    Freeze DNABERT: {config['dnabert']['freeze_dnabert']}")
    print(f"    Max sequence length: {config['dnabert']['max_seq_length']}")
    print(f"    Hidden size: {config['dnabert']['hidden_size']}")
    print(f"    Dropout: {config['dnabert']['dropout']}")
    print(f"  Regression head config:")
    print(f"    Hidden dim: {config['regression_head']['hidden_dim']}")
    print(f"    Dropout: {config['regression_head']['dropout']}")
    print(f"    Num layers: {config['regression_head']['num_layers']}")
    print(f"  Training params:")
    print(f"    batch_size: {config['training']['batch_size']}")
    print(f"    learning_rate: {config['training']['learning_rate']}")
    print(f"    epochs: {config['training']['epochs']}")
    print(f"  Data config:")
    print(f"    Data file: {config['data']['merged_csv_path']}")
    print(f"    Train/val/test ratios: {config['data']['train_ratio']}/{config['data']['val_ratio']}/{config['data']['test_ratio']}")
    
    return config


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_loss_weights(loss_config):
    """
    Get related loss hyperparameters from config.
    
    Args:
        loss_config: Loss config dict containing alpha, beta, gamma, etc.
    
    Returns:
        Dict of weights / hyperparameters
    """
    return {
        'focal_alpha': loss_config.get('focal_alpha', 0.25),
        'focal_gamma': loss_config.get('focal_gamma', 2.0),
        'use_regression_in_focal': loss_config.get('use_regression_in_focal', True),
        'regression_weight': loss_config.get('regression_weight', 0.5),
        'use_l1': loss_config.get('use_l1', True)
    }
