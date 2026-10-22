#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset utility functions
"""

import torch
import numpy as np
import pandas as pd


def load_data(merged_csv_path='data/dataset/merged_dataset.csv',
              use_negative_samples=True,
              max_seq_length=512):
    """
    Load DNA sequences and labels.
    
    Args:
        merged_csv_path: Path to the merged dataset CSV
        use_negative_samples: Whether to include negative samples
        max_seq_length: Max sequence length (DNABERT limit is 512)
    
    Returns:
        (sequences_list, labels_list, chromosomes_list):
            Sequence list, label list, and corresponding chromosome list
    """
    print("Loading data...")
    
    # 1. Load merged dataset
    print(f"  Reading merged dataset: {merged_csv_path}")
    merged_df = pd.read_csv(merged_csv_path)
    print(f"    {len(merged_df)} records")
    
    # 2. Filter by use_negative_samples
    if not use_negative_samples:
        merged_df = merged_df[merged_df['classification_label'] == 1]
        print(f"    After filtering (positives only): {len(merged_df)} records")
    
    # 3. Process each sequence
    print("  Processing sequences and labels...")
    sequences_list = []
    labels_list = []
    chromosomes_list = []
    
    for idx, row in merged_df.iterrows():
        try:
            # Get sequence
            sequence = str(row['sequence']).upper()
            if len(sequence) == 0:
                continue
            
            # Truncate or pad to max_seq_length (if specified)
            if max_seq_length is not None and len(sequence) > max_seq_length:
                sequence = sequence[:max_seq_length]
                seq_len = max_seq_length
            else:
                seq_len = len(sequence)
            
            # Parse regression labels
            regression_label_str = str(row.get('regression_label', ''))
            if regression_label_str and regression_label_str != 'nan':
                if ',' in regression_label_str:
                    label_values = [float(x.strip()) for x in regression_label_str.split(',')]
                else:
                    label_values = [float(regression_label_str)]
                
                # Ensure label length matches sequence length
                if len(label_values) != seq_len:
                    if len(label_values) > seq_len:
                        label_values = label_values[:seq_len]
                    else:
                        label_values.extend([0.0] * (seq_len - len(label_values)))
                
                label_array = np.array(label_values, dtype=np.float32)
            else:
                # If no labels, create zero labels
                label_array = np.zeros(seq_len, dtype=np.float32)
            
            sequences_list.append(sequence)
            labels_list.append(label_array)
            chromosomes_list.append(str(row.get('chromosome', '')))
            
        except Exception as e:
            print(f"    Error: failed processing sequence {row.get('unique_id', idx)}: {e}")
            continue
    
    if len(sequences_list) == 0:
        raise ValueError("No valid sequence data")
    
    print(f"  Successfully processed {len(sequences_list)} sequences")
    print(f"  Sequence length range: {min(len(s) for s in sequences_list)} - {max(len(s) for s in sequences_list)}")
    
    return sequences_list, labels_list, chromosomes_list


def create_dataloader(
    sequences,
    labels,
    tokenizer,
    batch_size=16,
    shuffle=True,
    max_length=512,
    num_workers=0,
    persistent_workers=False,
    prefetch_factor=None,
    pin_memory=False,
):
    """
    Create a data loader.
    
    Args:
        sequences: List of sequences
        labels: List of labels
        tokenizer: DNABERT tokenizer
        batch_size: Batch size
        shuffle: Whether to shuffle
        max_length: Max sequence length
        num_workers: Number of DataLoader worker processes; >0 overlaps tokenize/collate with GPU
        persistent_workers: Keep workers alive across epochs when num_workers>0
        prefetch_factor: Prefetch batches per worker; only when num_workers>0; default 2
        pin_memory: Pin tensors in page-locked memory for async GPU transfer
    
    Returns:
        DataLoader
    """
    class TSSDataset(torch.utils.data.Dataset):
        def __init__(self, sequences, labels, tokenizer, max_length):
            self.sequences = sequences
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            sequence = self.sequences[idx]
            label = self.labels[idx]
            
            # Tokenize sequence
            # Note: DNABERT-2 uses k-mer tokenization (typically 6-mer); label alignment needs care
            tokenizer_kwargs = {
                'return_tensors': 'pt',
                'padding': 'longest' if self.max_length is None else 'max_length',
                'truncation': self.max_length is not None,
                'return_offsets_mapping': True  # Map tokens back to the original sequence
            }
            if self.max_length is not None:
                tokenizer_kwargs['max_length'] = self.max_length
            
            encoded = self.tokenizer(sequence, **tokenizer_kwargs)
            input_ids = encoded['input_ids'].squeeze(0)  # (max_length,)
            attention_mask = encoded['attention_mask'].squeeze(0)  # (max_length,)
            
            # Label alignment: DNABERT k-mer tokenization requires correct mapping
            tokenized_length = input_ids.shape[0]
            
            # Get offset mapping if available
            offset_mapping = None
            if 'offset_mapping' in encoded and encoded['offset_mapping'] is not None:
                offset_mapping = encoded['offset_mapping'].squeeze(0)  # (max_length, 2)
            
            # Original sequence length
            original_seq_length = len(sequence)
            
            # Keep labels at base level (e.g. 600bp)
            # Do not convert to token-level labels; the model now outputs base-level predictions
            label_tensor = torch.FloatTensor(label).unsqueeze(0)  # (1, 600)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': label_tensor,  # (1, 600) - base-level labels
                'offset_mapping': offset_mapping,  # (L_token, 2)
                'original_seq_length': original_seq_length,  # int
                'original_sequence': sequence  # str - original sequence for one-hot encoding
            }
    
    def collate_fn(batch):
        """Dynamically pad sequences in a batch to the same length."""
        # Max token length and max base length in the batch
        max_token_len = max(item['input_ids'].shape[0] for item in batch)
        max_base_len = max(item['labels'].shape[1] for item in batch)  # labels are (1, L_base)
        
        # Tokenizer pad_token_id
        pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
        
        # Pad input_ids, attention_mask, labels, and offset_mapping
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        padded_offset_mappings = []
        original_seq_lengths = []
        original_sequences = []  # Keep original sequences for one-hot encoding
        
        for item in batch:
            input_ids = item['input_ids']
            attention_mask = item['attention_mask']
            labels = item['labels']  # (1, L_base) - base-level labels
            offset_mapping = item.get('offset_mapping', None)
            original_seq_length = item.get('original_seq_length', labels.shape[1])
            original_sequence = item.get('original_sequence', '')  # Original sequence
            
            seq_len = input_ids.shape[0]
            pad_token_length = max_token_len - seq_len
            
            # Pad token-level data
            if pad_token_length > 0:
                # Pad input_ids
                padding = torch.full((pad_token_length,), pad_token_id, dtype=input_ids.dtype)
                padded_input_id = torch.cat([input_ids, padding])
                
                # Pad attention_mask
                padding_mask = torch.zeros(pad_token_length, dtype=attention_mask.dtype)
                padded_mask = torch.cat([attention_mask, padding_mask])
                
                # Pad offset_mapping
                if offset_mapping is not None:
                    padding_offset = torch.zeros(pad_token_length, 2, dtype=offset_mapping.dtype)
                    padded_offset = torch.cat([offset_mapping, padding_offset], dim=0)
                else:
                    padded_offset = None
            else:
                padded_input_id = input_ids
                padded_mask = attention_mask
                padded_offset = offset_mapping
            
            # Pad base-level labels
            base_len = labels.shape[1]
            pad_base_length = max_base_len - base_len
            if pad_base_length > 0:
                padding_labels = torch.zeros(1, pad_base_length, dtype=labels.dtype)
                padded_label = torch.cat([labels, padding_labels], dim=1)
            else:
                padded_label = labels
            
            padded_input_ids.append(padded_input_id)
            padded_attention_masks.append(padded_mask)
            padded_labels.append(padded_label)
            padded_offset_mappings.append(padded_offset)
            original_seq_lengths.append(original_seq_length)
            original_sequences.append(original_sequence)
        
        result = {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_masks),
            'labels': torch.stack(padded_labels),
            'offset_mapping': padded_offset_mappings,  # List of tensors
            'original_seq_length': original_seq_lengths,  # List of int
            'original_sequence': original_sequences  # List of str - original sequences
        }
        
        return result
    
    dataset = TSSDataset(sequences, labels, tokenizer, max_length)
    dl_kw = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": int(num_workers),
        "collate_fn": collate_fn,
        "pin_memory": bool(pin_memory),
    }
    nw = int(num_workers)
    if nw > 0:
        dl_kw["persistent_workers"] = bool(persistent_workers)
        pf = 2 if prefetch_factor is None else int(prefetch_factor)
        dl_kw["prefetch_factor"] = max(1, pf)
    dataloader = torch.utils.data.DataLoader(dataset, **dl_kw)

    return dataloader


def split_data(sequences, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split dataset into train / validation / test."""
    n = len(sequences)
    
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    indices = np.random.permutation(n)
    
    train_sequences = [sequences[i] for i in indices[:train_end]]
    train_labels = [labels[i] for i in indices[:train_end]]
    
    val_sequences = [sequences[i] for i in indices[train_end:val_end]]
    val_labels = [labels[i] for i in indices[train_end:val_end]]
    
    test_sequences = [sequences[i] for i in indices[val_end:]]
    test_labels = [labels[i] for i in indices[val_end:]]
    
    return (train_sequences, train_labels), (val_sequences, val_labels), (test_sequences, test_labels)
