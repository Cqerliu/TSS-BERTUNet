#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DNABERT-2-117M TSS localization model
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Add DNABERT-2-117M directory to path
# Resolve project root
current_file = os.path.abspath(__file__)
model_dir = os.path.dirname(current_file)
project_root = os.path.dirname(model_dir)
dnabert_path = os.path.join(project_root, 'DNABERT-2-117M')
if os.path.exists(dnabert_path):
    sys.path.insert(0, dnabert_path)


class UNet1D(nn.Module):
    """1D U-Net architecture for sequence regression.
    
    Encoder-decoder with skip connections for predicting TSS probabilities from feature sequences.
    """
    
    def __init__(
        self,
        in_channels: int = 768,
        base_channels: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            in_channels: Input feature dim (DNABERT hidden_size)
            base_channels: Base channel count (first encoder layer)
            num_layers: Number of U-Net layers (encoder/decoder depth)
            dropout: Dropout rate
        """
        super(UNet1D, self).__init__()
        
        self.num_layers = num_layers
        self.base_channels = base_channels
        self.in_channels = in_channels  # Keep for forward layout checks
        
        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        encoder_channels = [in_channels] + [base_channels * (2 ** i) for i in range(num_layers)]
        
        for i in range(num_layers):
            in_ch = encoder_channels[i]
            out_ch = encoder_channels[i + 1]
            
            # Each encoder block: Conv1d -> BatchNorm -> ReLU -> Dropout -> Conv1d -> BatchNorm -> ReLU
            encoder_block = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.encoder.append(encoder_block)
        
        # Bottleneck (bottom layer)
        bottleneck_channels = encoder_channels[-1]
        self.bottleneck = nn.Sequential(
            nn.Conv1d(bottleneck_channels, bottleneck_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(bottleneck_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(bottleneck_channels * 2, bottleneck_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(bottleneck_channels),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()
        # Decoder channels: start from bottleneck output, gradually reduce back toward input channels
        # Bottleneck output channels = encoder_channels[-1] (same as last encoder layer)
        # Decoder layer outputs = reversed encoder channels (from second-to-last down to first)
        decoder_channels = [encoder_channels[-1]]  # Bottleneck output (= first decoder input)
        # Append decoder output channel counts (reversed encoder channels, 2nd-last -> first)
        for i in range(num_layers - 1, -1, -1):
            decoder_channels.append(encoder_channels[i])
        
        for i in range(num_layers):
            # Channel arithmetic for decoder layer i:
            # 1. Upsample input: decoder_channels[i] (bottleneck or previous decoder output)
            # 2. Upsample output: decoder_channels[i] (unchanged)
            # 3. Encoder skip: encoder_features[i] corresponds to encoder_channels[num_layers - i]
            #    because encoder_features[0] stores the first encoder block output (encoder_channels[1])
            #    and encoder_features[1] stores the second (encoder_channels[2]), so
            #    encoder_features[num_layers - 1 - i] maps to encoder_channels[num_layers - i]
            # 4. Concat input: decoder_channels[i] + encoder_channels[num_layers - i]
            # 5. Decoder output: decoder_channels[i + 1]
            upsample_in_ch = decoder_channels[i]
            upsample_out_ch = decoder_channels[i]  # Upsampling does not change channels
            # encoder_features[num_layers - 1 - i] <-> encoder_channels[num_layers - i]
            encoder_skip_ch = encoder_channels[num_layers - i]
            conv_in_ch = upsample_out_ch + encoder_skip_ch
            conv_out_ch = decoder_channels[i + 1]
            
            # Upsampling layer
            upsample = nn.ConvTranspose1d(
                upsample_in_ch, 
                upsample_out_ch, 
                kernel_size=2, 
                stride=2
            )
            
            # Decoder block
            decoder_block = nn.Sequential(
                nn.Conv1d(conv_in_ch, conv_out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(conv_out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(conv_out_ch, conv_out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(conv_out_ch),
                nn.ReLU(inplace=True)
            )
            
            self.decoder.append(nn.ModuleDict({
                'upsample': upsample,
                'conv': decoder_block
            }))
        
        # Output layer: map features to 1-D (TSS probability)
        self.output_conv = nn.Conv1d(decoder_channels[-1], 1, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features (B, L, C) or (B, C, L)
        
        Returns:
            output: Predictions (B, 1, L)
        """
        # Ensure input layout is (B, C, L)
        # If dim 1 equals in_channels, treat as (B, C, L); otherwise (B, L, C) and transpose
        if x.ndim == 3:
            # Check whether dim 1 equals in_channels
            # A large dim 1 (e.g. 600) is usually sequence length -> transpose
            # Dim 1 equal to in_channels (e.g. 772 = 768+4) means already (B, C, L)
            if x.shape[1] != self.in_channels and x.shape[2] == self.in_channels:
                # (B, L, C) -> (B, C, L)
                x = x.transpose(1, 2)  # (B, L, C) -> (B, C, L)
        
        # Encoder path (store features for skip connections)
        encoder_features = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            encoder_features.append(x)
            # Downsample with stride-2 pooling
            x = F.max_pool1d(x, kernel_size=2, stride=2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path (with skip connections)
        for i, decoder_dict in enumerate(self.decoder):
            # Upsample
            x = decoder_dict['upsample'](x)
            
            # Matching encoder feature (skip connection)
            # encoder_features indices roughly: 0->256, 1->512, 2->1024, 3->2048
            # Decoder layer i should connect to encoder layer (num_layers - 1 - i)
            encoder_idx = self.num_layers - 1 - i
            encoder_feat = encoder_features[encoder_idx]
            
            # Match lengths (pooling may cause size mismatch)
            if x.shape[2] != encoder_feat.shape[2]:
                # Interpolate to match
                x = F.interpolate(x, size=encoder_feat.shape[2], mode='linear', align_corners=False)
            
            # Concatenate (skip connection)
            x = torch.cat([x, encoder_feat], dim=1)
            
            # Decoder convolutions
            x = decoder_dict['conv'](x)
        
        # Output layer
        output = self.output_conv(x)
        
        return output  # (B, 1, L)


class DNABERTForTSSLocalization(nn.Module):
    """DNABERT-2-based TSS localization model.
    
    Architecture:
    1. DNABERT-2 extracts sequence features
    2. Regression head predicts per-position TSS probabilities
    """
    
    def __init__(
        self,
        dnabert_path: str = 'DNABERT-2-117M',
        hidden_size: int = 768,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        freeze_dnabert: bool = False,
        regression_hidden_dim: int = None,
        regression_dropout: float = None,
        regression_num_layers: int = 2,
        max_original_length: int = 600,
        use_unet: bool = True,
        use_only_unet: bool = False
    ):
        """
        Args:
            dnabert_path: Path to the DNABERT-2 model
            hidden_size: DNABERT hidden size (default 768)
            max_seq_length: DNABERT max sequence length (default 512)
            dropout: Dropout rate
            freeze_dnabert: Whether to freeze DNABERT parameters
            regression_hidden_dim: Regression head hidden dim (default hidden_size // 2)
            regression_dropout: Regression head dropout (default dropout)
            regression_num_layers: Number of regression head layers (default 2)
            max_original_length: Max original sequence length (default 600bp)
            use_unet: Use U-Net as the base-level regression head (default True)
            use_only_unet: Ablation: U-Net only, without DNABERT-2 (default False)
        """
        super(DNABERTForTSSLocalization, self).__init__()
        
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.max_original_length = max_original_length
        self.use_only_unet = use_only_unet
        
        # One-hot map: A, T, G, C -> [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]
        self.base_to_onehot = {
            'A': torch.tensor([1, 0, 0, 0], dtype=torch.float32),
            'T': torch.tensor([0, 1, 0, 0], dtype=torch.float32),
            'G': torch.tensor([0, 0, 1, 0], dtype=torch.float32),
            'C': torch.tensor([0, 0, 0, 1], dtype=torch.float32)
        }
        # Ablation: U-Net only, without DNABERT-2
        if use_only_unet:
            print("=" * 60)
            print("Ablation mode: U-Net only, without DNABERT-2")
            print("=" * 60)
            # Still load tokenizer for data loading (model does not use DNABERT)
            # Ensure absolute path
            if not os.path.isabs(dnabert_path):
                current_file = os.path.abspath(__file__)
                model_dir = os.path.dirname(current_file)
                project_root = os.path.dirname(model_dir)
                dnabert_path = os.path.join(project_root, dnabert_path)
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    dnabert_path, 
                    trust_remote_code=True
                )
                print(f"  Loaded tokenizer for data loading: {dnabert_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load tokenizer: {e}\nPlease check that model files are complete")
            
            self.dnabert = None  # Do not use DNABERT
            # One-hot only; feature dim = 4
            self.enhanced_hidden_size = 4
        else:
            # Feature dim: DNABERT (768) + one-hot (4) = 772
            self.enhanced_hidden_size = hidden_size + 4
            
            # Load DNABERT-2 model and tokenizer
            # Ensure absolute path
            if not os.path.isabs(dnabert_path):
                # Convert relative path to absolute
                # Parent of this file's directory is the project root
                current_file = os.path.abspath(__file__)
                model_dir = os.path.dirname(current_file)
                # Project root is the parent of the model/ directory
                project_root = os.path.dirname(model_dir)
                dnabert_path = os.path.join(project_root, dnabert_path)
            
            print(f"Loading DNABERT-2 model from: {dnabert_path}")
            if not os.path.exists(dnabert_path):
                raise FileNotFoundError(
                    f"DNABERT-2 model directory not found: {dnabert_path}\n"
                    f"Please ensure DNABERT-2-117M is under the project root"
                )
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    dnabert_path, 
                    trust_remote_code=True
                )
                self.dnabert = AutoModel.from_pretrained(
                    dnabert_path, 
                    trust_remote_code=True
                )
                print(f"  [OK] Model loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load DNABERT-2 model: {e}\nPlease check that model files are complete")
            
            # Optionally freeze DNABERT parameters
            if freeze_dnabert:
                print("Freezing DNABERT parameters")
                for param in self.dnabert.parameters():
                    param.requires_grad = False
        
        # Regression head: map DNABERT outputs to TSS probabilities
        self.dropout = nn.Dropout(dropout)
        # Use configured regression hidden dim, or hidden_size // 2
        if regression_hidden_dim is None:
            regression_hidden_dim = hidden_size // 2
        if regression_dropout is None:
            regression_dropout = dropout
        
        # Token-level regression head (coarse localization; optional; kept simple)
        token_layers = []
        token_layers.append(nn.Linear(hidden_size, regression_hidden_dim))
        token_layers.append(nn.ReLU())
        token_layers.append(nn.Dropout(regression_dropout))
        
        for _ in range(regression_num_layers - 2):
            token_layers.append(nn.Linear(regression_hidden_dim, regression_hidden_dim))
            token_layers.append(nn.ReLU())
            token_layers.append(nn.Dropout(regression_dropout))
        
        token_layers.append(nn.Linear(regression_hidden_dim, 1))
        self.token_regression_head = nn.Sequential(*token_layers)
        
        # Base-level regression head: U-Net or simple MLP per config
        # Note: input dim depends on use_only_unet
        # - use_only_unet=False: enhanced_hidden_size = 768+4=772 (DNABERT + one-hot)
        # - use_only_unet=True: enhanced_hidden_size = 4 (one-hot only)
        if use_unet:
            # U-Net architecture
            # base_channels: U-Net base channel count; default to regression_hidden_dim
            unet_base_channels = regression_hidden_dim if regression_hidden_dim >= 128 else 128
            self.base_regression_head = UNet1D(
                in_channels=self.enhanced_hidden_size,  # 772 (768+4) or 4 (one-hot only)
                base_channels=unet_base_channels,
                num_layers=4,  # 4 encoder/decoder layers
                dropout=regression_dropout
            )
            print(f"  Using U-Net as the base-level regression head")
            if use_only_unet:
                print(f"    U-Net input dim: {self.enhanced_hidden_size} (one-hot only: 4)")
            else:
                print(f"    U-Net input dim: {self.enhanced_hidden_size} (DNABERT: {hidden_size} + one-hot: 4)")
            print(f"    U-Net base channels: {unet_base_channels}")
            print(f"    U-Net layers: 4")
        else:
            # Simple regression head (multi-layer MLP)
            base_layers = []
            base_layers.append(nn.Linear(self.enhanced_hidden_size, regression_hidden_dim))  # 772 -> hidden_dim
            base_layers.append(nn.ReLU())
            base_layers.append(nn.Dropout(regression_dropout))
            
            for _ in range(regression_num_layers - 2):
                base_layers.append(nn.Linear(regression_hidden_dim, regression_hidden_dim))
                base_layers.append(nn.ReLU())
                base_layers.append(nn.Dropout(regression_dropout))
            
            base_layers.append(nn.Linear(regression_hidden_dim, 1))
            self.base_regression_head = nn.Sequential(*base_layers)
            print(f"  Using a simple MLP as the base-level regression head")
            if use_only_unet:
                print(f"    Input dim: {self.enhanced_hidden_size} (one-hot only: 4)")
            else:
                print(f"    Input dim: {self.enhanced_hidden_size} (DNABERT: {hidden_size} + one-hot: 4)")
            print(f"    Hidden dim: {regression_hidden_dim}")
            print(f"    Num layers: {regression_num_layers}")
        
        # For compatibility, keep regression_head pointing at base_regression_head
        self.regression_head = self.base_regression_head
        
        # Sigmoid (map outputs to [0, 1])
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask=None, target=None, threshold: float = 0.1, 
                offset_mapping=None, original_seq_length=None, original_sequence=None):
        """
        Forward pass.
        
        Args:
            input_ids: Tokenized input, shape (B, L_token)
            attention_mask: Attention mask, shape (B, L_token)
            target: Target labels (for training, optional) - unused in the forward pass;
                    kept for interface compatibility
            threshold: TSS decision threshold
            offset_mapping: List of (L_token, 2) tensors - base-position ranges per token
            original_seq_length: List of int - original sequence length per sample
            original_sequence: List of str - original DNA sequences (for one-hot encoding)
        
        Returns:
            output: Predicted TSS probabilities, shape (B, 1, max_original_length)
        
        Notes:
            - Predictions depend only on input_ids (DNA sequence), not on target
            - offset_mapping maps token-level features to base level for precise base-level prediction
            - original_sequence provides one-hot features to improve base-level discriminability
        """
        # Ablation: U-Net only; build one-hot features directly from the original sequence
        if self.use_only_unet:
            if original_sequence is None or original_seq_length is None:
                raise ValueError("When use_only_unet=True, original_sequence and original_seq_length are required")
            
            # Resolve device from input_ids or attention_mask if provided
            device = None
            if input_ids is not None:
                device = input_ids.device
            elif attention_mask is not None:
                device = attention_mask.device
            
            # One-hot features from original sequences
            base_features = self._generate_onehot_features(
                original_sequence, original_seq_length, device=device
            )
            
            # Base-level regression
            if isinstance(self.base_regression_head, UNet1D):
                base_predictions = self.base_regression_head(base_features)  # (B, 1, max_length)
            else:
                base_predictions = self.base_regression_head(base_features)  # (B, max_length, 1)
                base_predictions = base_predictions.transpose(1, 2)  # (B, 1, max_length)
            
            # Sigmoid activation
            output = self.sigmoid(base_predictions)
            
            # # Apply mask if available
            # if attention_mask is not None and original_seq_length is not None:
            #     # Simple length-based mask
            #     if isinstance(original_seq_length, list):
            #         max_length = max(original_seq_length)
            #     else:
            #         max_length = original_seq_length
            #     
            #     base_mask = torch.zeros(
            #         base_features.shape[0], max_length,
            #         device=base_features.device,
            #         dtype=torch.float32
            #     )
            #     for i, seq_len in enumerate(original_seq_length if isinstance(original_seq_length, list) else [original_seq_length]):
            #         base_mask[i, :seq_len] = 1.0
            #     
            #     if base_mask.shape[1] == output.shape[2]:
            #         output = output * base_mask.unsqueeze(1)
            
            return output
        
        # Normal mode: use DNABERT-2
        # DNABERT forward
        dnabert_outputs = self.dnabert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        
        # Hidden states: [batch_size, seq_length, hidden_size]
        if isinstance(dnabert_outputs, tuple):
            hidden_states = dnabert_outputs[0]
        elif hasattr(dnabert_outputs, 'last_hidden_state'):
            hidden_states = dnabert_outputs.last_hidden_state
        else:
            hidden_states = dnabert_outputs
        
        # Dropout
        hidden_states = self.dropout(hidden_states)
        
        # Map token features to base level via offset_mapping
        if offset_mapping is not None and original_seq_length is not None:
            # Improved scheme: map feature vectors to base level, then regress
            # Add one-hot so each base position has a distinctive feature
            base_features = self._map_token_features_to_bases(
                hidden_states, offset_mapping, original_seq_length, original_sequence
            )
            
            # Base-level regression
            # base_features: (B, max_length, 772) - DNABERT (768) + one-hot (4)
            if isinstance(self.base_regression_head, UNet1D):
                # U-Net accepts (B, C, L) or (B, L, C) and handles layout automatically
                base_predictions = self.base_regression_head(base_features)  # (B, 1, max_length)
            else:
                # Simple head expects (B, L, C)
                base_predictions = self.base_regression_head(base_features)  # (B, max_length, 1)
                base_predictions = base_predictions.transpose(1, 2)  # (B, 1, max_length)
            
            # Sigmoid activation
            output = self.sigmoid(base_predictions)
            
            # Apply attention_mask (expanded to base level)
            if attention_mask is not None:
                base_mask = self._expand_mask_to_bases_with_offset(
                    attention_mask, offset_mapping, original_seq_length
                )
                # Ensure base_mask length matches
                if base_mask.shape[1] == output.shape[2]:
                    output = output * base_mask.unsqueeze(1)  # (B, 1, max_length)
                else:
                    # Length mismatch: apply mask only on the overlapping prefix
                    min_len = min(base_mask.shape[1], output.shape[2])
                    output[:, :, :min_len] = output[:, :, :min_len] * base_mask[:, :min_len].unsqueeze(1)
        else:
            # Without offset_mapping, fall back to token-level prediction (legacy compatibility)
            logits = self.token_regression_head(hidden_states)  # (B, L_token, 1)
            logits = logits.transpose(1, 2)  # (B, 1, L_token)
            output = self.sigmoid(logits)
            
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(1).float()
                output = output * mask
        
        return output
    
    def _generate_onehot_features(self, original_sequence, original_seq_length, device=None):
        """
        Ablation helper: build one-hot features from original sequences (no DNABERT).
        
        Args:
            original_sequence: List of str - original DNA sequence per sample
            original_seq_length: List of int - original sequence length per sample
            device: torch.device (optional)
        
        Returns:
            base_features: (B, max_length, 4) - base-level one-hot features
        """
        batch_size = len(original_sequence) if isinstance(original_sequence, list) else 1
        
        # Handle original_seq_length
        if isinstance(original_seq_length, list):
            max_length = max(original_seq_length)
        else:
            max_length = original_seq_length if original_seq_length is not None else self.max_original_length
        
        base_features_list = []
        for i in range(batch_size):
            # Get sequence
            if isinstance(original_sequence, list):
                seq = original_sequence[i]
            else:
                seq = original_sequence
            
            # Handle sequence length
            if isinstance(original_seq_length, list):
                orig_len = original_seq_length[i]
            else:
                orig_len = original_seq_length if original_seq_length is not None else self.max_original_length
            
            # Initialize one-hot features
            onehot_features = torch.zeros(
                orig_len, 4,
                dtype=torch.float32,
                device=device
            )
            
            # One-hot encode each base position
            for pos in range(min(orig_len, len(seq))):
                base_char = seq[pos].upper() if pos < len(seq) else 'A'
                # Default to A if the base is not in ATGC
                onehot = self.base_to_onehot.get(base_char, self.base_to_onehot['A'])
                onehot_features[pos] = onehot.to(device=device) if device is not None else onehot
            
            # Pad to max_length
            if orig_len < max_length:
                padding = torch.zeros(
                    max_length - orig_len, 4,
                    dtype=torch.float32,
                    device=device
                )
                onehot_features = torch.cat([onehot_features, padding], dim=0)
            
            base_features_list.append(onehot_features)
        
        return torch.stack(base_features_list)  # (B, max_length, 4)
    
    def _map_token_features_to_bases(self, hidden_states, offset_mapping, original_seq_length, original_sequence=None):
        """
        Map token-level features to base level via offset_mapping, with one-hot enrichment.
        
        Key improvements:
        1. Map feature vectors (not predictions) so the model learns at base level
        2. Add one-hot encoding so bases within the same token remain distinguishable
        
        Args:
            hidden_states: (B, L_token, 768) - token-level feature vectors
            offset_mapping: List of (L_token, 2) tensors - base-position ranges per token
            original_seq_length: List of int - original sequence length per sample
            original_sequence: List of str - original DNA sequences (for one-hot encoding)
        
        Returns:
            base_features: (B, max_length, 772) - base-level features
                           (768-D DNABERT + 4-D one-hot)
        """
        batch_size = hidden_states.shape[0]
        
        # original_seq_length may be a list or a scalar
        if isinstance(original_seq_length, list):
            max_length = max(original_seq_length)
        else:
            max_length = original_seq_length if original_seq_length is not None else self.max_original_length
        
        base_features_list = []
        for i in range(batch_size):
            offsets = offset_mapping[i]  # (L_token, 2) or None
            token_features = hidden_states[i]  # (L_token, 768)
            
            # Skip sample if offsets are missing
            if offsets is None:
                print(f"Warning: sample {i} has no offset_mapping; skipping")
                continue
            
            # Ensure offsets is a tensor
            if not isinstance(offsets, torch.Tensor):
                offsets = torch.tensor(offsets, device=hidden_states.device)
            
            # Handle original_seq_length
            if isinstance(original_seq_length, list):
                orig_len = original_seq_length[i]
            else:
                orig_len = original_seq_length if original_seq_length is not None else self.max_original_length
            
            # Get original sequence (for one-hot)
            seq = None
            if original_sequence is not None and isinstance(original_sequence, list) and i < len(original_sequence):
                seq = original_sequence[i]
            elif original_sequence is not None and not isinstance(original_sequence, list):
                seq = original_sequence
            
            # Initialize base-level features (zeros)
            base_features = torch.zeros(
                orig_len, self.hidden_size,
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )
            
            # Initialize one-hot features
            onehot_features = torch.zeros(
                orig_len, 4,
                device=hidden_states.device,
                dtype=torch.float32
            )
            
            # Map features via offset_mapping
            # Ensure offsets shape is (L_token, 2)
            if offsets.ndim != 2 or offsets.shape[1] != 2:
                # Try reshape or skip
                if offsets.numel() > 0:
                    try:
                        offsets = offsets.view(-1, 2)
                    except:
                        print(f"Warning: sample {i} has invalid offset_mapping shape: {offsets.shape}")
                        continue
                else:
                    continue
            
            num_tokens = offsets.shape[0]
            for token_idx in range(num_tokens):
                # Read start/end
                start_val = offsets[token_idx, 0]
                end_val = offsets[token_idx, 1]
                
                # Convert to Python scalars
                if isinstance(start_val, torch.Tensor):
                    if start_val.numel() == 1:
                        start = start_val.item()
                    else:
                        # Multi-element tensor: take the first
                        start = int(start_val.flatten()[0].item())
                else:
                    start = int(start_val)
                
                if isinstance(end_val, torch.Tensor):
                    if end_val.numel() == 1:
                        end = end_val.item()
                    else:
                        end = int(end_val.flatten()[0].item())
                else:
                    end = int(end_val)
                
                # Skip special tokens (CLS, SEP, PAD)
                if start == end == 0:
                    continue
                
                # Clamp indices to valid range
                start = max(0, int(start))
                end = min(orig_len, int(end))
                
                if start < end:
                    # Copy the token feature vector to all bases covered by this token
                    token_feat = token_features[token_idx:token_idx+1]  # (1, 768)
                    base_features[start:end] = token_feat.expand(end - start, self.hidden_size)
                    
                    # One-hot encode each base position
                    if seq is not None and len(seq) >= end:
                        for pos in range(start, end):
                            base_char = seq[pos].upper() if pos < len(seq) else 'A'
                            # Default to A if not in ATGC
                            onehot = self.base_to_onehot.get(base_char, self.base_to_onehot['A'])
                            onehot_features[pos] = onehot.to(device=hidden_states.device)
            
            # Fuse DNABERT features and one-hot
            # base_features: (orig_len, 768), onehot_features: (orig_len, 4)
            # enhanced_features: (orig_len, 772)
            enhanced_features = torch.cat([base_features, onehot_features], dim=1)
            
            # Pad to max_length (uniform length within the batch)
            if orig_len < max_length:
                padding = torch.zeros(
                    max_length - orig_len, self.enhanced_hidden_size,
                    device=hidden_states.device,
                    dtype=enhanced_features.dtype
                )
                enhanced_features = torch.cat([enhanced_features, padding], dim=0)
            
            base_features_list.append(enhanced_features)
        
        return torch.stack(base_features_list)  # (B, max_length, 772)
    
    def _expand_mask_to_bases_with_offset(self, token_mask, offset_mapping, original_seq_length):
        """
        Expand a token-level mask to base level via offset_mapping.
        
        Args:
            token_mask: (B, L_token) - token-level mask
            offset_mapping: List of (L_token, 2) tensors
            original_seq_length: List of int
        
        Returns:
            base_mask: (B, max_length) - base-level mask
        """
        batch_size = token_mask.shape[0]
        
        # Handle original_seq_length
        if isinstance(original_seq_length, list):
            max_length = max(original_seq_length)
        else:
            max_length = original_seq_length if original_seq_length is not None else self.max_original_length
        
        base_masks = []
        for i in range(batch_size):
            offsets = offset_mapping[i]
            token_mask_i = token_mask[i]
            
            # Handle original_seq_length
            if isinstance(original_seq_length, list):
                orig_len = original_seq_length[i]
            else:
                orig_len = original_seq_length if original_seq_length is not None else self.max_original_length
            
            base_mask = torch.zeros(orig_len, device=token_mask.device, dtype=token_mask.dtype)
            
            # Skip if offsets are missing
            if offsets is None:
                continue
            
            # Ensure offsets is a tensor
            if not isinstance(offsets, torch.Tensor):
                offsets = torch.tensor(offsets, device=token_mask.device)
            
            # Ensure offsets shape is (L_token, 2)
            if offsets.ndim != 2 or offsets.shape[1] != 2:
                if offsets.numel() > 0:
                    try:
                        offsets = offsets.view(-1, 2)
                    except:
                        continue
                else:
                    continue
            
            num_tokens = offsets.shape[0]
            for token_idx in range(num_tokens):
                # Read start/end
                start_val = offsets[token_idx, 0]
                end_val = offsets[token_idx, 1]
                
                # Convert to Python scalars
                if isinstance(start_val, torch.Tensor):
                    if start_val.numel() == 1:
                        start = start_val.item()
                    else:
                        start = int(start_val.flatten()[0].item())
                else:
                    start = int(start_val)
                
                if isinstance(end_val, torch.Tensor):
                    if end_val.numel() == 1:
                        end = end_val.item()
                    else:
                        end = int(end_val.flatten()[0].item())
                else:
                    end = int(end_val)
                
                if start == end == 0:
                    continue
                
                start = max(0, int(start))
                end = min(orig_len, int(end))
                
                if start < end and token_mask_i[token_idx] > 0:
                    base_mask[start:end] = 1.0
            
            if orig_len < max_length:
                padding = torch.zeros(max_length - orig_len, device=token_mask.device)
                base_mask = torch.cat([base_mask, padding])
            
            base_masks.append(base_mask)
        
        return torch.stack(base_masks)  # (B, max_length)
