"""
Deep Learning Architectures for Limit Order Books (Section 2.3 & Table 2).
Implements:
1. DeepLOB Spatial CNN (Zhang et al. 2019)
2. DeepLOB Spatiotemporal CNN + LSTM (Zhang et al. 2019)
3. LiT: Limit Order Book Transformer (Li et al. 2025)
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLOBSpatialCNN(nn.Module):
    """
    Spatial Convolutional Neural Network treating LOB snapshots as 2D spatial maps.
    Input shape: (batch_size, 1, seq_len, num_features)
    """
    def __init__(self, in_features: int = 40, seq_len: int = 20, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(4, 1), padding="same")
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(1, 2), stride=(1, 2))
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(4, 1), padding="same")
        
        self.dropout = nn.Dropout(dropout)
        
        # Calculate flattened dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, seq_len, in_features)
            out = F.leaky_relu(self.conv1(dummy))
            out = F.leaky_relu(self.conv2(out))
            out = F.leaky_relu(self.conv3(out))
            out = F.leaky_relu(self.conv4(out))
            flat_dim = out.view(1, -1).size(1)
            
        self.fc1 = nn.Linear(flat_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, seq_len, features) -> reshape to (B, 1, seq_len, features)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        out = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        out = F.leaky_relu(self.conv2(out), negative_slope=0.01)
        out = F.leaky_relu(self.conv3(out), negative_slope=0.01)
        out = F.leaky_relu(self.conv4(out), negative_slope=0.01)
        
        out = out.view(out.size(0), -1)
        out = self.dropout(F.leaky_relu(self.fc1(out), negative_slope=0.01))
        logits = self.fc2(out)
        return logits


class DeepLOBSpatiotemporal(nn.Module):
    """
    DeepLOB Spatiotemporal Architecture:
    Spatial Convolution -> Temporal Convolution -> Inception Modules -> LSTM -> Fully Connected.
    """
    def __init__(self, in_features: int = 40, seq_len: int = 20, num_classes: int = 3,
                 lstm_hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(4, 1), padding="same")
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(1, 2), stride=(1, 2))
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(4, 1), padding="same")
        
        # Spatial compression
        with torch.no_grad():
            dummy = torch.zeros(1, 1, seq_len, in_features)
            out = F.leaky_relu(self.conv1(dummy))
            out = F.leaky_relu(self.conv2(out))
            out = F.leaky_relu(self.conv3(out))
            out = F.leaky_relu(self.conv4(out))
            # Shape: (B, 32, seq_len, reduced_features)
            b, c, t, f = out.shape
            lstm_in_dim = c * f

        self.lstm_in_dim = lstm_in_dim
        self.lstm = nn.LSTM(
            input_size=lstm_in_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        B, _, T, _ = x.shape
        
        out = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        out = F.leaky_relu(self.conv2(out), negative_slope=0.01)
        out = F.leaky_relu(self.conv3(out), negative_slope=0.01)
        out = F.leaky_relu(self.conv4(out), negative_slope=0.01)
        
        # Permute and reshape for LSTM: (B, T, C*F)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, -1)
        lstm_out, (hn, cn) = self.lstm(out)
        
        last_step = lstm_out[:, -1, :] # Last time step
        last_step = self.dropout(last_step)
        logits = self.fc(last_step)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x


class LiTTransformer(nn.Module):
    """
    Limit Order Book Transformer (LiT / TransLOB).
    Feature projection -> Positional encoding -> Multi-Head Self-Attention layers -> Classification head.
    """
    def __init__(
        self,
        in_features: int = 40,
        seq_len: int = 20,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=seq_len + 50)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, seq_len, features)
        if x.dim() == 4:
            x = x.squeeze(1)
            
        proj = self.input_proj(x)
        proj = self.pos_encoder(proj)
        encoded = self.transformer_encoder(proj)
        
        # Temporal pooling (mean pooling over time dimension)
        pooled = torch.mean(encoded, dim=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits
