"""
DIET (Dynamic Positional Encoding) Implementation
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class DIETPositionalEncoding(nn.Module):
    """
    DIET (Dynamic Positional Encoding) for mathematical reasoning.
    
    Features:
    - Dynamic position adjustments
    - Content-aware encoding
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 8192,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Base sinusoidal encoding
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Dynamic adjustment layers
        self.position_transform = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        device = x.device
        # Generate base sinusoidal encoding
        if seq_len > self.max_seq_len:
            pe = self._generate_base_encoding(seq_len, d_model, device)
        else:
            pe = self.base_encoding[:, :seq_len, :].to(device)
        # Ensure pe matches x's last dimension
        if pe.shape[-1] != d_model:
            if d_model % pe.shape[-1] == 0:
                repeat_factor = d_model // pe.shape[-1]
                pe = pe.repeat(1, 1, repeat_factor)
            else:
                pe = pe.expand(batch_size, seq_len, d_model)
        # Dynamic transformation
        pe_dynamic = self.position_transform(pe)
        return self.dropout(x + pe_dynamic)

    def to(self, device):
        super().to(device)
        if hasattr(self, 'position_transform'):
            self.position_transform = self.position_transform.to(device)
        return self