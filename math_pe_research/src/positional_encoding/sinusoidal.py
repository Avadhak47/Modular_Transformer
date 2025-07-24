"""
Sinusoidal Positional Encoding Implementation
Enhanced for Mathematical Reasoning Tasks
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding with enhancements for mathematical reasoning.
    
    Features:
    - Standard sin/cos positional encoding
    - Learnable scaling factor
    - Mathematical sequence optimization
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        learnable_scale: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Create positional encoding table
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(base) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Learnable scaling
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('scale', torch.ones(1))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply sinusoidal positional encoding.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            position_ids: Optional position IDs
        
        Returns:
            Tensor with positional encoding added
        """
        batch_size, seq_len, d_model = x.shape
        
        if position_ids is not None:
            # Use provided position IDs
            pe = self.pe.squeeze(0)[position_ids]
        else:
            # Handle sequences longer than max_seq_len
            if seq_len > self.max_seq_len:
                # Generate positional encoding on-the-fly for longer sequences
                positions = torch.arange(seq_len, device=x.device, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, d_model, 2, device=x.device).float() * -(math.log(self.base) / d_model)
                )
                pe = torch.zeros(seq_len, d_model, device=x.device)
                pe[:, 0::2] = torch.sin(positions * div_term)
                pe[:, 1::2] = torch.cos(positions * div_term)
                pe = pe.unsqueeze(0)  # Add batch dimension
            else:
                # Use standard sequential positions
                pe = self.pe[:, :seq_len, :]
        
        # Apply scaling and add to input
        return self.dropout(x + self.scale.to(x.device) * pe)