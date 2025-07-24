"""
Sinusoidal positional encoding implementation.
"""
import torch
import torch.nn as nn
import math
from typing import Dict, Any
from .base import BasePositionalEncoding


class SinusoidalPositionalEncoding(BasePositionalEncoding):
    """
    Sinusoidal positional encoding as described in 'Attention is All You Need'.
    Uses sine and cosine functions of different frequencies to encode position.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize sinusoidal positional encoding.
        
        Args:
            d_model: Model dimension (must be even)
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__(d_model, max_len, dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate div_term for frequency scaling
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        # Apply sine to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices (1, 3, 5, ...)
        if d_model % 2 == 1:
            # If d_model is odd, handle the last dimension safely
            if div_term.numel() > 1:
                pe[:, 1:-1:2] = torch.cos(position * div_term[:-1])
            pe[:, -1] = torch.cos(position.squeeze(-1) * div_term[-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add sinusoidal positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added, same shape as input
        """
        seq_len = x.size(1)
        
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_len}")
        
        # Add positional encoding (broadcasting handles batch dimension)
        x = x + self.get_buffer('pe')[:, :seq_len, :]
        return self.dropout(x)
    
    def get_encoding_type(self) -> str:
        """Return the encoding type name."""
        return "sinusoidal"
    
    def can_extrapolate(self) -> bool:
        """Sinusoidal encoding doesn't naturally extrapolate beyond max_len."""
        return False
    
    def get_positional_encoding(self, seq_len: int) -> torch.Tensor:
        """
        Get the positional encoding for a specific sequence length.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Positional encoding tensor of shape (1, seq_len, d_model)
        """
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_len}")
        
        return self.get_buffer('pe')[:, :seq_len, :]
    
    def extend_max_len(self, new_max_len: int):
        """
        Extend the maximum sequence length by recomputing the positional encoding.
        
        Args:
            new_max_len: New maximum sequence length
        """
        if new_max_len <= self.max_len:
            return
        
        # Recompute positional encoding with new max length
        pe = torch.zeros(new_max_len, self.d_model)
        position = torch.arange(0, new_max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                           -(math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        
        if self.d_model % 2 == 1:
            if div_term.numel() > 1:
                pe[:, 1:-1:2] = torch.cos(position * div_term[:-1])
            pe[:, -1] = torch.cos(position.squeeze(-1) * div_term[-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.max_len = new_max_len