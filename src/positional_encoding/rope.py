"""
Rotary Positional Encoding (RoPE) implementation.
"""
import torch
import torch.nn as nn
import math
from typing import Tuple
from .base import BasePositionalEncoding


class RotaryPositionalEncoding(BasePositionalEncoding):
    """
    Rotary Positional Encoding (RoPE) as described in RoFormer.
    Encodes position information using rotation matrices applied to query and key vectors.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1, base: float = 10000.0):
        """
        Initialize RoPE positional encoding.
        
        Args:
            d_model: Model dimension (must be even)
            max_len: Maximum sequence length
            dropout: Dropout rate
            base: Base for frequency computation
        """
        super().__init__(d_model, max_len, dropout)
        self.base = base
        
        # RoPE requires even dimension for pairing
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with rotary positional encoding applied
        """
        seq_len = x.size(1)
        
        # Generate position indices
        position = torch.arange(seq_len, device=x.device, dtype=torch.float)
        
        # Calculate frequencies for each position
        freqs = torch.einsum('i,j->ij', position, self.inv_freq)
        
        # Get cos and sin values
        cos_vals = freqs.cos()
        sin_vals = freqs.sin()
        
        # Apply rotation
        x_rotated = self._apply_rotary_pos_emb(x, cos_vals, sin_vals)
        
        return self.dropout(x_rotated)
    
    def _apply_rotary_pos_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embedding to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            cos: Cosine values of shape (seq_len, d_model//2)
            sin: Sine values of shape (seq_len, d_model//2)
            
        Returns:
            Rotated tensor of same shape as input
        """
        # Reshape input to work with rotation
        batch_size, seq_len, d_model = x.shape
        x = x.view(batch_size, seq_len, d_model // 2, 2)
        
        # Extract x1 and x2 (real and imaginary parts)
        x1 = x[..., 0]  # (batch_size, seq_len, d_model//2)
        x2 = x[..., 1]  # (batch_size, seq_len, d_model//2)
        
        # Apply rotation: [cos -sin; sin cos] * [x1; x2]
        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x1 * sin + x2 * cos
        
        # Recombine
        x_rotated = torch.stack([x1_rotated, x2_rotated], dim=-1)
        
        return x_rotated.view(batch_size, seq_len, d_model)
    
    def get_cos_sin(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cosine and sine values for a given sequence length.
        
        Args:
            seq_len: Sequence length
            device: Device to place tensors on
            
        Returns:
            Tuple of (cos, sin) tensors
        """
        position = torch.arange(seq_len, device=device, dtype=torch.float)
        freqs = torch.einsum('i,j->ij', position, self.inv_freq.to(device))
        
        cos_vals = freqs.cos()
        sin_vals = freqs.sin()
        
        return cos_vals, sin_vals
    
    def get_encoding_type(self) -> str:
        """Return the encoding type name."""
        return "rope"
    
    def can_extrapolate(self) -> bool:
        """RoPE can extrapolate to longer sequences."""
        return True
    
    def apply_to_query_key(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors separately.
        This is the typical usage in attention mechanisms.
        
        Args:
            q: Query tensor of shape (batch_size, seq_len, d_model)
            k: Key tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tuple of rotated (q, k) tensors
        """
        assert q.size() == k.size(), "Query and key must have the same shape"
        
        seq_len = q.size(1)
        device = q.device
        
        # Get cos and sin values
        cos_vals, sin_vals = self.get_cos_sin(seq_len, device)
        
        # Apply rotation to both query and key
        q_rotated = self._apply_rotary_pos_emb(q, cos_vals, sin_vals)
        k_rotated = self._apply_rotary_pos_emb(k, cos_vals, sin_vals)
        
        return q_rotated, k_rotated