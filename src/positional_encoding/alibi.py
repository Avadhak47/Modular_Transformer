"""
ALiBi (Attention with Linear Biases) positional encoding implementation.
"""
import torch
import torch.nn as nn
import math
from typing import Optional
from .base import BasePositionalEncoding


class ALiBiPositionalEncoding(BasePositionalEncoding):
    """
    ALiBi (Attention with Linear Biases) positional encoding.
    Adds linear biases to attention scores based on token distance.
    """
    
    def __init__(self, d_model: int, n_heads: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize ALiBi positional encoding.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            max_len: Maximum sequence length (not strictly enforced)
            dropout: Dropout rate
        """
        super().__init__(d_model, max_len, dropout)
        self.n_heads = n_heads
        
        # Calculate slopes for each attention head
        slopes = self._get_slopes(n_heads)
        self.register_buffer('slopes', slopes)
    
    def _get_slopes(self, n_heads: int) -> torch.Tensor:
        """
        Calculate slopes for ALiBi bias based on number of heads.
        
        Args:
            n_heads: Number of attention heads
            
        Returns:
            Tensor of slopes for each head
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return [start * ratio ** i for i in range(n)]
        
        # Handle cases where n_heads is not a power of 2
        if math.log2(n_heads).is_integer():
            slopes = get_slopes_power_of_2(n_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            
            # Add extra slopes for remaining heads
            extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)
            slopes += extra_slopes[0::2][:n_heads - closest_power_of_2]
        
        return torch.tensor(slopes, dtype=torch.float32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ALiBi doesn't modify input embeddings directly.
        The bias is applied in the attention mechanism.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Input tensor with dropout applied (unchanged otherwise)
        """
        return self.dropout(x)
    
    def get_alibi_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate ALiBi bias matrix for attention scores.
        
        Args:
            seq_len: Sequence length
            device: Device to place the tensor on
            
        Returns:
            Bias tensor of shape (n_heads, seq_len, seq_len)
        """
        # Create position indices
        positions = torch.arange(seq_len, device=device)
        
        # Calculate distances between all pairs of positions
        distances = positions.unsqueeze(0) - positions.unsqueeze(1)
        distances = distances.abs().float()
        
        # Apply slopes to distances (negative because we subtract from attention scores)
        bias = -distances.unsqueeze(0) * torch.as_tensor(self.slopes).unsqueeze(1).unsqueeze(2).to(device)
        
        return bias
    
    def get_causal_alibi_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal ALiBi bias matrix (for decoder self-attention).
        
        Args:
            seq_len: Sequence length
            device: Device to place the tensor on
            
        Returns:
            Causal bias tensor of shape (n_heads, seq_len, seq_len)
        """
        # Create position indices
        positions = torch.arange(seq_len, device=device)
        
        # Calculate relative positions (only past positions)
        rel_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # Create causal mask (future positions get large negative values)
        causal_mask = (rel_positions > 0).float() * -1e9
        
        # Apply slopes to past positions only
        bias = torch.where(
            rel_positions <= 0,
            -rel_positions.abs().float().unsqueeze(0) * torch.as_tensor(self.slopes).unsqueeze(1).unsqueeze(2).to(device),
            causal_mask.unsqueeze(0).to(device)
        )
        
        return bias
    
    def get_encoding_type(self) -> str:
        """Return the encoding type name."""
        return "alibi"
    
    def can_extrapolate(self) -> bool:
        """ALiBi can extrapolate to longer sequences."""
        return True
    
    def get_slopes_info(self) -> dict:
        """Get information about the slopes used."""
        return {
            "n_heads": self.n_heads,
            "slopes": torch.as_tensor(self.slopes).tolist(),
            "min_slope": torch.as_tensor(self.slopes).min().item(),
            "max_slope": torch.as_tensor(self.slopes).max().item()
        }
    
    def apply_bias_to_attention(self, attention_scores: torch.Tensor, causal: bool = False) -> torch.Tensor:
        """
        Apply ALiBi bias directly to attention scores.
        
        Args:
            attention_scores: Attention scores of shape (batch_size, n_heads, seq_len, seq_len)
            causal: Whether to apply causal masking
            
        Returns:
            Attention scores with ALiBi bias applied
        """
        batch_size, n_heads, seq_len, _ = attention_scores.shape
        device = attention_scores.device
        
        # Verify number of heads matches
        assert n_heads == self.n_heads, f"Expected {self.n_heads} heads, got {n_heads}"
        
        # Get appropriate bias
        if causal:
            bias = self.get_causal_alibi_bias(seq_len, device)
        else:
            bias = self.get_alibi_bias(seq_len, device)
        
        # Add bias to attention scores
        return attention_scores + bias.unsqueeze(0)
    
    def extend_to_length(self, seq_len: int, device: torch.device, causal: bool = False) -> torch.Tensor:
        """
        Generate ALiBi bias for any sequence length (supports extrapolation).
        
        Args:
            seq_len: Target sequence length
            device: Device for the tensor
            causal: Whether to use causal masking
            
        Returns:
            ALiBi bias tensor
        """
        if causal:
            return self.get_causal_alibi_bias(seq_len, device)
        else:
            return self.get_alibi_bias(seq_len, device)