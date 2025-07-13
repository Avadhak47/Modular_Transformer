"""
Transformer Encoder implementation.
"""
import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .feed_forward import PositionWiseFeedForward
from .layer_norm import LayerNorm


class TransformerEncoderLayer(nn.Module):
    """Single Transformer Encoder Layer."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                pos_encoding=None) -> torch.Tensor:
        """Forward pass through encoder layer."""
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attention(x, x, x, mask, pos_encoding)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """Stack of Transformer Encoder Layers."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 n_layers: int, dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                pos_encoding=None) -> torch.Tensor:
        """Forward pass through all encoder layers."""
        for layer in self.layers:
            x = layer(x, mask, pos_encoding)
        
        return self.norm(x)
