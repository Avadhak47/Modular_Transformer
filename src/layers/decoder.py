"""
Transformer Decoder implementation.
"""
import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .feed_forward import PositionWiseFeedForward
from .layer_norm import LayerNorm


class TransformerDecoderLayer(nn.Module):
    """Single Transformer Decoder Layer."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Masked multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Multi-head cross-attention
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                pos_encoding=None) -> torch.Tensor:
        """Forward pass through decoder layer."""
        # Masked self-attention
        self_attn_output, _ = self.self_attention(x, x, x, tgt_mask, pos_encoding)
        x = self.norm1(x + self.dropout1(self_attn_output))
        
        # Cross-attention
        cross_attn_output, _ = self.cross_attention(
            x, encoder_output, encoder_output, src_mask, pos_encoding
        )
        x = self.norm2(x + self.dropout2(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x


class TransformerDecoder(nn.Module):
    """Stack of Transformer Decoder Layers."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 n_layers: int, dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                pos_encoding=None) -> torch.Tensor:
        """Forward pass through all decoder layers."""
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask, pos_encoding)
        
        return self.norm(x)
