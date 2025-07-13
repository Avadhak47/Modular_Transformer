"""
DIET (Decoupled Positional Attention) implementation.
"""
import torch
import torch.nn as nn
import math
from .base import BasePositionalEncoding


class DIETPositionalEncoding(BasePositionalEncoding):
    """
    DIET (Decoupled Positional Attention) positional encoding.
    Each attention head gets its own positional encoding parameters.
    """
    
    def __init__(self, d_model: int, n_heads: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__(d_model, max_len, dropout)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Per-head positional embeddings
        self.head_pos_embeddings = nn.ModuleList([
            nn.Embedding(max_len, self.d_head) for _ in range(n_heads)
        ])
        
        # Initialize embeddings
        for embedding in self.head_pos_embeddings:
            nn.init.normal_(embedding.weight, 0, 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DIET positional encoding."""
        batch_size, seq_len, d_model = x.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Reshape input for per-head processing
        x = x.view(batch_size, seq_len, self.n_heads, self.d_head)
        
        # Apply per-head positional encodings
        for head_idx in range(self.n_heads):
            head_pos = self.head_pos_embeddings[head_idx](positions)
            x[:, :, head_idx, :] += head_pos
        
        # Reshape back to original form
        x = x.view(batch_size, seq_len, d_model)
        
        return self.dropout(x)
    
    def get_encoding_type(self) -> str:
        return "diet"
    
    def get_info(self) -> dict:
        return {
            "type": "DIET",
            "n_heads": self.n_heads,
            "d_head": self.d_head,
            "max_len": self.max_len
        }
