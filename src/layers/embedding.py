"""
Token and positional embedding implementations.
"""
import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """Token embedding with optional scaling."""
    
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, 0, 0.1)
        if padding_idx is not None:
            nn.init.constant_(self.embedding.weight[padding_idx], 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with scaling."""
        return self.embedding(x) * math.sqrt(self.d_model)


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings as an alternative to sinusoidal."""
    
    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        
        # Initialize positional embeddings
        nn.init.normal_(self.pos_embedding.weight, 0, 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned positional embeddings."""
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_len}")
        
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_embeddings = self.pos_embedding(positions)
        return self.dropout(x + pos_embeddings)
