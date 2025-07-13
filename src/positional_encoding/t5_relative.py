"""
T5-style relative positional encoding implementation.
"""
import torch
import torch.nn as nn
import math
from .base import BasePositionalEncoding


class T5RelativePositionalEncoding(BasePositionalEncoding):
    """
    T5-style relative positional encoding with bucketed distances.
    """
    
    def __init__(self, d_model: int, n_heads: int, max_len: int = 5000, 
                 dropout: float = 0.1, num_buckets: int = 32, max_distance: int = 128):
        super().__init__(d_model, max_len, dropout = float(0.1))
        self.n_heads = n_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        
        # Relative position embeddings
        self.relative_attention_bias = nn.Embedding(num_buckets, n_heads)
        
        # Initialize embeddings
        nn.init.normal_(self.relative_attention_bias.weight, 0, 0.02)
    
    def _relative_position_bucket(self, relative_position: torch.Tensor, 
                                 bidirectional: bool = True) -> torch.Tensor:
        """Convert relative positions to bucket indices."""
        relative_buckets = 0
        
        if bidirectional:
            num_buckets = self.num_buckets // 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            num_buckets = self.num_buckets
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        
        # Half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        # The other half of the buckets are for logarithmically bigger bins
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact) / 
            math.log(self.max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        
        relative_position_if_large = torch.min(
            relative_position_if_large, 
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply T5-style relative positional encoding."""
        return self.dropout(x)
    
    def get_bidirectional_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate bidirectional relative position bias."""
        # Create relative position matrix
        context_position = torch.arange(seq_len, device=device)[:, None]
        memory_position = torch.arange(seq_len, device=device)[None, :]
        relative_position = memory_position - context_position
        
        # Convert to buckets
        relative_position_bucket = self._relative_position_bucket(relative_position, bidirectional=True)
        
        # Get bias values
        bias = self.relative_attention_bias(relative_position_bucket)
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, n_heads, seq_len, seq_len)
        
        return bias
    
    def get_encoding_type(self) -> str:
        return "t5_relative"
    
    def get_info(self) -> dict:
        return {
            "type": "T5-relative",
            "n_heads": self.n_heads,
            "num_buckets": self.num_buckets,
            "max_distance": self.max_distance
        }
