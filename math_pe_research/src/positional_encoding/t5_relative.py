"""
T5 Relative Positional Encoding Implementation
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class T5RelativePositionalBias(nn.Module):
    """
    T5-style relative positional encoding.
    
    Features:
    - Relative position biases
    - Learned embeddings for relative distances
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        bidirectional: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.bidirectional = bidirectional
        
        self.relative_attention_bias = nn.Embedding(relative_attention_num_buckets, num_heads)
    
    def _relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe7b1/mesh_tensorflow/transformer/transformer_layers.py#L593
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        
        # now relative_position is in the range [0, inf)
        
        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
    
    def forward(
        self,
        x: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Apply T5 relative positional bias."""
        if attention_scores is not None:
            batch_size, num_heads, seq_len_q, seq_len_k = attention_scores.shape
            
            query_position = torch.arange(seq_len_q, dtype=torch.long, device=x.device)[:, None]
            key_position = torch.arange(seq_len_k, dtype=torch.long, device=x.device)[None, :]
            relative_position = key_position - query_position  # shape (seq_len_q, seq_len_k)
            
            relative_position_bucket = self._relative_position_bucket(
                relative_position,
                bidirectional=self.bidirectional,
                num_buckets=self.relative_attention_num_buckets,
                max_distance=self.relative_attention_max_distance,
            )
            
            values = self.relative_attention_bias(relative_position_bucket)  # shape (seq_len_q, seq_len_k, num_heads)
            values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, seq_len_q, seq_len_k)
            
            return attention_scores + values
        
        # If no attention scores provided, just return input
        return x