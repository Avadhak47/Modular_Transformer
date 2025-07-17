"""
T5-Style Relative Positional Bias

Implements T5's relative positional bias mechanism optimized for mathematical reasoning.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class T5RelativePositionalBias(nn.Module):
    """T5-style relative positional bias for mathematical reasoning."""
    
    def __init__(
        self,
        num_heads: int,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        bidirectional: bool = True,
        math_enhanced: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.bidirectional = bidirectional
        self.math_enhanced = math_enhanced
        
        # Relative position bias table
        self.relative_attention_bias = nn.Embedding(
            relative_attention_num_buckets, num_heads
        )
        
        # Mathematical enhancement
        if math_enhanced:
            self.math_bias_scale = nn.Parameter(torch.ones(num_heads))
            self.math_pattern_bias = nn.Embedding(10, num_heads)  # 10 math patterns
    
    def _relative_position_bucket(
        self,
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128
    ) -> torch.Tensor:
        """Compute relative position buckets."""
        relative_buckets = 0
        
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        
        # Half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        # The other half are for logarithmically bigger bins
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        
        return relative_buckets
    
    def forward(
        self,
        attention_scores: torch.Tensor,
        query_length: Optional[int] = None,
        key_length: Optional[int] = None
    ) -> torch.Tensor:
        """Apply T5 relative positional bias."""
        if query_length is None:
            query_length = attention_scores.size(-2)
        if key_length is None:
            key_length = attention_scores.size(-1)
        
        device = attention_scores.device
        
        # Create position matrices
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        
        # Compute relative positions
        relative_position = memory_position - context_position
        
        # Get relative position buckets
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance
        )
        
        # Get bias values
        bias = self.relative_attention_bias(relative_position_bucket)
        bias = bias.permute([2, 0, 1]).unsqueeze(0)  # [1, num_heads, query_length, key_length]
        
        # Apply mathematical enhancement
        if self.math_enhanced:
            # Scale bias for mathematical reasoning
            bias = bias * self.math_bias_scale.view(1, -1, 1, 1)
            
            # Add pattern-specific bias (simplified)
            pattern_ids = torch.zeros(query_length, key_length, dtype=torch.long, device=device)
            pattern_bias = self.math_pattern_bias(pattern_ids)
            pattern_bias = pattern_bias.permute([2, 0, 1]).unsqueeze(0)
            bias = bias + pattern_bias
        
        return attention_scores + bias