"""
Multi-head attention implementation with positional encoding support.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism with support for different positional encodings."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                pos_encoding=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional positional encoding support."""
        batch_size = query.size(0)
        q_len = query.size(1)
        k_len = key.size(1)
        v_len = value.size(1)

        # Linear projections and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, q_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, k_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, v_len, self.n_heads, self.d_k).transpose(1, 2)

        # Apply RoPE if specified
        if pos_encoding and hasattr(pos_encoding, 'get_encoding_type') and pos_encoding.get_encoding_type() == 'rope':
            Q_rope = Q.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
            K_rope = K.transpose(1, 2).contiguous().view(batch_size, k_len, self.d_model)
            Q_rope, K_rope = pos_encoding.apply_to_query_key(Q_rope, K_rope)
            Q = Q_rope.view(batch_size, q_len, self.n_heads, self.d_k).transpose(1, 2)
            K = K_rope.view(batch_size, k_len, self.n_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply positional encoding bias if ALiBi or T5-relative
        if pos_encoding:
            if hasattr(pos_encoding, 'get_encoding_type'):
                if pos_encoding.get_encoding_type() == 'alibi':
                    alibi_bias = pos_encoding.get_alibi_bias(k_len, query.device)
                    # Ensure scores and alibi_bias have matching (q_len, k_len) for cross-attention
                    if (scores.shape[-2:] != alibi_bias.shape[-2:]):
                        min_q = min(scores.shape[-2], alibi_bias.shape[-2])
                        min_k = min(scores.shape[-1], alibi_bias.shape[-1])
                        scores = scores[..., :min_q, :min_k]
                        alibi_bias = alibi_bias[..., :min_q, :min_k]
                    scores += alibi_bias.unsqueeze(0)
                elif pos_encoding.get_encoding_type() == 't5_relative':
                    relative_bias = pos_encoding.get_bidirectional_bias(k_len, query.device)
                    # Ensure scores and relative_bias have matching (q_len, k_len) for cross-attention/generation
                    if (scores.shape[-2:] != relative_bias.shape[-2:]):
                        min_q = min(scores.shape[-2], relative_bias.shape[-2])
                        min_k = min(scores.shape[-1], relative_bias.shape[-1])
                        scores = scores[..., :min_q, :min_k]
                        relative_bias = relative_bias[..., :min_q, :min_k]
                    scores += relative_bias

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(
            batch_size, q_len, self.d_model)

        output = self.w_o(context)

        return output, attention_weights


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention mechanism."""
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for scaled dot-product attention."""
        d_k = query.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, value)
        
        return context, attention_weights
