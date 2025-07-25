"""
ALiBi (Attention with Linear Biases) Positional Encoding
Enhanced for Mathematical Reasoning Tasks
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class ALiBiPositionalEncoding(nn.Module):
    """
    ALiBi (Attention with Linear Biases) positional encoding.
    
    Features:
    - Linear bias applied to attention scores
    - No explicit positional embeddings
    - Excellent extrapolation properties
    - Mathematical reasoning optimizations
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        max_seq_len: int = 8192,
        alibi_bias_max: int = 8,
        learnable_slopes: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.alibi_bias_max = alibi_bias_max
        
        # Calculate ALiBi slopes
        slopes = self._get_alibi_slopes(num_heads)
        
        if learnable_slopes:
            self.slopes = nn.Parameter(slopes)
        else:
            self.register_buffer('slopes', slopes)
        
        # Pre-compute bias matrix for efficiency
        self._build_alibi_bias()
    
    def _get_alibi_slopes(self, num_heads: int) -> torch.Tensor:
        """Calculate ALiBi slopes for each attention head."""
        
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            return torch.tensor(get_slopes_power_of_2(num_heads))
        else:
            # Handle non-power-of-2 head counts
            closest_power_of_2 = 2**math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            
            # Add remaining slopes
            extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)
            slopes.extend(extra_slopes[0:num_heads - closest_power_of_2])
            
            return torch.tensor(slopes[:num_heads])
    
    def _build_alibi_bias(self):
        """Pre-compute ALiBi bias matrix."""
        # Create position matrix
        context_position = torch.arange(self.max_seq_len)[:, None]
        memory_position = torch.arange(self.max_seq_len)[None, :]
        
        # Calculate relative distances
        relative_position = memory_position - context_position
        
        # Apply ALiBi biases (only for positions where memory <= context)
        alibi_bias = torch.zeros((self.max_seq_len, self.max_seq_len))
        
        # Create causal mask and apply bias
        causal_mask = relative_position <= 0
        alibi_bias[causal_mask] = relative_position[causal_mask].float()
        
        # Add head dimension
        alibi_bias = alibi_bias.unsqueeze(0)  # (1, seq_len, seq_len)
        
        self.register_buffer('alibi_bias', alibi_bias)
    
    def get_bias(self, seq_len: int) -> torch.Tensor:
        """Get ALiBi bias for given sequence length."""
        if seq_len > self.max_seq_len:
            # Extend bias matrix if needed
            self._extend_bias(seq_len)
        
        # Get bias for current sequence length
        bias = self.alibi_bias[:, :seq_len, :seq_len]  # Shape: (1, seq_len, seq_len)
        
        # Apply slopes for each head
        head_biases = []
        for slope in self.slopes:
            head_biases.append(bias.squeeze(0) * slope)  # Remove extra dimension and apply slope
        
        # Stack head biases: (num_heads, seq_len, seq_len)
        return torch.stack(head_biases, dim=0)
    
    def _extend_bias(self, new_max_len: int):
        """Extend bias matrix for longer sequences."""
        self.max_seq_len = new_max_len
        self._build_alibi_bias()
    
    def forward(
        self, 
        x: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply ALiBi bias. 
        
        Note: ALiBi is typically applied directly to attention scores,
        not to input embeddings. This method returns the input unchanged
        and provides the bias for attention computation.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            attention_scores: Optional attention scores to apply bias to
        
        Returns:
            Input tensor (unchanged) or attention scores with bias applied
        """
        if attention_scores is not None:
            batch_size, num_heads, seq_len_q, seq_len_k = attention_scores.shape
            bias = self.get_bias(max(seq_len_q, seq_len_k))
            # Ensure bias shape matches attention_scores
            if bias.shape[-2:] != (seq_len_q, seq_len_k):
                bias = bias[:, :seq_len_q, :seq_len_k]
            return attention_scores + bias[:num_heads]
        
        # If no attention scores provided, just return input
        return x
    
    def get_alibi_slopes(self) -> torch.Tensor:
        """Get the ALiBi slopes for analysis."""
        return self.slopes

    def to(self, device):
        super().to(device)
        if hasattr(self, 'slopes'):
            self.slopes = self.slopes.to(device)
        if hasattr(self, 'alibi_bias'):
            self.alibi_bias = self.alibi_bias.to(device)
        return self


class MathematicalALiBi(ALiBiPositionalEncoding):
    """
    ALiBi variant optimized for mathematical reasoning.
    
    Features:
    - Adaptive slopes based on mathematical content
    - Enhanced bias patterns for mathematical structures
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Mathematical content adaptive parameters
        self.math_slope_adjustment = nn.Parameter(torch.ones(self.num_heads))
        self.operator_bias_scale = nn.Parameter(torch.ones(1))
        
    def forward(
        self, 
        x: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Enhanced forward pass with mathematical adaptations."""
        
        if attention_scores is not None:
            # Get base bias
            batch_size, num_heads, seq_len_q, seq_len_k = attention_scores.shape
            bias = self.get_bias(max(seq_len_q, seq_len_k))
            
            # Apply mathematical adaptations
            device = attention_scores.device
            adjusted_slopes = self.slopes.to(device) * self.math_slope_adjustment.to(device)
            
            # Recalculate bias with adjusted slopes
            math_bias = []
            for slope in adjusted_slopes:
                math_bias.append(self.alibi_bias.to(device)[:, :seq_len_q, :seq_len_k] * slope)
            
            math_bias = torch.stack(math_bias, dim=0)
            
            # Apply operator-specific scaling if token_ids provided
            if token_ids is not None:
                operator_mask = self._identify_operators(token_ids)
                if operator_mask.any():
                    operator_bias = math_bias * self.operator_bias_scale.to(device)
                    # Apply operator bias where operators are present
                    # This is a simplified implementation
                    math_bias = math_bias + 0.1 * operator_bias
            
            return attention_scores + math_bias[:num_heads]
        
        return x
    
    def _identify_operators(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Identify mathematical operators in token sequence."""
        # Simplified operator identification
        # In practice, this would use tokenizer vocabulary mapping
        operator_tokens = torch.tensor([43, 45, 42, 47, 61], device=token_ids.device)  # +, -, *, /, =
        
        operator_mask = torch.isin(token_ids, operator_tokens)
        return operator_mask

    def to(self, device):
        super().to(device)
        if hasattr(self, 'math_slope_adjustment'):
            self.math_slope_adjustment = self.math_slope_adjustment.to(device)
        if hasattr(self, 'operator_bias_scale'):
            self.operator_bias_scale = self.operator_bias_scale.to(device)
        return self


# Factory function
def create_alibi_encoding(variant: str = "standard", **kwargs) -> ALiBiPositionalEncoding:
    """Create ALiBi encoding variant."""
    if variant == "standard":
        return ALiBiPositionalEncoding(**kwargs)
    elif variant == "mathematical":
        return MathematicalALiBi(**kwargs)
    else:
        raise ValueError(f"Unknown ALiBi variant: {variant}")


if __name__ == "__main__":
    # Test ALiBi implementation
    d_model = 512
    num_heads = 8
    seq_len = 128
    batch_size = 2
    
    # Create ALiBi encoder
    alibi = ALiBiPositionalEncoding(d_model=d_model, num_heads=num_heads)
    
    # Test input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test without attention scores (should return input unchanged)
    output = alibi(x)
    print(f"Input unchanged test: {torch.equal(x, output)}")
    
    # Test with attention scores
    attention_scores = torch.randn(batch_size, num_heads, seq_len, seq_len)
    biased_scores = alibi(x, attention_scores=attention_scores)
    
    print(f"Attention scores shape: {attention_scores.shape}")
    print(f"Biased scores shape: {biased_scores.shape}")
    print(f"Bias applied: {not torch.equal(attention_scores, biased_scores)}")
    
    # Test mathematical variant
    math_alibi = MathematicalALiBi(d_model=d_model, num_heads=num_heads)
    token_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    math_biased_scores = math_alibi(x, attention_scores=attention_scores, token_ids=token_ids)
    print(f"Mathematical ALiBi test passed: {math_biased_scores.shape == attention_scores.shape}")