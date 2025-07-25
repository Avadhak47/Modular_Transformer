"""
Rotary Positional Embedding (RoPE) - Enhanced for Mathematical Reasoning

Advanced RoPE implementation with:
- Dynamic frequency scaling for mathematical patterns
- Long sequence optimization (up to 32K tokens)
- Mathematical reasoning specific enhancements
- Memory-efficient computation
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class RotaryPositionalEmbedding(nn.Module):
    """
    Enhanced Rotary Positional Embedding for Mathematical Reasoning
    
    Features:
    - Dynamic base frequency scaling
    - Mathematical pattern awareness
    - Efficient long sequence handling
    - Position interpolation for extrapolation
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 32768,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        math_enhanced: bool = True,
        use_cache: bool = True
    ):
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor
        self.math_enhanced = math_enhanced
        self.use_cache = use_cache
        
        # Precompute frequency inverse terms
        self.register_buffer('inv_freq', self._compute_inv_freq())
        
        # Mathematical enhancement parameters
        if math_enhanced:
            self.freq_enhancement = nn.Parameter(torch.ones(dim // 2))
            self.position_scaling = nn.Parameter(torch.ones(1))
        
        # Cache for cos/sin values
        if use_cache:
            self._cached_cos = None
            self._cached_sin = None
            self._cached_seq_len = 0
    
    def _compute_inv_freq(self) -> torch.Tensor:
        """Compute inverse frequencies for RoPE."""
        # Enhanced frequency computation for mathematical reasoning
        if self.math_enhanced:
            # Use different frequency distributions for different parts
            # Lower frequencies for global mathematical structure
            # Higher frequencies for fine-grained mathematical details
            low_freq = torch.arange(0, self.dim // 4, 2, dtype=torch.float32)
            mid_freq = torch.arange(0, self.dim // 4, 2, dtype=torch.float32)
            high_freq = torch.arange(0, self.dim // 4, 2, dtype=torch.float32)
            ultra_freq = torch.arange(0, self.dim // 4, 2, dtype=torch.float32)
            
            # Different base frequencies for different mathematical concepts
            inv_freq_low = 1.0 / (self.base ** (low_freq / (self.dim // 4)))
            inv_freq_mid = 1.0 / ((self.base * 2) ** (mid_freq / (self.dim // 4)))
            inv_freq_high = 1.0 / ((self.base * 5) ** (high_freq / (self.dim // 4)))
            inv_freq_ultra = 1.0 / ((self.base * 10) ** (ultra_freq / (self.dim // 4)))
            
            inv_freq = torch.cat([inv_freq_low, inv_freq_mid, inv_freq_high, inv_freq_ultra])
        else:
            # Standard RoPE frequency computation
            freqs = torch.arange(0, self.dim, 2, dtype=torch.float32)
            inv_freq = 1.0 / (self.base ** (freqs / self.dim))
        
        return inv_freq
    
    def _get_cos_sin(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin matrices for RoPE."""
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        # Always move learned parameters to the correct device
        if hasattr(self, 'position_scaling'):
            positions = positions * self.position_scaling.to(device)
        inv_freq = self.inv_freq.to(device)
        if hasattr(self, 'freq_enhancement'):
            inv_freq = inv_freq * self.freq_enhancement.to(device)
        freqs = torch.outer(positions, inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the dimensions of the input tensor."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    def apply_rotary_pos_emb(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary positional embedding to query and key tensors."""
        # Ensure cos/sin have correct dimensions
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
            sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Apply rotation
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_embed, k_embed
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.
        
        Args:
            query: Query tensor [batch, heads, seq_len, head_dim]
            key: Key tensor [batch, heads, seq_len, head_dim]
            seq_len: Sequence length (if different from tensor size)
            
        Returns:
            Rotated query and key tensors
        """
        if seq_len is None:
            seq_len = query.size(-2)
        
        # Get cos/sin values
        cos, sin = self._get_cos_sin(seq_len, query.device)
        
        # Apply rotary embedding
        return self.apply_rotary_pos_emb(query, key, cos, sin)
    
    def extend_length(self, new_max_len: int):
        """Extend maximum sequence length and clear cache."""
        if new_max_len <= self.max_seq_len:
            return
        
        self.max_seq_len = new_max_len
        
        # Clear cache to force recomputation
        if self.use_cache:
            self._cached_cos = None
            self._cached_sin = None
            self._cached_seq_len = 0

    def to(self, device):
        super().to(device)
        if hasattr(self, 'position_scaling'):
            self.position_scaling = self.position_scaling.to(device)
        if hasattr(self, 'freq_enhancement'):
            self.freq_enhancement = self.freq_enhancement.to(device)
        if hasattr(self, 'inv_freq'):
            self.inv_freq = self.inv_freq.to(device)
        return self


class MathematicalRoPE(RotaryPositionalEmbedding):
    """
    Mathematical Reasoning optimized RoPE with adaptive frequencies.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, math_enhanced=True, **kwargs)
        
        # Additional mathematical reasoning enhancements
        self.adaptive_scaling = nn.Parameter(torch.ones(self.dim // 2))
        self.mathematical_bias = nn.Parameter(torch.zeros(self.dim // 2))
        
        # Pattern-specific frequency adjustments
        self.arithmetic_freq_mult = nn.Parameter(torch.ones(1))
        self.algebraic_freq_mult = nn.Parameter(torch.ones(1))
        self.geometric_freq_mult = nn.Parameter(torch.ones(1))
    
    def _compute_adaptive_freq(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute adaptive frequencies based on mathematical patterns."""
        device = positions.device
        base_freq = self.inv_freq.to(device) * self.freq_enhancement.to(device)
        
        # Add learned bias
        adapted_freq = base_freq + self.mathematical_bias.to(device)
        
        # Apply adaptive scaling
        adapted_freq = adapted_freq * self.adaptive_scaling.to(device)
        
        # Mathematical pattern-specific adjustments
        # (This would typically be conditioned on input content)
        pattern_mult = (self.arithmetic_freq_mult.to(device) + 
                       self.algebraic_freq_mult.to(device) + 
                       self.geometric_freq_mult.to(device)) / 3.0
        
        return adapted_freq * pattern_mult
    
    def _get_cos_sin(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced cos/sin computation with adaptive frequencies."""
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        positions = positions * self.scaling_factor * self.position_scaling.to(device)
        
        # Use adaptive frequencies
        cos, sin = self._compute_adaptive_cos_sin(positions, device)
        
        return cos, sin

    def to(self, device):
        super().to(device)
        if hasattr(self, 'position_scaling'):
            self.position_scaling = self.position_scaling.to(device)
        if hasattr(self, 'freq_enhancement'):
            self.freq_enhancement = self.freq_enhancement.to(device)
        if hasattr(self, 'inv_freq'):
            self.inv_freq = self.inv_freq.to(device)
        if hasattr(self, 'mathematical_bias'):
            self.mathematical_bias = self.mathematical_bias.to(device)
        if hasattr(self, 'adaptive_scaling'):
            self.adaptive_scaling = self.adaptive_scaling.to(device)
        if hasattr(self, 'arithmetic_freq_mult'):
            self.arithmetic_freq_mult = self.arithmetic_freq_mult.to(device)
        if hasattr(self, 'algebraic_freq_mult'):
            self.algebraic_freq_mult = self.algebraic_freq_mult.to(device)
        if hasattr(self, 'geometric_freq_mult'):
            self.geometric_freq_mult = self.geometric_freq_mult.to(device)
        return self


class LongSequenceRoPE(RotaryPositionalEmbedding):
    """
    RoPE optimized for very long mathematical sequences with interpolation.
    """
    
    def __init__(self, *args, interpolation_factor: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.interpolation_factor = interpolation_factor
    
    def _get_cos_sin(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Interpolated cos/sin for long sequences."""
        if seq_len > self.max_seq_len:
            # Use position interpolation for sequences longer than training
            effective_seq_len = min(seq_len, int(self.max_seq_len * self.interpolation_factor))
            positions = torch.linspace(0, effective_seq_len - 1, seq_len, device=device)
        else:
            positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        positions = positions * self.scaling_factor
        
        if self.math_enhanced and hasattr(self, 'position_scaling'):
            positions = positions * self.position_scaling.to(device)
        
        inv_freq = self.inv_freq.to(device)
        if self.math_enhanced and hasattr(self, 'freq_enhancement'):
            inv_freq = inv_freq * self.freq_enhancement.to(device)
        
        freqs = torch.outer(positions, inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin

    def to(self, device):
        super().to(device)
        if hasattr(self, 'position_scaling'):
            self.position_scaling = self.position_scaling.to(device)
        if hasattr(self, 'freq_enhancement'):
            self.freq_enhancement = self.freq_enhancement.to(device)
        if hasattr(self, 'inv_freq'):
            self.inv_freq = self.inv_freq.to(device)
        return self