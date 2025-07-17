"""
Advanced Sinusoidal Positional Encoding

Enhanced implementation with mathematical reasoning optimizations:
- Extended frequency ranges for better mathematical pattern capture
- Learnable frequency scaling for adaptive positioning
- Support for very long sequences (up to 32K tokens)
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SinusoidalPositionalEncoding(nn.Module):
    """
    Advanced Sinusoidal Positional Encoding for Mathematical Reasoning
    
    Features:
    - Extended frequency range for mathematical patterns
    - Learnable scaling factors
    - Memory-efficient implementation
    - Support for sequences up to 32K tokens
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 32768,
        base: float = 10000.0,
        learnable_scaling: bool = True,
        math_optimized: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        self.math_optimized = math_optimized
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        self.register_buffer('pe', self._create_positional_encoding())
        
        # Learnable scaling factors for mathematical reasoning
        if learnable_scaling:
            self.freq_scale = nn.Parameter(torch.ones(d_model // 2))
            self.amplitude_scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('freq_scale', torch.ones(d_model // 2))
            self.register_buffer('amplitude_scale', torch.ones(1))
            
        # Mathematical pattern enhancement
        if math_optimized:
            self.math_enhancement = nn.Linear(d_model, d_model)
        else:
            self.math_enhancement = None
    
    def _create_positional_encoding(self) -> torch.Tensor:
        """Create the positional encoding matrix."""
        pe = torch.zeros(self.max_seq_len, self.d_model)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Enhanced frequency calculation for mathematical reasoning
        if self.math_optimized:
            # Use different frequency ranges for different mathematical concepts
            div_term_low = torch.exp(torch.arange(0, self.d_model//4, 2).float() * 
                                   (-math.log(self.base) / (self.d_model//4)))
            div_term_mid = torch.exp(torch.arange(0, self.d_model//4, 2).float() * 
                                   (-math.log(self.base * 10) / (self.d_model//4)))
            div_term_high = torch.exp(torch.arange(0, self.d_model//4, 2).float() * 
                                    (-math.log(self.base * 100) / (self.d_model//4)))
            div_term_extra = torch.exp(torch.arange(0, self.d_model//4, 2).float() * 
                                     (-math.log(self.base * 1000) / (self.d_model//4)))
            
            div_term = torch.cat([div_term_low, div_term_mid, div_term_high, div_term_extra])
        else:
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                               (-math.log(self.base) / self.d_model))
        
        # Apply sinusoidal functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Add batch dimension
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to input embeddings.
        
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
            
        Returns:
            Embeddings with positional encoding applied
        """
        batch_size, seq_len, d_model = x.shape
        
        # Get positional encodings for current sequence length
        pos_encoding = self.pe[:, :seq_len, :]
        
        # Apply learnable scaling
        if hasattr(self, 'freq_scale'):
            # Reshape frequency scaling for broadcasting
            freq_scale_expanded = torch.ones(d_model, device=x.device)
            freq_scale_expanded[0::2] = self.freq_scale.repeat_interleave(2)[:d_model//2]
            freq_scale_expanded[1::2] = self.freq_scale.repeat_interleave(2)[:d_model//2]
            
            pos_encoding = pos_encoding * freq_scale_expanded.unsqueeze(0).unsqueeze(0)
        
        # Apply amplitude scaling
        pos_encoding = pos_encoding * self.amplitude_scale
        
        # Add positional encoding to input
        x = x + pos_encoding
        
        # Apply mathematical enhancement if enabled
        if self.math_enhancement is not None:
            x = x + self.math_enhancement(pos_encoding)
        
        return self.dropout(x)
    
    def extend_length(self, new_max_len: int):
        """Extend the maximum sequence length."""
        if new_max_len <= self.max_seq_len:
            return
            
        self.max_seq_len = new_max_len
        new_pe = self._create_positional_encoding()
        self.register_buffer('pe', new_pe)
    
    def get_frequencies(self) -> torch.Tensor:
        """Get the frequency components for analysis."""
        return self.freq_scale if hasattr(self, 'freq_scale') else torch.ones(self.d_model // 2)


class AdaptiveSinusoidalPE(SinusoidalPositionalEncoding):
    """
    Adaptive Sinusoidal PE that learns optimal frequencies for mathematical reasoning.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Additional learnable parameters for mathematical reasoning
        self.math_freq_bias = nn.Parameter(torch.zeros(self.d_model // 2))
        self.position_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced forward pass with adaptive frequencies."""
        batch_size, seq_len, d_model = x.shape
        
        # Get base positional encoding
        pos_encoding = self.pe[:, :seq_len, :]
        
        # Apply adaptive frequency modifications
        position = torch.arange(seq_len, device=x.device).float().unsqueeze(1)
        
        # Enhanced frequency calculation with learned bias
        freq_with_bias = self.freq_scale + self.math_freq_bias
        
        # Recalculate with adaptive frequencies for mathematical patterns
        adaptive_pe = torch.zeros_like(pos_encoding)
        for i in range(0, d_model, 2):
            freq_idx = i // 2
            freq = freq_with_bias[freq_idx] if freq_idx < len(freq_with_bias) else 1.0
            
            adaptive_pe[:, :, i] = torch.sin(position.squeeze() * freq * self.position_weight)
            if i + 1 < d_model:
                adaptive_pe[:, :, i+1] = torch.cos(position.squeeze() * freq * self.position_weight)
        
        # Combine original and adaptive components
        final_pe = (pos_encoding + adaptive_pe) * self.amplitude_scale
        
        # Add to input
        x = x + final_pe
        
        # Apply mathematical enhancement
        if self.math_enhancement is not None:
            x = x + self.math_enhancement(final_pe)
        
        return self.dropout(x)