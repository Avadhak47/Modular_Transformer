"""
Advanced Positional Encoding Implementations for Mathematical Reasoning

This module contains state-of-the-art positional encoding methods optimized
for mathematical reasoning tasks and long sequence processing.
"""

from .sinusoidal import SinusoidalPositionalEncoding
from .rope import RoPEPositionalEncoding
from .alibi import ALiBiPositionalEncoding
from .diet import DIETPositionalEncoding
from .t5_relative import T5RelativePositionalBias

__all__ = [
    'SinusoidalPositionalEncoding',
    'RotaryPositionalEmbedding', 
    'ALiBiPositionalBias',
    'DIETPositionalEncoding',
    'T5RelativePositionalBias'
]

# PE Method Registry for easy access
PE_REGISTRY = {
    'sinusoidal': SinusoidalPositionalEncoding,
    'rope': RoPEPositionalEncoding,
    'alibi': ALiBiPositionalEncoding,
    'diet': DIETPositionalEncoding,
    't5_relative': T5RelativePositionalBias,
    'math_adaptive': MathAdaptivePositionalEncoding
}

def get_positional_encoding(pe_type: str, **kwargs):
    """
    Factory function to get positional encoding implementation.
    
    Args:
        pe_type (str): Type of positional encoding
        **kwargs: Arguments for the PE constructor
        
    Returns:
        Positional encoding instance
    """
    if pe_type not in PE_REGISTRY:
        raise ValueError(f"Unknown positional encoding type: {pe_type}. "
                        f"Available types: {list(PE_REGISTRY.keys())}")
    
    return PE_REGISTRY[pe_type](**kwargs)