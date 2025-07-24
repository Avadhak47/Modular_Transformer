"""
Advanced Positional Encoding Implementations for Mathematical Reasoning

This module contains state-of-the-art positional encoding methods optimized
for mathematical reasoning tasks and long sequence processing.
"""

from .sinusoidal import SinusoidalPositionalEncoding
from .rope import RotaryPositionalEmbedding  # Fixed import name
from .alibi import ALiBiPositionalEncoding
from .diet import DIETPositionalEncoding
from .t5_relative import T5RelativePositionalBias
from .math_adaptive import MathAdaptivePositionalEncoding

__all__ = [
    'SinusoidalPositionalEncoding',
    'RotaryPositionalEmbedding', 
    'ALiBiPositionalEncoding',  # Fixed name
    'DIETPositionalEncoding',
    'T5RelativePositionalBias',
    'MathAdaptivePositionalEncoding'
]

# PE Method Registry for easy access
PE_REGISTRY = {
    'sinusoidal': SinusoidalPositionalEncoding,
    'rope': RotaryPositionalEmbedding,  # Fixed class name
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
    
    # Filter kwargs based on PE type to avoid argument mismatches
    pe_class = PE_REGISTRY[pe_type]
    
    # Get the actual constructor parameters for each PE type
    if pe_type == 'rope':
        # RoPE expects: dim, max_seq_len, base, scaling_factor, math_enhanced, use_cache
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k in ['dim', 'max_seq_len', 'base', 'scaling_factor', 'math_enhanced', 'use_cache']}
        # Map d_model to dim if present
        if 'd_model' in kwargs and 'dim' not in filtered_kwargs:
            filtered_kwargs['dim'] = kwargs['d_model']
    elif pe_type == 't5_relative':
        # T5 expects: d_model, num_heads, relative_attention_num_buckets, relative_attention_max_distance, bidirectional
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k in ['d_model', 'num_heads', 'relative_attention_num_buckets', 
                                  'relative_attention_max_distance', 'bidirectional']}
    elif pe_type == 'alibi':
        # ALiBi expects: d_model, num_heads, max_seq_len, alibi_bias_max, learnable_slopes
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k in ['d_model', 'num_heads', 'max_seq_len', 'alibi_bias_max', 'learnable_slopes']}
    elif pe_type == 'sinusoidal':
        # Sinusoidal expects: d_model, max_seq_len, learnable
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k in ['d_model', 'max_seq_len', 'learnable']}
    elif pe_type == 'diet':
        # DIET expects: d_model, max_seq_len, num_layers
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k in ['d_model', 'max_seq_len', 'num_layers']}
    elif pe_type == 'math_adaptive':
        # Math adaptive expects many parameters, pass most through
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k in ['d_model', 'max_seq_len', 'num_hierarchy_levels', 'symbol_vocab_size',
                                  'operator_weight', 'number_weight', 'variable_weight', 'function_weight',
                                  'bracket_weight', 'learnable_frequencies']}
    else:
        # For unknown types, pass all kwargs
        filtered_kwargs = kwargs
    
    return pe_class(**filtered_kwargs)