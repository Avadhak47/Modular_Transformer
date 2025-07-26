"""
Advanced Positional Encoding Implementations for Mathematical Reasoning

This module contains state-of-the-art positional encoding methods optimized
for mathematical reasoning tasks and long sequence processing.
"""

try:
    from .sinusoidal import SinusoidalPositionalEncoding
except ImportError as e:
    print(f"Warning: Could not import SinusoidalPositionalEncoding: {e}")
    SinusoidalPositionalEncoding = None

try:
    from .rope import RotaryPositionalEmbedding
except ImportError as e:
    print(f"Warning: Could not import RotaryPositionalEmbedding: {e}")
    RotaryPositionalEmbedding = None

try:
    from .alibi import ALiBiPositionalEncoding
except ImportError as e:
    print(f"Warning: Could not import ALiBiPositionalEncoding: {e}")
    ALiBiPositionalEncoding = None

try:
    from .diet import DIETPositionalEncoding
except ImportError as e:
    print(f"Warning: Could not import DIETPositionalEncoding: {e}")
    DIETPositionalEncoding = None

try:
    from .t5_relative import T5RelativePositionalBias
except ImportError as e:
    print(f"Warning: Could not import T5RelativePositionalBias: {e}")
    T5RelativePositionalBias = None

try:
    from .math_adaptive import MathAdaptivePositionalEncoding
except ImportError as e:
    print(f"Warning: Could not import MathAdaptivePositionalEncoding: {e}")
    MathAdaptivePositionalEncoding = None

__all__ = []
if SinusoidalPositionalEncoding is not None:
    __all__.append('SinusoidalPositionalEncoding')
if RotaryPositionalEmbedding is not None:
    __all__.append('RotaryPositionalEmbedding')
if ALiBiPositionalEncoding is not None:
    __all__.append('ALiBiPositionalEncoding')
if DIETPositionalEncoding is not None:
    __all__.append('DIETPositionalEncoding')
if T5RelativePositionalBias is not None:
    __all__.append('T5RelativePositionalBias')
if MathAdaptivePositionalEncoding is not None:
    __all__.append('MathAdaptivePositionalEncoding')

# PE Method Registry for easy access - only include successfully imported classes
PE_REGISTRY = {}
if SinusoidalPositionalEncoding is not None:
    PE_REGISTRY['sinusoidal'] = SinusoidalPositionalEncoding
if RotaryPositionalEmbedding is not None:
    PE_REGISTRY['rope'] = RotaryPositionalEmbedding
if ALiBiPositionalEncoding is not None:
    PE_REGISTRY['alibi'] = ALiBiPositionalEncoding
if DIETPositionalEncoding is not None:
    PE_REGISTRY['diet'] = DIETPositionalEncoding
if T5RelativePositionalBias is not None:
    PE_REGISTRY['t5_relative'] = T5RelativePositionalBias
if MathAdaptivePositionalEncoding is not None:
    PE_REGISTRY['math_adaptive'] = MathAdaptivePositionalEncoding

print(f"Successfully loaded PE methods: {list(PE_REGISTRY.keys())}")

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
        available_types = list(PE_REGISTRY.keys())
        if not available_types:
            raise ValueError(f"No positional encoding methods successfully loaded! "
                           f"Check import errors above.")
        raise ValueError(f"Unknown positional encoding type: {pe_type}. "
                        f"Available types: {available_types}")
    
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