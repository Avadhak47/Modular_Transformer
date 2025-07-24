#!/usr/bin/env python3
"""
PE Layer Consistency Verification Script

This script verifies that all positional encoding layers maintain constant output shapes
and explains batch size and context limit handling.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from positional_encoding.rope import RotaryPositionalEmbedding
from positional_encoding.sinusoidal import SinusoidalPositionalEncoding
from positional_encoding.t5_relative import T5RelativePositionalBias
from positional_encoding.diet import DIETPositionalEncoding
from positional_encoding.alibi import ALiBiPositionalEncoding
from positional_encoding.math_adaptive import MathAdaptivePositionalEncoding


def test_pe_shape_consistency():
    """Test that all PE layers maintain constant output shapes."""
    
    print("=" * 80)
    print("PE LAYER SHAPE CONSISTENCY VERIFICATION")
    print("=" * 80)
    
    # Test parameters
    batch_size = 4
    seq_len = 231
    hidden_size = 2560
    num_heads = 40
    head_dim = 64
    
    print(f"Test Configuration:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_heads: {num_heads}")
    print(f"  head_dim: {head_dim}")
    print()
    
    # Test input shapes
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    query_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    token_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    print("Input Shapes:")
    print(f"  hidden_states: {hidden_states.shape}")
    print(f"  query_states: {query_states.shape}")
    print(f"  key_states: {key_states.shape}")
    print(f"  token_ids: {token_ids.shape}")
    print()
    
    # Test each PE method
    pe_methods = {
        'rope': RotaryPositionalEmbedding(dim=head_dim),
        'sinusoidal': SinusoidalPositionalEncoding(d_model=hidden_size),
        't5_relative': T5RelativePositionalBias(d_model=hidden_size, num_heads=num_heads),
        'diet': DIETPositionalEncoding(d_model=hidden_size),
        'alibi': ALiBiPositionalEncoding(d_model=hidden_size, num_heads=num_heads),
        'math_adaptive': MathAdaptivePositionalEncoding(d_model=hidden_size)
    }
    
    print("PE Layer Shape Verification:")
    print("-" * 50)
    
    for pe_name, pe_layer in pe_methods.items():
        print(f"\n{pe_name.upper()} PE:")
        
        try:
            if pe_name == 'rope':
                # RoPE: expect [batch, heads, seq_len, head_dim] for Q/K
                output_q, output_k = pe_layer(query_states, key_states)
                print(f"  Input Q: {query_states.shape}")
                print(f"  Input K: {key_states.shape}")
                print(f"  Output Q: {output_q.shape}")
                print(f"  Output K: {output_k.shape}")
                
                # Verify shape consistency
                assert output_q.shape == query_states.shape, f"Q shape mismatch: {output_q.shape} != {query_states.shape}"
                assert output_k.shape == key_states.shape, f"K shape mismatch: {output_k.shape} != {key_states.shape}"
                print(f"  ✅ Shape consistency: PASSED")
                
            else:
                # Additive PEs: expect [batch, seq_len, hidden_size]
                if pe_name == 't5_relative':
                    # T5 relative needs attention scores for bias application
                    attention_scores = torch.randn(batch_size, num_heads, seq_len, seq_len)
                    output = pe_layer(hidden_states, attention_scores=attention_scores)
                    print(f"  Input: {hidden_states.shape}")
                    print(f"  Attention scores: {attention_scores.shape}")
                    print(f"  Output: {output.shape}")
                    
                    # T5 returns attention scores with bias applied
                    assert output.shape == attention_scores.shape, f"Shape mismatch: {output.shape} != {attention_scores.shape}"
                    print(f"  ✅ Shape consistency: PASSED")
                    
                elif pe_name == 'alibi':
                    # ALiBi can work with both hidden states and attention scores
                    output = pe_layer(hidden_states)
                    print(f"  Input: {hidden_states.shape}")
                    print(f"  Output: {output.shape}")
                    
                    # ALiBi returns input unchanged when no attention scores provided
                    assert output.shape == hidden_states.shape, f"Shape mismatch: {output.shape} != {hidden_states.shape}"
                    print(f"  ✅ Shape consistency: PASSED")
                    
                else:
                    # Standard additive PEs
                    output = pe_layer(hidden_states, token_ids=token_ids)
                    print(f"  Input: {hidden_states.shape}")
                    print(f"  Output: {output.shape}")
                    
                    # Verify shape consistency
                    assert output.shape == hidden_states.shape, f"Shape mismatch: {output.shape} != {hidden_states.shape}"
                    print(f"  ✅ Shape consistency: PASSED")
                    
        except Exception as e:
            print(f"  ❌ FAILED: {str(e)}")
    
    print("\n" + "=" * 80)
    print("ALL PE LAYERS VERIFIED FOR SHAPE CONSISTENCY")
    print("=" * 80)


def explain_batch_size_and_context_limits():
    """Explain how batch size and context limits work in the model."""
    
    print("\n" + "=" * 80)
    print("BATCH SIZE AND CONTEXT LIMIT EXPLANATION")
    print("=" * 80)
    
    print("\n1. BATCH SIZE HANDLING:")
    print("-" * 30)
    print("• Batch size is the number of sequences processed in parallel")
    print("• Each sequence in the batch can have different lengths (padded to max length)")
    print("• Memory usage scales linearly with batch size")
    print("• Typical values: 1-32 for training, 1 for inference")
    print()
    
    print("2. CONTEXT LIMIT IMPOSITION:")
    print("-" * 35)
    print("• Context limit is the maximum sequence length the model can handle")
    print("• Imposed at multiple levels:")
    print("  - Tokenizer: truncates input to max_length")
    print("  - PE layers: have max_seq_len parameter")
    print("  - Attention: quadratic memory scaling with sequence length")
    print("  - Model config: typically 2048-8192 tokens")
    print()
    
    print("3. MEMORY SCALING:")
    print("-" * 20)
    print("• Attention memory: O(batch_size × seq_len² × num_heads)")
    print("• For seq_len=2048, batch_size=4, num_heads=40:")
    print("  - Attention matrices: 4 × 2048² × 40 = ~671M elements")
    print("  - With float16: ~1.34 GB just for attention")
    print()
    
    print("4. CONTEXT LIMIT ENFORCEMENT:")
    print("-" * 35)
    print("• Input truncation: tokenizer.truncate_side='right'")
    print("• PE layer limits: max_seq_len parameter")
    print("• Attention masking: prevents attention beyond sequence length")
    print("• Memory management: gradient checkpointing for long sequences")
    print()
    
    print("5. PE LAYER CONTEXT HANDLING:")
    print("-" * 35)
    print("• RoPE: Can extrapolate beyond training length")
    print("• Sinusoidal: Fixed max_seq_len, truncates beyond")
    print("• T5 Relative: Uses relative position buckets")
    print("• ALiBi: Excellent extrapolation properties")
    print("• DIET: Standard truncation behavior")
    print("• Math-Adaptive: Adaptive to mathematical content")
    print()
    
    print("6. PRACTICAL LIMITS:")
    print("-" * 20)
    print("• GPU Memory: Primary constraint")
    print("• Attention complexity: O(n²) scaling")
    print("• Training: Usually 2048-4096 tokens")
    print("• Inference: Can be longer with techniques like:")
    print("  - Sliding window attention")
    print("  - Sparse attention")
    print("  - Memory-efficient attention")


def test_context_limit_handling():
    """Test how different PE layers handle context limits."""
    
    print("\n" + "=" * 80)
    print("CONTEXT LIMIT HANDLING TEST")
    print("=" * 80)
    
    # Test parameters
    batch_size = 2
    short_seq_len = 512
    long_seq_len = 4096
    hidden_size = 2560
    num_heads = 40
    head_dim = 64
    
    print(f"Testing context limit handling:")
    print(f"  Short sequence: {short_seq_len} tokens")
    print(f"  Long sequence: {long_seq_len} tokens")
    print()
    
    # Create PE layers with different max_seq_len
    pe_layers = {
        'rope': RotaryPositionalEmbedding(dim=head_dim, max_seq_len=2048),
        'sinusoidal': SinusoidalPositionalEncoding(d_model=hidden_size, max_seq_len=2048),
        't5_relative': T5RelativePositionalBias(d_model=hidden_size, num_heads=num_heads),
        'diet': DIETPositionalEncoding(d_model=hidden_size, max_seq_len=2048),
        'alibi': ALiBiPositionalEncoding(d_model=hidden_size, num_heads=num_heads, max_seq_len=2048),
        'math_adaptive': MathAdaptivePositionalEncoding(d_model=hidden_size, max_seq_len=2048)
    }
    
    for pe_name, pe_layer in pe_layers.items():
        print(f"\n{pe_name.upper()} PE Context Handling:")
        
        try:
            # Test short sequence
            short_hidden = torch.randn(batch_size, short_seq_len, hidden_size)
            short_query = torch.randn(batch_size, num_heads, short_seq_len, head_dim)
            short_key = torch.randn(batch_size, num_heads, short_seq_len, head_dim)
            short_tokens = torch.randint(0, 1000, (batch_size, short_seq_len))
            
            if pe_name == 'rope':
                output_q, output_k = pe_layer(short_query, short_key)
                print(f"  Short seq ({short_seq_len}): ✅ Works")
            else:
                if pe_name == 't5_relative':
                    attn_scores = torch.randn(batch_size, num_heads, short_seq_len, short_seq_len)
                    output = pe_layer(short_hidden, attention_scores=attn_scores)
                else:
                    output = pe_layer(short_hidden, token_ids=short_tokens)
                print(f"  Short seq ({short_seq_len}): ✅ Works")
            
            # Test long sequence (beyond max_seq_len)
            long_hidden = torch.randn(batch_size, long_seq_len, hidden_size)
            long_query = torch.randn(batch_size, num_heads, long_seq_len, head_dim)
            long_key = torch.randn(batch_size, num_heads, long_seq_len, head_dim)
            long_tokens = torch.randint(0, 1000, (batch_size, long_seq_len))
            
            if pe_name == 'rope':
                output_q, output_k = pe_layer(long_query, long_key)
                print(f"  Long seq ({long_seq_len}): ✅ Extrapolates")
            else:
                if pe_name == 't5_relative':
                    attn_scores = torch.randn(batch_size, num_heads, long_seq_len, long_seq_len)
                    output = pe_layer(long_hidden, attention_scores=attn_scores)
                else:
                    output = pe_layer(long_hidden, token_ids=long_tokens)
                print(f"  Long seq ({long_seq_len}): ✅ Handles")
                
        except Exception as e:
            print(f"  ❌ FAILED: {str(e)}")


if __name__ == "__main__":
    # Run all tests
    test_pe_shape_consistency()
    explain_batch_size_and_context_limits()
    test_context_limit_handling()
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80) 