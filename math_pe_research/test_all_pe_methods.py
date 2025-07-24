#!/usr/bin/env python3
"""
Comprehensive test script to verify all PE methods work with device fix
"""

import torch
import sys
import os
import time

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from positional_encoding.rope import RotaryPositionalEmbedding, MathematicalRoPE, LongSequenceRoPE
from positional_encoding.sinusoidal import SinusoidalPositionalEncoding
from positional_encoding.t5_relative import T5RelativePositionalBias
from positional_encoding.diet import DIETPositionalEncoding
from positional_encoding.alibi import ALiBiPositionalEncoding
from positional_encoding.math_adaptive import MathAdaptivePositionalEncoding
from models.mathematical_reasoning_model import create_mathematical_reasoning_model


def test_pe_method(pe_name, pe_class, test_params, device):
    """Test a specific PE method."""
    print(f"\n{'='*60}")
    print(f"Testing {pe_name}")
    print(f"{'='*60}")
    
    try:
        # Create PE layer
        pe_layer = pe_class(**test_params)
        pe_layer = pe_layer.to(device)
        print(f"✓ {pe_name} created successfully")
        
        # Test forward pass
        if pe_name in ["RoPE", "MathematicalRoPE", "LongSequenceRoPE"]:
            # RoPE-like PEs take query and key
            batch_size, num_heads, seq_len, head_dim = 2, 8, 128, 64
            query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
            key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
            
            output_q, output_k = pe_layer(query, key)
            print(f"✓ {pe_name} forward pass successful")
            print(f"  Query shape: {output_q.shape}, device: {output_q.device}")
            print(f"  Key shape: {output_k.shape}, device: {output_k.device}")
            
        else:
            # Additive PEs take hidden states
            batch_size, seq_len, hidden_dim = 2, 128, 512
            hidden_states = torch.randn(batch_size, seq_len, hidden_dim, device=device)
            
            if pe_name == "MathAdaptive":
                # MathAdaptive needs token_ids
                token_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
                output = pe_layer(hidden_states, token_ids=token_ids)
            else:
                output = pe_layer(hidden_states)
            
            print(f"✓ {pe_name} forward pass successful")
            print(f"  Output shape: {output.shape}, device: {output.device}")
        
        # Test different sequence lengths
        print(f"  Testing sequence length flexibility...")
        if pe_name in ["RoPE", "MathematicalRoPE", "LongSequenceRoPE"]:
            # Test longer sequence
            long_query = torch.randn(1, 4, 256, 64, device=device)
            long_key = torch.randn(1, 4, 256, 64, device=device)
            long_q, long_k = pe_layer(long_query, long_key)
            print(f"    ✓ Long sequence (256) works")
            
            # Test shorter sequence
            short_query = torch.randn(1, 4, 64, 64, device=device)
            short_key = torch.randn(1, 4, 64, 64, device=device)
            short_q, short_k = pe_layer(short_query, short_key)
            print(f"    ✓ Short sequence (64) works")
        else:
            # Test longer sequence
            long_hidden = torch.randn(1, 256, hidden_dim, device=device)
            if pe_name == "MathAdaptive":
                long_tokens = torch.randint(0, 1000, (1, 256), device=device)
                long_output = pe_layer(long_hidden, token_ids=long_tokens)
            else:
                long_output = pe_layer(long_hidden)
            print(f"    ✓ Long sequence (256) works")
            
            # Test shorter sequence
            short_hidden = torch.randn(1, 64, hidden_dim, device=device)
            if pe_name == "MathAdaptive":
                short_tokens = torch.randint(0, 1000, (1, 64), device=device)
                short_output = pe_layer(short_hidden, token_ids=short_tokens)
            else:
                short_output = pe_layer(short_hidden)
            print(f"    ✓ Short sequence (64) works")
        
        return True
        
    except Exception as e:
        print(f"✗ {pe_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_with_pe(pe_method, device):
    """Test model creation and forward pass with specific PE method."""
    print(f"\n{'='*60}")
    print(f"Testing Model with {pe_method.upper()} PE")
    print(f"{'='*60}")
    
    try:
        # Create model with specific PE
        model = create_mathematical_reasoning_model(
            pe_method=pe_method,
            base_model="microsoft/DialoGPT-small",  # Small model for testing
            load_in_4bit=False,
            use_lora=False,  # Disable LoRA for simpler testing
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        print(f"✓ Model with {pe_method} PE created successfully")
        
        # Test forward pass
        input_text = "What is 2 + 3?"
        input_ids = model.tokenizer.encode(input_text, return_tensors="pt")
        
        if torch.cuda.is_available():
            input_ids = input_ids.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            print(f"✓ Forward pass successful")
            print(f"  Output logits shape: {outputs['logits'].shape}")
            print(f"  Output device: {outputs['logits'].device}")
        
        # Test generation
        generated = model.generate(
            input_ids=input_ids,
            max_length=20,
            do_sample=False,
            pad_token_id=model.tokenizer.eos_token_id
        )
        print(f"✓ Generation successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Model with {pe_method} PE failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run comprehensive PE testing."""
    print("🧪 Comprehensive PE Method Testing")
    print("=" * 60)
    
    # Check device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using CUDA device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print(f"Using CPU device: {device}")
    
    # Define test parameters for each PE method
    pe_tests = [
        ("RoPE", RotaryPositionalEmbedding, {"dim": 64, "math_enhanced": True}),
        ("MathematicalRoPE", MathematicalRoPE, {"dim": 64}),
        ("LongSequenceRoPE", LongSequenceRoPE, {"dim": 64, "interpolation_factor": 1.0}),
        ("Sinusoidal", SinusoidalPositionalEncoding, {"d_model": 512, "max_seq_len": 2048}),
        ("T5Relative", T5RelativePositionalBias, {"d_model": 512, "num_heads": 8, "relative_attention_num_buckets": 32}),
        ("DIET", DIETPositionalEncoding, {"d_model": 512, "max_seq_len": 2048}),
        ("ALiBi", ALiBiPositionalEncoding, {"d_model": 512, "num_heads": 8, "max_seq_len": 2048}),
        ("MathAdaptive", MathAdaptivePositionalEncoding, {"d_model": 512, "max_seq_len": 2048}),
    ]
    
    # Test individual PE methods
    print("\n📋 Testing Individual PE Methods")
    print("-" * 40)
    
    successful_pe_tests = 0
    total_pe_tests = len(pe_tests)
    
    for pe_name, pe_class, test_params in pe_tests:
        if test_pe_method(pe_name, pe_class, test_params, device):
            successful_pe_tests += 1
    
    # Test model integration with each PE method
    print("\n🤖 Testing Model Integration")
    print("-" * 40)
    
    pe_methods_for_model = ["rope", "sinusoidal", "t5_relative", "diet", "alibi", "math_adaptive"]
    successful_model_tests = 0
    total_model_tests = len(pe_methods_for_model)
    
    for pe_method in pe_methods_for_model:
        if test_model_with_pe(pe_method, device):
            successful_model_tests += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Individual PE Methods: {successful_pe_tests}/{total_pe_tests} passed")
    print(f"Model Integration: {successful_model_tests}/{total_model_tests} passed")
    
    if successful_pe_tests == total_pe_tests and successful_model_tests == total_model_tests:
        print(f"\n✅ ALL TESTS PASSED!")
        print(f"🎉 Device fix works for all PE methods!")
        return True
    else:
        print(f"\n❌ SOME TESTS FAILED!")
        print(f"🔧 Need to investigate failed tests")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🚀 Ready for Kaggle T4 X2 training with any PE method!")
    else:
        print(f"\n⚠️  Some issues need to be resolved before training") 