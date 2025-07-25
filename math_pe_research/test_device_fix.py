#!/usr/bin/env python3
"""
Test script to verify the device mismatch fix
"""

import torch
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from positional_encoding.rope import RotaryPositionalEmbedding
from models.mathematical_reasoning_model import create_mathematical_reasoning_model


def test_device_fix():
    """Test that the device mismatch is fixed."""
    
    print("Testing device mismatch fix...")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using CUDA device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using CPU device: {device}")
    
    try:
        # Test RoPE PE directly
        print("\n1. Testing RoPE PE device handling...")
        rope_pe = RotaryPositionalEmbedding(dim=64, math_enhanced=True)
        rope_pe = rope_pe.to(device)
        
        # Create test tensors on device
        batch_size, num_heads, seq_len, head_dim = 2, 8, 128, 64
        query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        
        # Test forward pass
        output_q, output_k = rope_pe(query, key)
        print(f"✓ RoPE PE forward pass successful")
        print(f"  Query device: {output_q.device}")
        print(f"  Key device: {output_k.device}")
        
        # Test model creation and forward pass
        print("\n2. Testing model device handling...")
        model = create_mathematical_reasoning_model(
            pe_method="rope",
            base_model="microsoft/DialoGPT-small",  # Small model for testing
            load_in_4bit=False,
            use_lora=False,  # Disable LoRA for simpler testing
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        print("✓ Model created successfully")
        
        # Test with a simple input
        input_text = "What is 2 + 3?"
        tokenizer = model.module.tokenizer if hasattr(model, 'module') else model.tokenizer
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        
        # Move input to device
        if torch.cuda.is_available():
            input_ids = input_ids.to(device)
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Input device: {input_ids.device}")
        
        # Test forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            print("✓ Forward pass successful")
            print(f"Output logits shape: {outputs['logits'].shape}")
            print(f"Output device: {outputs['logits'].device}")
        
        # Test generation
        generated = model.generate(
            input_ids=input_ids,
            max_length=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        print("✓ Generation successful")
        
        # Decode the generated text
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Generated text: {generated_text}")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = test_device_fix()
    if success:
        print("\n✅ Device mismatch fix verified!")
    else:
        print("\n❌ Device mismatch fix failed!") 