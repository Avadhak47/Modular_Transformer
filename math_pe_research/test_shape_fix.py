#!/usr/bin/env python3
"""
Test script to verify the shape mismatch fix
"""

import torch
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.mathematical_reasoning_model import create_mathematical_reasoning_model


def test_shape_fix():
    """Test that the shape mismatch is fixed."""
    
    print("Testing shape mismatch fix...")
    
    # Test with a small model to avoid memory issues
    try:
        model = create_mathematical_reasoning_model(
            pe_method="rope",
            base_model="microsoft/DialoGPT-small",  # Small model for testing
            load_in_4bit=False,
            use_lora=False,  # Disable LoRA for simpler testing
            torch_dtype=torch.float32,
            device_map="cpu"  # Force CPU to avoid MPS issues
        )
        
        print("✓ Model created successfully")
        
        # Test with a simple input
        input_text = "What is 2 + 3?"
        input_ids = model.tokenizer.encode(input_text, return_tensors="pt")
        
        print(f"Input shape: {input_ids.shape}")
        
        # Test forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            print("✓ Forward pass successful")
            print(f"Output logits shape: {outputs['logits'].shape}")
        
        # Test generation
        generated = model.generate(
            input_ids=input_ids,
            max_length=50,
            do_sample=False,
            pad_token_id=model.tokenizer.eos_token_id
        )
        print("✓ Generation successful")
        
        # Decode the generated text
        generated_text = model.tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Generated text: {generated_text}")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = test_shape_fix()
    if success:
        print("\n✅ Shape mismatch fix verified!")
    else:
        print("\n❌ Shape mismatch fix failed!") 