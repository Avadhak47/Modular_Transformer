#!/usr/bin/env python3
"""
Test script to verify the training script parameters work correctly.
"""

import sys
import torch
from pathlib import Path

# Add the src directory to Python path for module imports
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))

from models.mathematical_reasoning_model import create_mathematical_reasoning_model

def test_training_script_parameters():
    """Test that the training script parameters work correctly."""
    print("Testing training script parameters...")
    
    # Test with the same parameters as the training script
    model = create_mathematical_reasoning_model(
        pe_method="rope",
        base_model="microsoft/DialoGPT-small",  # Small model for testing
        use_lora=True,
        load_in_4bit=False,
        enable_gradient_checkpointing=False,  # This should be False by default
        torch_dtype=torch.float32,  # Use float32 for testing
        device_map="cpu"
    )
    
    # Check that gradient checkpointing is disabled
    if hasattr(model.base_model, 'config'):
        print(f"use_cache: {getattr(model.base_model.config, 'use_cache', 'Not set')}")
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (2, 8), dtype=torch.long)
    labels = input_ids.clone()
    model.train()
    
    outputs = model(input_ids=input_ids, labels=labels)
    if isinstance(outputs, dict):
        loss = outputs.get('loss', None)
    else:
        loss = getattr(outputs, 'loss', None)
    
    print(f"Loss: {loss}")
    print(f"Loss requires grad: {getattr(loss, 'requires_grad', None)}, grad_fn: {getattr(loss, 'grad_fn', None)}")
    
    if loss is not None and loss.requires_grad:
        try:
            loss.backward()
            print("✅ SUCCESS: Training script parameters work correctly!")
            return True
        except Exception as e:
            print(f"❌ ERROR: Backward pass failed: {e}")
            return False
    else:
        print("❌ ERROR: Loss does not require grad!")
        return False

if __name__ == "__main__":
    success = test_training_script_parameters()
    if success:
        print("\n🎉 Training script parameters test passed!")
    else:
        print("\n💥 Training script parameters test failed!")
        sys.exit(1) 