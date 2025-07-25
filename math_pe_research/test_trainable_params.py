#!/usr/bin/env python3
"""
Test script to verify that the model has trainable parameters.
"""

import torch
import sys
from pathlib import Path

# Add the src directory to Python path for module imports
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))

from models.mathematical_reasoning_model import create_mathematical_reasoning_model

def test_trainable_parameters():
    """Test that the model has trainable parameters and gradients flow."""
    print("Testing trainable parameters...")
    
    # Test with a small model
    model = create_mathematical_reasoning_model(
        pe_method="rope",
        base_model="microsoft/DialoGPT-small",  # Small model for testing
        use_lora=True,
        load_in_4bit=False
    )
    
    # Disable gradient checkpointing for testing
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'config'):
        model.base_model.config.use_cache = True
        if hasattr(model.base_model, 'gradient_checkpointing_disable'):
            model.base_model.gradient_checkpointing_disable()
        elif hasattr(model.base_model, 'transformer') and hasattr(model.base_model.transformer, 'gradient_checkpointing'):
            model.base_model.transformer.gradient_checkpointing = False
    
    # Check trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total ({100 * trainable_params / total_params:.2f}%)")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable: {name}, shape: {param.shape}, device: {param.device}")
    
    if trainable_params == 0:
        print("❌ ERROR: No trainable parameters found!")
        return False
    else:
        print("✅ SUCCESS: Model has trainable parameters!")
    
    # Minimal forward+backward pass test
    print("\nTesting forward and backward pass...")
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
    if loss is not None:
        try:
            loss.backward()
            print("✅ SUCCESS: Backward pass worked!")
            # Print gradient norms for trainable parameters
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    print(f"Grad OK: {name}, grad norm: {param.grad.norm().item():.4f}")
                elif param.requires_grad:
                    print(f"No grad: {name}")
            return True
        except Exception as e:
            print(f"❌ ERROR: Backward pass failed: {e}")
            return False
    else:
        print("❌ ERROR: No loss returned from model!")
        return False

if __name__ == "__main__":
    success = test_trainable_parameters()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")
        sys.exit(1) 