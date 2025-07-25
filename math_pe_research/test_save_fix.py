#!/usr/bin/env python3
"""
Test script to verify the shared tensor save issue is fixed.
"""

import os
import sys
import torch
from pathlib import Path

# Add the src directory to Python path for module imports
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))

from models.mathematical_reasoning_model import create_mathematical_reasoning_model

def test_model_saving():
    """Test if the model can be saved without shared tensor issues."""
    print("🧪 Testing model saving with RoPE PE...")
    
    try:
        # Create a small model for testing
        model = create_mathematical_reasoning_model(
            pe_method='rope',
            base_model='microsoft/DialoGPT-small',  # Small model for testing
            use_lora=True,
            load_in_4bit=False,
            enable_gradient_checkpointing=False,
            torch_dtype=torch.float16,
            device_map="cpu"  # Use CPU for testing
        )
        
        print("✅ Model created successfully")
        
        # Test forward pass
        input_ids = torch.randint(0, 1000, (1, 64), dtype=torch.long)
        outputs = model(input_ids=input_ids)
        print("✅ Forward pass successful")
        
        # Test saving
        save_dir = "/tmp/test_model_save"
        model.save_pretrained(save_dir)
        print("✅ Model saved successfully!")
        
        # Test loading
        from models.mathematical_reasoning_model import MathematicalReasoningModel
        loaded_model = MathematicalReasoningModel.from_pretrained(save_dir)
        print("✅ Model loaded successfully!")
        
        # Test forward pass with loaded model
        outputs = loaded_model(input_ids=input_ids)
        print("✅ Forward pass with loaded model successful!")
        
        print("🎉 All tests passed! Shared tensor issue is fixed.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step():
    """Test if training step works without memory issues."""
    print("\n🧪 Testing training step...")
    
    try:
        # Create model
        model = create_mathematical_reasoning_model(
            pe_method='rope',
            base_model='microsoft/DialoGPT-small',
            use_lora=True,
            load_in_4bit=False,
            enable_gradient_checkpointing=False,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        
        # Test training step
        model.train()
        input_ids = torch.randint(0, 1000, (1, 64), dtype=torch.long)
        labels = input_ids.clone()
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        print("✅ Training step successful!")
        print(f"   Loss: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Testing Shared Tensor Save Fix")
    print("=" * 40)
    
    # Test 1: Model saving
    save_test = test_model_saving()
    
    # Test 2: Training step
    training_test = test_training_step()
    
    print("\n📊 Test Results:")
    print(f"   Model Saving: {'✅ PASS' if save_test else '❌ FAIL'}")
    print(f"   Training Step: {'✅ PASS' if training_test else '❌ FAIL'}")
    
    if save_test and training_test:
        print("\n🎉 All tests passed! You can now train without save issues.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.") 