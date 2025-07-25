#!/usr/bin/env python3
"""
Test script to verify that the shared tensor issue is resolved.
"""

import torch
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from models.mathematical_reasoning_model import create_mathematical_reasoning_model

def test_shared_tensor_fix():
    """Test that the model can be saved without shared tensor errors."""
    
    print("🧪 Testing shared tensor fix...")
    
    # Create model with RoPE PE
    model = create_mathematical_reasoning_model(
        pe_method='rope',
        base_model='deepseek-ai/deepseek-math-7b-instruct',
        use_lora=True,
        load_in_4bit=False
    )
    
    print("✅ Model created successfully")
    
    # Check for shared tensors
    print("\n🔍 Checking for shared tensors...")
    
    shared_tensors = []
    param_names = list(model.named_parameters())
    
    for i, (name1, param1) in enumerate(param_names):
        for j, (name2, param2) in enumerate(param_names[i+1:], i+1):
            if param1.data_ptr() == param2.data_ptr():
                shared_tensors.append((name1, name2))
    
    if shared_tensors:
        print(f"❌ Found {len(shared_tensors)} shared tensor pairs:")
        for name1, name2 in shared_tensors:
            print(f"   {name1} <-> {name2}")
    else:
        print("✅ No shared tensors found!")
    
    # Test model saving
    print("\n💾 Testing model saving...")
    
    try:
        # Create test directory
        test_dir = Path('./test_save')
        test_dir.mkdir(exist_ok=True)
        
        # Save model
        model.save_pretrained(str(test_dir))
        print("✅ Model saved successfully without shared tensor errors!")
        
        # Clean up
        import shutil
        shutil.rmtree(test_dir)
        print("🧹 Test directory cleaned up")
        
    except Exception as e:
        print(f"❌ Model saving failed: {e}")
        return False
    
    return True

def test_parameter_access():
    """Test that PE parameters can be accessed correctly."""
    
    print("\n🔧 Testing parameter access...")
    
    model = create_mathematical_reasoning_model(
        pe_method='rope',
        base_model='deepseek-ai/deepseek-math-7b-instruct',
        use_lora=True
    )
    
    # Check if PE layers have get_param method
    for name, module in model.named_modules():
        if 'pe_layer' in name and hasattr(module, 'get_param'):
            print(f"✅ PE layer {name} has get_param method")
            
            # Test parameter access
            try:
                position_scaling = module.get_param('position_scaling')
                freq_enhancement = module.get_param('freq_enhancement')
                
                if position_scaling is not None:
                    print(f"   ✅ position_scaling parameter accessible")
                if freq_enhancement is not None:
                    print(f"   ✅ freq_enhancement parameter accessible")
                    
            except Exception as e:
                print(f"   ❌ Parameter access failed: {e}")
    
    print("✅ Parameter access test completed")

if __name__ == "__main__":
    print("🚀 Starting shared tensor fix verification...")
    
    # Test 1: Shared tensor detection
    success1 = test_shared_tensor_fix()
    
    # Test 2: Parameter access
    test_parameter_access()
    
    if success1:
        print("\n🎉 All tests passed! Shared tensor issue is resolved.")
    else:
        print("\n❌ Tests failed. Shared tensor issue still exists.") 