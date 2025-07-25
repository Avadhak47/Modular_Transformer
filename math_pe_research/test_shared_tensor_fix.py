#!/usr/bin/env python3
"""
Test script to verify the shared tensor issue is completely fixed.
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

def check_shared_tensors(model):
    """Check if any tensors are shared between PE layers."""
    print("🔍 Checking for shared tensors...")
    
    # Get all PE layers
    pe_layers = []
    for name, module in model.named_modules():
        if 'pe_layer' in name and hasattr(module, 'inv_freq'):
            pe_layers.append((name, module))
    
    print(f"Found {len(pe_layers)} PE layers")
    
    # Check for shared tensors
    shared_tensors = []
    for i, (name1, layer1) in enumerate(pe_layers):
        for j, (name2, layer2) in enumerate(pe_layers[i+1:], i+1):
            # Check inv_freq
            if hasattr(layer1, 'inv_freq') and hasattr(layer2, 'inv_freq'):
                if layer1.inv_freq.data_ptr() == layer2.inv_freq.data_ptr():
                    shared_tensors.append((name1, name2, 'inv_freq'))
            
            # Check freq_enhancement
            if hasattr(layer1, 'freq_enhancement') and hasattr(layer2, 'freq_enhancement'):
                if layer1.freq_enhancement.data_ptr() == layer2.freq_enhancement.data_ptr():
                    shared_tensors.append((name1, name2, 'freq_enhancement'))
            
            # Check position_scaling
            if hasattr(layer1, 'position_scaling') and hasattr(layer2, 'position_scaling'):
                if layer1.position_scaling.data_ptr() == layer2.position_scaling.data_ptr():
                    shared_tensors.append((name1, name2, 'position_scaling'))
    
    if shared_tensors:
        print("❌ Found shared tensors:")
        for name1, name2, param_name in shared_tensors:
            print(f"   {name1} and {name2} share {param_name}")
        return False
    else:
        print("✅ No shared tensors found!")
        return True

def test_model_creation():
    """Test model creation with different PE methods."""
    print("🧪 Testing model creation...")
    
    pe_methods = ['rope', 'sinusoidal', 'alibi']
    
    for pe_method in pe_methods:
        print(f"\n📋 Testing {pe_method.upper()} PE...")
        
        try:
            # Create model
            model = create_mathematical_reasoning_model(
                pe_method=pe_method,
                base_model='microsoft/DialoGPT-small',
                use_lora=True,
                load_in_4bit=False,
                enable_gradient_checkpointing=False,
                torch_dtype=torch.float16,
                device_map="cpu"
            )
            
            print(f"✅ {pe_method} model created successfully")
            
            # Check for shared tensors
            no_shared = check_shared_tensors(model)
            
            if no_shared:
                print(f"✅ {pe_method} PE: No shared tensors")
            else:
                print(f"❌ {pe_method} PE: Has shared tensors")
                return False
            
            # Test forward pass
            input_ids = torch.randint(0, 1000, (1, 64), dtype=torch.long)
            outputs = model(input_ids=input_ids)
            print(f"✅ {pe_method} forward pass successful")
            
        except Exception as e:
            print(f"❌ {pe_method} test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

def test_model_saving():
    """Test model saving without shared tensor errors."""
    print("\n🧪 Testing model saving...")
    
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
        
        print("✅ Model created")
        
        # Test saving
        save_dir = "/tmp/test_model_save_fixed"
        model.save_pretrained(save_dir)
        print("✅ Model saved successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Save test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step():
    """Test training step with the fixed model."""
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
    print("🔍 Testing Shared Tensor Fix")
    print("=" * 40)
    
    # Test 1: Model creation and shared tensor check
    creation_test = test_model_creation()
    
    # Test 2: Model saving
    save_test = test_model_saving()
    
    # Test 3: Training step
    training_test = test_training_step()
    
    print("\n📊 Test Results:")
    print(f"   Model Creation: {'✅ PASS' if creation_test else '❌ FAIL'}")
    print(f"   Model Saving: {'✅ PASS' if save_test else '❌ FAIL'}")
    print(f"   Training Step: {'✅ PASS' if training_test else '❌ FAIL'}")
    
    if creation_test and save_test and training_test:
        print("\n🎉 All tests passed! Shared tensor issue is completely fixed.")
        print("   You can now train without save errors.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.") 