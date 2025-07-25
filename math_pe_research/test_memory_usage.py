#!/usr/bin/env python3
"""
Test memory usage with different model configurations.
"""

import os
import sys
import torch
from pathlib import Path

# Set memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add the src directory to Python path for module imports
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))

from models.mathematical_reasoning_model import create_mathematical_reasoning_model

def print_memory_usage():
    """Print current memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"💾 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
    else:
        print("💾 Using CPU")

def test_model_memory(model_name, pe_method="rope"):
    """Test memory usage for a specific model."""
    print(f"\n🧪 Testing {model_name} with {pe_method} PE...")
    print_memory_usage()
    
    try:
        # Clear cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load model
        model = create_mathematical_reasoning_model(
            pe_method=pe_method,
            base_model=model_name,
            use_lora=True,
            load_in_4bit=False,
            enable_gradient_checkpointing=False,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        print("✅ Model loaded successfully")
        print_memory_usage()
        
        # Test forward pass
        input_ids = torch.randint(0, 1000, (1, 512), dtype=torch.long)
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        
        print("✅ Forward pass successful")
        print_memory_usage()
        
        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Error with {model_name}: {e}")
        return False

def main():
    print("🔍 Testing memory usage for different models...")
    
    # Test different model sizes
    models_to_test = [
        "microsoft/DialoGPT-small",  # ~117M parameters
        "microsoft/DialoGPT-medium",  # ~345M parameters
        "microsoft/DialoGPT-large",   # ~774M parameters
    ]
    
    results = {}
    for model_name in models_to_test:
        success = test_model_memory(model_name, "rope")
        results[model_name] = success
    
    print("\n📊 Results:")
    for model_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {model_name}: {status}")

if __name__ == "__main__":
    main() 