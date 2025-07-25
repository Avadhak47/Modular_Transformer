#!/usr/bin/env python3
"""
Memory monitoring script for Pythia-2.8B training.
"""

import os
import sys
import torch
import time
from pathlib import Path

# Add the src directory to Python path for module imports
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))

from models.mathematical_reasoning_model import create_mathematical_reasoning_model

def get_memory_stats():
    """Get detailed memory statistics."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        utilization = (allocated / total) * 100
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': free,
            'utilization_percent': utilization
        }
    else:
        return {'error': 'CUDA not available'}

def print_memory_stats():
    """Print current memory statistics."""
    stats = get_memory_stats()
    
    if 'error' in stats:
        print("❌ CUDA not available")
        return
    
    print(f"💾 GPU Memory Status:")
    print(f"   Allocated: {stats['allocated_gb']:.2f}GB")
    print(f"   Reserved:  {stats['reserved_gb']:.2f}GB")
    print(f"   Free:      {stats['free_gb']:.2f}GB")
    print(f"   Total:     {stats['total_gb']:.2f}GB")
    print(f"   Usage:     {stats['utilization_percent']:.1f}%")
    
    # Memory warnings
    if stats['free_gb'] < 1.0:
        print("⚠️  WARNING: Less than 1GB free memory!")
    if stats['utilization_percent'] > 90:
        print("⚠️  WARNING: GPU utilization > 90%!")

def test_pythia_memory_usage():
    """Test memory usage for Pythia-2.8B with different settings."""
    print("🧪 Testing Pythia-2.8B memory usage...")
    
    # Test configurations
    configs = [
        {'max_length': 256, 'batch_size': 1, 'name': 'Conservative'},
        {'max_length': 512, 'batch_size': 1, 'name': 'Balanced'},
        {'max_length': 1024, 'batch_size': 1, 'name': 'Aggressive'},
    ]
    
    for config in configs:
        print(f"\n📋 Testing {config['name']} config:")
        print(f"   Max length: {config['max_length']}")
        print(f"   Batch size: {config['batch_size']}")
        
        try:
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print_memory_stats()
            
            # Load model
            model = create_mathematical_reasoning_model(
                pe_method='rope',
                base_model='EleutherAI/pythia-2.8b',
                use_lora=True,
                load_in_4bit=False,
                enable_gradient_checkpointing=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            print("✅ Model loaded")
            print_memory_stats()
            
            # Test forward pass
            input_ids = torch.randint(0, 1000, (config['batch_size'], config['max_length']), dtype=torch.long)
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
            
            print("✅ Forward pass successful")
            print_memory_stats()
            
            # Test training step
            model.train()
            outputs = model(input_ids=input_ids, labels=input_ids.clone())
            loss = outputs.loss
            loss.backward()
            
            print("✅ Training step successful")
            print_memory_stats()
            
            # Clean up
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"✅ {config['name']} config: PASS")
            
        except Exception as e:
            print(f"❌ {config['name']} config: FAIL - {e}")
            
        time.sleep(2)  # Wait between tests

def get_optimal_settings():
    """Get optimal settings based on available memory."""
    stats = get_memory_stats()
    
    if 'error' in stats:
        return None
    
    total_gb = stats['total_gb']
    
    if total_gb >= 24:  # 24GB+ GPU
        return {
            'max_length': 1024,
            'batch_size': 2,
            'gradient_accumulation_steps': 4,
            'enable_gradient_checkpointing': False
        }
    elif total_gb >= 16:  # 16GB GPU (your case)
        return {
            'max_length': 512,
            'batch_size': 1,
            'gradient_accumulation_steps': 8,
            'enable_gradient_checkpointing': True
        }
    elif total_gb >= 12:  # 12GB GPU
        return {
            'max_length': 256,
            'batch_size': 1,
            'gradient_accumulation_steps': 16,
            'enable_gradient_checkpointing': True
        }
    else:  # < 12GB GPU
        return {
            'max_length': 128,
            'batch_size': 1,
            'gradient_accumulation_steps': 32,
            'enable_gradient_checkpointing': True
        }

if __name__ == "__main__":
    print("🔍 Pythia-2.8B Memory Monitor")
    print("=" * 40)
    
    print("\n📊 Current Memory Status:")
    print_memory_stats()
    
    print("\n🎯 Optimal Settings for Your GPU:")
    optimal = get_optimal_settings()
    if optimal:
        for key, value in optimal.items():
            print(f"   {key}: {value}")
    
    print("\n🧪 Running Memory Tests...")
    test_pythia_memory_usage()
    
    print("\n💡 Recommendations:")
    print("   - Use gradient checkpointing for memory savings")
    print("   - Start with single dataset (gsm8k)")
    print("   - Monitor memory during training")
    print("   - Use --memory_efficient flag")
    print("   - Consider reducing max_length if needed") 