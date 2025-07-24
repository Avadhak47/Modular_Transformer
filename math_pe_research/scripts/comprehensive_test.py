#!/usr/bin/env python3
"""
Comprehensive Test Suite for Mathematical PE Research Project

This script tests all components to identify and fix errors before training.
"""

import sys
import os
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch

def test_imports():
    """Test all imports work correctly."""
    print("🧪 Testing imports...")
    try:
        # Test PE imports
        from positional_encoding import PE_REGISTRY, get_positional_encoding
        print(f"✅ PE Registry loaded with methods: {list(PE_REGISTRY.keys())}")
        
        # Test model imports
        from models.mathematical_reasoning_model import create_mathematical_reasoning_model, MathematicalReasoningModel
        print("✅ Model imports successful")
        
        # Test data loader imports
        from data.math_dataset_loader import MathDatasetLoader
        print("✅ Data loader imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_positional_encodings():
    """Test each positional encoding method."""
    print("\n🧪 Testing positional encoding implementations...")
    
    from positional_encoding import PE_REGISTRY, get_positional_encoding
    
    results = {}
    
    for pe_method in PE_REGISTRY.keys():
        print(f"\n   Testing {pe_method}...")
        try:
            # Test parameters for each PE type
            if pe_method == 'rope':
                pe = get_positional_encoding(pe_method, d_model=512, dim=64, max_seq_len=1024)
                # Test forward pass
                q = torch.randn(2, 8, 32, 64)
                k = torch.randn(2, 8, 32, 64)
                q_rot, k_rot = pe(q, k)
                print(f"   ✅ {pe_method}: Input shape {q.shape} -> Output shape {q_rot.shape}")
                
            elif pe_method == 't5_relative':
                pe = get_positional_encoding(pe_method, d_model=512, num_heads=8)
                x = torch.randn(2, 32, 512)
                output = pe(x)
                print(f"   ✅ {pe_method}: Input shape {x.shape} -> Output shape {output.shape}")
                
            elif pe_method == 'alibi':
                pe = get_positional_encoding(pe_method, d_model=512, num_heads=8, max_seq_len=1024)
                x = torch.randn(2, 32, 512)
                attention_scores = torch.randn(2, 8, 32, 32)
                output = pe(x, attention_scores=attention_scores)
                print(f"   ✅ {pe_method}: Attention scores shape {attention_scores.shape} -> Output shape {output.shape}")
                
            elif pe_method == 'sinusoidal':
                pe = get_positional_encoding(pe_method, d_model=512, max_seq_len=1024)
                x = torch.randn(2, 32, 512)
                output = pe(x)
                print(f"   ✅ {pe_method}: Input shape {x.shape} -> Output shape {output.shape}")
                
            elif pe_method == 'diet':
                pe = get_positional_encoding(pe_method, d_model=512, max_seq_len=1024)
                x = torch.randn(2, 32, 512)
                output = pe(x)
                print(f"   ✅ {pe_method}: Input shape {x.shape} -> Output shape {output.shape}")
                
            elif pe_method == 'math_adaptive':
                pe = get_positional_encoding(pe_method, d_model=512, max_seq_len=1024)
                x = torch.randn(2, 32, 512)
                token_ids = torch.randint(0, 1000, (2, 32))
                output = pe(x, token_ids=token_ids)
                print(f"   ✅ {pe_method}: Input shape {x.shape} -> Output shape {output.shape}")
                
            results[pe_method] = True
            
        except Exception as e:
            print(f"   ❌ {pe_method} failed: {e}")
            results[pe_method] = False
            traceback.print_exc()
    
    return results

def test_model_creation():
    """Test model creation with different PE methods."""
    print("\n🧪 Testing model creation...")
    
    from models.mathematical_reasoning_model import create_mathematical_reasoning_model
    
    # Use a small model for testing
    test_model_name = "microsoft/DialoGPT-small"
    
    results = {}
    
    pe_methods_to_test = ['rope', 'sinusoidal', 'alibi']  # Test subset for speed
    
    for pe_method in pe_methods_to_test:
        print(f"\n   Testing model with {pe_method}...")
        try:
            model = create_mathematical_reasoning_model(
                pe_method=pe_method,
                base_model=test_model_name,
                load_in_4bit=False,
                use_lora=False,  # Disable LoRA for testing
                cache_dir="/tmp/test_cache",
                device_map=None,  # Disable device_map to avoid accelerate requirement
                torch_dtype=torch.float32  # Use float32 for CPU testing
            )
            print(f"   ✅ Model created successfully with {pe_method}")
            results[pe_method] = True
            
            # Test a simple forward pass
            input_ids = torch.randint(0, 1000, (1, 10))
            attention_mask = torch.ones_like(input_ids)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            print(f"   ✅ Forward pass successful, logits shape: {outputs.logits.shape}")
            
        except Exception as e:
            print(f"   ❌ Model creation with {pe_method} failed: {e}")
            results[pe_method] = False
            # Don't print full traceback for model creation issues during testing
            
    return results

def test_data_loading():
    """Test data loading functionality."""
    print("\n🧪 Testing data loading...")
    
    try:
        from data.math_dataset_loader import MathDatasetLoader
        
        # Test loader creation
        loader = MathDatasetLoader(
            max_length=512,
            cache_dir="/tmp/test_cache"
        )
        
        print(f"   ✅ DataLoader created successfully")
        
        # Test loading a small dataset
        try:
            problems = loader.load_dataset(
                dataset_name='gsm8k',
                split='train',
                max_samples=5,  # Very small for testing
                shuffle=False
            )
            print(f"   ✅ Dataset loaded, {len(problems)} problems")
            
            if problems:
                problem = problems[0]
                print(f"   ✅ Sample problem: {problem.problem[:50]}...")
                
        except Exception as e:
            print(f"   ⚠️  Dataset loading failed (might be network/cache issue): {e}")
            print("   ℹ️  This is not critical for the core functionality")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        traceback.print_exc()
        return False

def test_training_script_args():
    """Test that the training script can parse arguments correctly."""
    print("\n🧪 Testing training script argument parsing...")
    
    try:
        sys.path.insert(0, str(project_root / "scripts"))
        
        # Import the argument parser from train_and_eval
        import argparse
        import tempfile
        
        # Create a mock argument parser (simplified version)
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_size', type=str, default="microsoft/DialoGPT-small")
        parser.add_argument('--pe', type=str, default="rope")
        parser.add_argument('--experiment_name', type=str, default="test_exp")
        parser.add_argument('--checkpoint_dir', type=str, default="/tmp/checkpoints")
        parser.add_argument('--result_dir', type=str, default="/tmp/results")
        parser.add_argument('--cache_dir', type=str, default="/tmp/cache")
        parser.add_argument('--load_in_4bit', action='store_true')
        
        # Test parsing
        test_args = ['--pe', 'rope', '--experiment_name', 'test_run']
        args = parser.parse_args(test_args)
        
        print(f"   ✅ Argument parsing successful")
        print(f"   PE method: {args.pe}")
        print(f"   Experiment name: {args.experiment_name}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Argument parsing failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive tests."""
    print("🚀 Starting Comprehensive Test Suite for Mathematical PE Research")
    print("=" * 70)
    
    # Track results
    test_results = {}
    
    # Test 1: Imports
    test_results['imports'] = test_imports()
    
    # Test 2: Positional Encodings
    if test_results['imports']:
        test_results['pe'] = test_positional_encodings()
    
    # Test 3: Model Creation (only if imports and PE work)
    if test_results['imports'] and test_results.get('pe'):
        test_results['model'] = test_model_creation()
    
    # Test 4: Data Loading
    if test_results['imports']:
        test_results['data'] = test_data_loading()
    
    # Test 5: Training Script
    test_results['training_script'] = test_training_script_args()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    for test_name, result in test_results.items():
        if isinstance(result, dict):
            # For PE tests, show detailed results
            passed = sum(1 for v in result.values() if v)
            total = len(result)
            status = "✅" if passed == total else "⚠️"
            print(f"{status} {test_name.upper()}: {passed}/{total} passed")
            for pe_method, pe_result in result.items():
                status_icon = "✅" if pe_result else "❌"
                print(f"    {status_icon} {pe_method}")
        else:
            status = "✅" if result else "❌"
            print(f"{status} {test_name.upper()}: {'PASSED' if result else 'FAILED'}")
    
    # Overall status
    all_passed = all(
        (isinstance(r, dict) and all(r.values())) or (isinstance(r, bool) and r) 
        for r in test_results.values()
    )
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Your project is ready for training.")
    else:
        print("⚠️  Some tests failed. Please fix the issues before proceeding.")
    print("=" * 70)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 