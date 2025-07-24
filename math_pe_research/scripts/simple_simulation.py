#!/usr/bin/env python3
"""
Simplified Experiment Simulation for Mathematical Reasoning PE Research

This script tests core components without requiring external dependencies.
It identifies basic issues in the experiment pipeline.

Usage: python3 simple_simulation.py [--pe_method rope]
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        from transformers import AutoTokenizer
        print("✅ Transformers available")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    return True

def test_positional_encoding(pe_method="rope"):
    """Test positional encoding implementation."""
    print(f"🧪 Testing positional encoding: {pe_method}")
    try:
        from positional_encoding import get_positional_encoding, PE_REGISTRY
        if pe_method not in PE_REGISTRY:
            print(f"❌ PE method '{pe_method}' not in registry: {list(PE_REGISTRY.keys())}")
            return False
        # Prepare kwargs for each PE type
        if pe_method == 't5_relative':
            pe_layer = get_positional_encoding(
                pe_method,
                d_model=512,
                num_heads=8,
                relative_attention_num_buckets=32,
                relative_attention_max_distance=128,
                bidirectional=True
            )
        elif pe_method == 'alibi':
            pe_layer = get_positional_encoding(
                pe_method,
                d_model=512,
                num_heads=8,
                max_seq_len=1024
            )
        elif pe_method == 'diet':
            pe_layer = get_positional_encoding(
                pe_method,
                d_model=512,
                max_seq_len=1024
            )
        elif pe_method == 'math_adaptive':
            pe_layer = get_positional_encoding(
                pe_method,
                d_model=512,
                max_seq_len=1024
            )
        else:
            pe_layer = get_positional_encoding(
                pe_method,
                d_model=512,
                max_seq_len=1024
            )
        print(f"✅ Created {pe_method} PE layer")
        # Test forward pass
        import torch
        if pe_method in ['rope', 'math_adaptive']:
            x = torch.randn(2, 64, 8, 64)  # (B, L, H, D)
            output = pe_layer(x)
            print(f"✅ {pe_method} forward pass: {x.shape} -> {output.shape}")
        elif pe_method == 't5_relative':
            x = torch.randn(2, 64, 512)
            attention_scores = torch.randn(2, 8, 64, 64)
            output = pe_layer(x, attention_scores=attention_scores)
            print(f"✅ {pe_method} forward pass with attention_scores: {attention_scores.shape} -> {output.shape}")
        else:
            x = torch.randn(2, 64, 512)
            output = pe_layer(x)
            print(f"✅ {pe_method} forward pass: {x.shape} -> {output.shape}")
        return True
    except Exception as e:
        print(f"❌ PE test failed: {e}")
        if "--verbose" in sys.argv:
            import traceback
            traceback.print_exc()
        return False

def test_all_positional_encodings():
    """Test all positional encoding types in the registry."""
    from positional_encoding import PE_REGISTRY
    all_passed = True
    for pe_method in PE_REGISTRY:
        print(f"\n=== Testing PE: {pe_method} ===")
        passed = test_positional_encoding(pe_method)
        all_passed = all_passed and passed
    return all_passed

def test_model_creation():
    """Test mathematical reasoning model creation."""
    print("🧪 Testing model creation...")
    
    try:
        from models.mathematical_reasoning_model import create_mathematical_reasoning_model
        
        # Use a very small model for testing
        model = create_mathematical_reasoning_model(
            pe_method="rope",
            base_model="microsoft/DialoGPT-small", 
            use_lora=True,
            load_in_4bit=False,
            device_map=None
        )
        
        print("✅ Model created successfully")
        
        # Test basic forward pass
        test_text = "What is 2 + 2?"
        inputs = model.tokenizer(test_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✅ Model forward pass successful")
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return False

def test_dataset_loading():
    """Test dataset loading functionality."""
    print("🧪 Testing dataset loading...")
    
    try:
        from data.math_dataset_loader import MathDatasetLoader, create_demo_dataset
        from transformers import AutoTokenizer
        
        # Create demo dataset
        problems = create_demo_dataset()
        print(f"✅ Demo dataset created: {len(problems)} problems")
        
        # Test data loader
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        loader = MathDatasetLoader(tokenizer=tokenizer, max_length=512)
        dataset = loader.create_pytorch_dataset(problems, is_training=True)
        
        print(f"✅ PyTorch dataset created: {len(dataset)} items")
        
        # Test sample
        sample = dataset[0]
        print(f"✅ Dataset sample keys: {list(sample.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return False

def test_training_setup():
    """Test training configuration."""
    print("🧪 Testing training setup...")
    
    try:
        from transformers import TrainingArguments
        
        training_args = TrainingArguments(
            output_dir="./test_output",
            max_steps=10,
            per_device_train_batch_size=1,
            learning_rate=1e-4,
            logging_steps=5,
            save_strategy="no",
            eval_strategy="no",  # Updated parameter name
            report_to=[]
        )
        
        print("✅ Training arguments created")
        return True
        
    except Exception as e:
        print(f"❌ Training setup failed: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return False

def test_file_structure():
    """Test if all required files exist."""
    print("🧪 Testing file structure...")
    
    required_files = [
        "src/positional_encoding/__init__.py",
        "src/positional_encoding/rope.py", 
        "src/positional_encoding/math_adaptive.py",
        "src/models/mathematical_reasoning_model.py",
        "src/data/math_dataset_loader.py",
        "scripts/run_experiment.sh",
        "requirements.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def run_simulation(pe_method="rope"):
    """Run the complete simulation."""
    
    print("🚀 Starting Mathematical Reasoning PE Research Simulation")
    print("=" * 60)
    print(f"PE Method: {pe_method}")
    print(f"Python: {sys.version}")
    print("=" * 60)
    
    results = {}
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("All Positional Encodings", test_all_positional_encodings),
        ("Dataset Loading", test_dataset_loading),
        ("Training Setup", test_training_setup),
        ("Model Creation", test_model_creation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        start_time = time.time()
        
        try:
            success = test_func()
            results[test_name] = {
                "success": success,
                "time": time.time() - start_time
            }
            if success:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = {
                "success": False,
                "time": time.time() - start_time,
                "error": str(e)
            }
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    for test_name, result in results.items():
        status = "✅" if result["success"] else "❌"
        print(f"{status} {test_name:<20} ({result['time']:.2f}s)")
    
    # Assessment
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Ready for deployment!")
        readiness = "READY"
    elif passed >= total * 0.8:
        print("\n⚠️  MOSTLY READY - Minor issues to address")
        readiness = "READY_WITH_WARNINGS"
    else:
        print("\n❌ NOT READY - Critical issues need fixing")
        readiness = "NOT_READY"
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if not results.get("Imports", {}).get("success", False):
        print("   • Install missing Python packages")
    
    if not results.get("Model Creation", {}).get("success", False):
        print("   • Check model loading and HuggingFace access")
        print("   • Consider using smaller models or CPU-only mode")
    
    if not results.get("Positional Encoding", {}).get("success", False):
        print("   • Fix positional encoding implementation")
        print("   • Check tensor shapes and device placement")
    
    print("\n📋 NEXT STEPS:")
    if readiness == "READY":
        print("   1. Run full experiment: ./scripts/run_experiment.sh")
        print("   2. Monitor with: tail -f logs/training.log")
    else:
        print("   1. Fix failed tests")
        print("   2. Re-run simulation")
        print("   3. Check error logs for details")
    
    return readiness == "READY"

def main():
    """Main function."""
    pe_method = "rope"
    
    # Parse simple arguments
    if "--pe_method" in sys.argv:
        idx = sys.argv.index("--pe_method")
        if idx + 1 < len(sys.argv):
            pe_method = sys.argv[idx + 1]
    
    success = run_simulation(pe_method)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()