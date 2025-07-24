#!/usr/bin/env python3
"""
Test script to verify dataset loading functionality.
"""
import sys
import os
sys.path.append(os.path.abspath('.'))

def test_dataset_loading():
    """Test the dataset loading functionality."""
    print("=== Testing Dataset Loading ===")
    
    try:
        from data.math_dataset_loader import MathematicalDatasetLoader
        print("✓ MathematicalDatasetLoader imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MathematicalDatasetLoader: {e}")
        return False
    
    try:
        # Initialize the loader
        loader = MathematicalDatasetLoader(tokenizer_name="gpt2", max_length=1024)
        print("✓ MathematicalDatasetLoader initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize MathematicalDatasetLoader: {e}")
        return False
    
    # Test GSM8K loading
    print("\n--- Testing GSM8K Dataset Loading ---")
    try:
        gsm8k_problems = loader.load_gsm8k_dataset("train")
        print(f"GSM8K train problems loaded: {len(gsm8k_problems)}")
        
        if len(gsm8k_problems) > 0:
            print("✓ GSM8K dataset loading successful")
            # Show first problem
            first_problem = gsm8k_problems[0]
            print(f"Sample problem: {first_problem.problem[:100]}...")
            print(f"Sample solution: {first_problem.solution[:100]}...")
            print(f"Final answer: {first_problem.final_answer}")
        else:
            print("✗ GSM8K dataset loading failed - no problems loaded")
            return False
    except Exception as e:
        print(f"✗ GSM8K dataset loading failed: {e}")
        return False
    
    # Test MATH dataset loading
    print("\n--- Testing MATH Dataset Loading ---")
    try:
        math_problems = loader.load_math_dataset("train", max_samples=100)
        print(f"MATH train problems loaded: {len(math_problems)}")
        
        if len(math_problems) > 0:
            print("✓ MATH dataset loading successful")
            # Show first problem
            first_problem = math_problems[0]
            print(f"Sample problem: {first_problem.problem[:100]}...")
            print(f"Sample solution: {first_problem.solution[:100]}...")
            print(f"Final answer: {first_problem.final_answer}")
        else:
            print("✗ MATH dataset loading failed - no problems loaded")
            return False
    except Exception as e:
        print(f"✗ MATH dataset loading failed: {e}")
        return False
    
    print("\n=== All Dataset Loading Tests Passed ===")
    return True

if __name__ == "__main__":
    success = test_dataset_loading()
    if success:
        print("✅ Dataset loading is working correctly!")
    else:
        print("❌ Dataset loading has issues that need to be fixed.")
        sys.exit(1) 