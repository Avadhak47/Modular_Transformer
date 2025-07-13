#!/usr/bin/env python3
"""
Test script for evaluation metrics implementation.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from evaluation.mathematical_metrics import MathematicalReasoningEvaluator
from src.model import TransformerModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_evaluation_metrics():
    """Test all evaluation metrics functionality."""
    print("=" * 60)
    print("EVALUATION METRICS IMPLEMENTATION AUDIT")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = MathematicalReasoningEvaluator()
    
    # Test data
    predictions = [
        "The answer is 42",
        "x = 15",
        "\\boxed{3.14}",
        "The solution is 2/3",
        "Result: 100"
    ]
    
    ground_truths = [
        "The answer is 42",
        "x = 15",
        "\\boxed{3.14}",
        "The solution is 2/3",
        "Result: 99"
    ]
    
    reasoning_chains = [
        ["Let x be the unknown", "x + 5 = 20", "x = 15"],
        ["Given the equation", "2x = 30", "x = 15"],
        ["We have", "π ≈ 3.14", "\\boxed{3.14}"],
        ["The fraction is", "4/6 = 2/3", "The solution is 2/3"],
        ["Calculate", "50 + 50 = 100", "Result: 100"]
    ]
    
    problems = [
        "Solve for x: x + 5 = 20",
        "Find x if 2x = 30",
        "What is π to 2 decimal places?",
        "Simplify 4/6",
        "What is 50 + 50?"
    ]
    
    solutions = [
        "x = 15",
        "x = 15",
        "\\boxed{3.14}",
        "2/3",
        "100"
    ]
    
    # Test 1: Exact Match Accuracy
    print("\n1. Testing Exact Match Accuracy...")
    try:
        accuracy_results = evaluator.exact_match_accuracy(predictions, ground_truths)
        print(f"✓ Exact match accuracy: {accuracy_results['exact_match_accuracy']:.3f}")
        print(f"✓ Numerical match accuracy: {accuracy_results['numerical_match_accuracy']:.3f}")
        print(f"✓ Normalized match accuracy: {accuracy_results['normalized_match_accuracy']:.3f}")
        print(f"✓ Total samples: {accuracy_results['total_samples']}")
        
        # Verify expected results
        expected_exact = 4/5  # 4 out of 5 exact matches
        expected_numerical = 4/5  # 4 out of 5 numerical matches (last one differs)
        
        assert abs(accuracy_results['exact_match_accuracy'] - expected_exact) < 0.01, \
            f"Expected exact accuracy {expected_exact}, got {accuracy_results['exact_match_accuracy']}"
        assert abs(accuracy_results['numerical_match_accuracy'] - expected_numerical) < 0.01, \
            f"Expected numerical accuracy {expected_numerical}, got {accuracy_results['numerical_match_accuracy']}"
        
        print("✓ Exact match accuracy test passed")
        
    except Exception as e:
        print(f"✗ Exact match accuracy test failed: {e}")
        return False
    
    # Test 2: Reasoning Step Correctness
    print("\n2. Testing Reasoning Step Correctness...")
    try:
        reasoning_results = evaluator.reasoning_step_correctness(reasoning_chains, problems, solutions)
        print(f"✓ Reasoning step correctness: {reasoning_results['reasoning_step_correctness']:.3f}")
        print(f"✓ Logical validity: {reasoning_results['logical_validity']:.3f}")
        print(f"✓ Informativeness: {reasoning_results['informativeness']:.3f}")
        print(f"✓ Total steps: {reasoning_results['total_steps']}")
        
        # Verify we have steps to evaluate
        assert reasoning_results['total_steps'] > 0, "No reasoning steps found"
        assert reasoning_results['reasoning_step_correctness'] >= 0.0, "Invalid correctness score"
        assert reasoning_results['reasoning_step_correctness'] <= 1.0, "Invalid correctness score"
        
        print("✓ Reasoning step correctness test passed")
        
    except Exception as e:
        print(f"✗ Reasoning step correctness test failed: {e}")
        return False
    
    # Test 3: Perplexity Calculation
    print("\n3. Testing Perplexity Calculation...")
    try:
        # Create a simple model for testing
        config = {
            'vocab_size': 50257,  # GPT-2 vocab size
            'd_model': 128,
            'n_heads': 8,
            'n_encoder_layers': 2,
            'n_decoder_layers': 2,
            'd_ff': 512,
            'max_seq_len': 512,
            'dropout': 0.1,
            'positional_encoding': 'sinusoidal'
        }
        
        model = TransformerModel(config)
        model.eval()
        
        test_texts = [
            "Solve for x: x + 5 = 20",
            "The answer is 15",
            "Let x be the unknown variable"
        ]
        
        perplexity_results = evaluator.calculate_perplexity(model, test_texts)
        print(f"✓ Overall perplexity: {perplexity_results['perplexity']:.3f}")
        print(f"✓ Mean perplexity: {perplexity_results['mean_perplexity']:.3f}")
        print(f"✓ Total tokens: {perplexity_results['total_tokens']}")
        print(f"✓ Number of sequences: {perplexity_results['num_sequences']}")
        
        # Verify perplexity is reasonable (untrained model will have high perplexity)
        assert perplexity_results['perplexity'] > 1.0, "Perplexity should be > 1"
        assert perplexity_results['perplexity'] < 100000.0, "Perplexity seems too high for untrained model"
        assert perplexity_results['total_tokens'] > 0, "No tokens processed"
        
        print("✓ Perplexity calculation test passed")
        
    except Exception as e:
        print(f"✗ Perplexity calculation test failed: {e}")
        return False
    
    # Test 4: Attention Entropy
    print("\n4. Testing Attention Entropy...")
    try:
        # Create dummy attention weights
        batch_size, n_heads, seq_len = 2, 8, 16
        attention_weights = torch.randn(batch_size, n_heads, seq_len, seq_len)
        
        # Apply softmax to make them valid attention distributions
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        entropy_results = evaluator.attention_entropy(attention_weights)
        print(f"✓ Mean attention entropy: {entropy_results['mean_attention_entropy']:.3f}")
        print(f"✓ Head entropy mean: {entropy_results['head_entropy_mean']:.3f}")
        print(f"✓ Position entropy mean: {entropy_results['position_entropy_mean']:.3f}")
        print(f"✓ Normalized entropy: {entropy_results['normalized_entropy']:.3f}")
        print(f"✓ Entropy efficiency: {entropy_results['entropy_efficiency']:.3f}")
        
        # Verify entropy values are reasonable
        max_entropy = np.log(seq_len)
        assert entropy_results['mean_attention_entropy'] >= 0.0, "Negative entropy"
        assert entropy_results['mean_attention_entropy'] <= max_entropy, "Entropy exceeds maximum"
        assert entropy_results['normalized_entropy'] >= 0.0, "Invalid normalized entropy"
        assert entropy_results['normalized_entropy'] <= 1.0, "Invalid normalized entropy"
        
        print("✓ Attention entropy test passed")
        
    except Exception as e:
        print(f"✗ Attention entropy test failed: {e}")
        return False
    
    # Test 5: Comprehensive Evaluation
    print("\n5. Testing Comprehensive Evaluation...")
    try:
        # Create dummy attention weights for comprehensive test
        attention_weights = torch.randn(1, 8, 16, 16)
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        comprehensive_results = evaluator.comprehensive_evaluation(
            model=model,
            predictions=predictions,
            ground_truths=ground_truths,
            reasoning_chains=reasoning_chains,
            problems=problems,
            solutions=solutions,
            attention_weights=attention_weights
        )
        
        print("✓ Comprehensive evaluation completed")
        print(f"  - Overall accuracy: {comprehensive_results['summary']['overall_accuracy']:.3f}")
        print(f"  - Numerical accuracy: {comprehensive_results['summary']['numerical_accuracy']:.3f}")
        print(f"  - Step correctness: {comprehensive_results['summary']['step_correctness']:.3f}")
        print(f"  - Logical validity: {comprehensive_results['summary']['logical_validity']:.3f}")
        print(f"  - Perplexity: {comprehensive_results['summary']['perplexity']:.3f}")
        print(f"  - Mean attention entropy: {comprehensive_results['summary']['mean_attention_entropy']:.3f}")
        
        # Verify all required metrics are present
        required_metrics = ['overall_accuracy', 'numerical_accuracy', 'step_correctness', 
                          'logical_validity', 'perplexity', 'mean_attention_entropy']
        for metric in required_metrics:
            assert metric in comprehensive_results['summary'], f"Missing metric: {metric}"
        
        print("✓ Comprehensive evaluation test passed")
        
    except Exception as e:
        print(f"✗ Comprehensive evaluation test failed: {e}")
        return False
    
    # Test 6: Edge Cases
    print("\n6. Testing Edge Cases...")
    try:
        # Empty predictions
        empty_results = evaluator.exact_match_accuracy([], [])
        assert empty_results['total_samples'] == 0, "Empty list should have 0 samples"
        
        # Single prediction
        single_results = evaluator.exact_match_accuracy(["test"], ["test"])
        assert single_results['exact_match_accuracy'] == 1.0, "Exact match should be 1.0"
        
        # Mismatched lengths
        try:
            evaluator.exact_match_accuracy(["a"], ["a", "b"])
            assert False, "Should raise ValueError for mismatched lengths"
        except ValueError:
            pass  # Expected
        
        print("✓ Edge cases test passed")
        
    except Exception as e:
        print(f"✗ Edge cases test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ ALL EVALUATION METRICS TESTS PASSED")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_evaluation_metrics()
    sys.exit(0 if success else 1) 