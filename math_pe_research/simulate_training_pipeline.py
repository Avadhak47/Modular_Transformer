#!/usr/bin/env python3
"""
Training Pipeline Simulation with Real-World Mathematical Problems
Tests the fixed PE methods with actual mathematical reasoning tasks
"""

import torch
import sys
import os
import time
import json
from typing import List, Dict, Any
from dataclasses import dataclass

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.mathematical_reasoning_model import create_mathematical_reasoning_model, get_best_device


@dataclass
class MathProblem:
    """Represents a mathematical problem with solution."""
    problem: str
    solution: str
    category: str
    difficulty: str


class TrainingSimulator:
    """Simulates a training pipeline with real mathematical problems."""
    
    def __init__(self, pe_method: str = "rope", device: str = "cpu"):
        self.pe_method = pe_method
        self.device = device
        self.model = None
        self.tokenizer = None
        self.training_stats = {
            "total_problems": 0,
            "successful_forward_passes": 0,
            "successful_generations": 0,
            "device_errors": 0,
            "other_errors": 0,
            "avg_forward_time": 0.0,
            "avg_generation_time": 0.0
        }
        
        # Real-world mathematical problems for testing
        self.math_problems = [
            MathProblem(
                problem="What is 2 + 3?",
                solution="2 + 3 = 5",
                category="arithmetic",
                difficulty="easy"
            ),
            MathProblem(
                problem="Solve for x: 3x + 7 = 22",
                solution="3x + 7 = 22\n3x = 22 - 7\n3x = 15\nx = 5",
                category="algebra",
                difficulty="medium"
            ),
            MathProblem(
                problem="What is the area of a circle with radius 5?",
                solution="Area = πr² = π(5)² = 25π ≈ 78.54",
                category="geometry",
                difficulty="medium"
            ),
            MathProblem(
                problem="Find the derivative of f(x) = x² + 3x + 1",
                solution="f'(x) = 2x + 3",
                category="calculus",
                difficulty="medium"
            ),
            MathProblem(
                problem="What is 15% of 200?",
                solution="15% of 200 = 0.15 × 200 = 30",
                category="percentage",
                difficulty="easy"
            ),
            MathProblem(
                problem="Solve the quadratic equation: x² - 5x + 6 = 0",
                solution="x² - 5x + 6 = 0\n(x - 2)(x - 3) = 0\nx = 2 or x = 3",
                category="algebra",
                difficulty="medium"
            ),
            MathProblem(
                problem="What is the sum of the first 10 natural numbers?",
                solution="Sum = n(n+1)/2 = 10(11)/2 = 55",
                category="series",
                difficulty="easy"
            ),
            MathProblem(
                problem="Find the slope of the line passing through (2,3) and (4,7)",
                solution="Slope = (7-3)/(4-2) = 4/2 = 2",
                category="geometry",
                difficulty="easy"
            ),
            MathProblem(
                problem="What is log₂(8)?",
                solution="log₂(8) = 3 because 2³ = 8",
                category="logarithms",
                difficulty="medium"
            ),
            MathProblem(
                problem="Solve for x: 2^x = 16",
                solution="2^x = 16\n2^x = 2⁴\nx = 4",
                category="exponents",
                difficulty="medium"
            ),
            MathProblem(
                problem="What is the probability of rolling a 6 on a fair die?",
                solution="Probability = 1/6 ≈ 0.167",
                category="probability",
                difficulty="easy"
            ),
            MathProblem(
                problem="Find the integral of ∫(2x + 3)dx",
                solution="∫(2x + 3)dx = x² + 3x + C",
                category="calculus",
                difficulty="medium"
            ),
            MathProblem(
                problem="What is the volume of a cube with side length 4?",
                solution="Volume = s³ = 4³ = 64 cubic units",
                category="geometry",
                difficulty="easy"
            ),
            MathProblem(
                problem="Solve for x: √(x + 5) = 3",
                solution="√(x + 5) = 3\nx + 5 = 9\nx = 4",
                category="algebra",
                difficulty="medium"
            ),
            MathProblem(
                problem="What is the factorial of 5?",
                solution="5! = 5 × 4 × 3 × 2 × 1 = 120",
                category="combinatorics",
                difficulty="easy"
            ),
            MathProblem(
                problem="Find the midpoint between (1,2) and (5,8)",
                solution="Midpoint = ((1+5)/2, (2+8)/2) = (3,5)",
                category="geometry",
                difficulty="easy"
            ),
            MathProblem(
                problem="What is the value of sin(30°)?",
                solution="sin(30°) = 1/2 = 0.5",
                category="trigonometry",
                difficulty="easy"
            ),
            MathProblem(
                problem="Solve for x: e^x = 20",
                solution="e^x = 20\nx = ln(20) ≈ 3.0",
                category="logarithms",
                difficulty="medium"
            ),
            MathProblem(
                problem="What is the greatest common divisor of 48 and 36?",
                solution="GCD(48, 36) = 12",
                category="number_theory",
                difficulty="medium"
            ),
            MathProblem(
                problem="Find the equation of the line with slope 2 passing through (1,3)",
                solution="y - 3 = 2(x - 1)\ny = 2x - 2 + 3\ny = 2x + 1",
                category="geometry",
                difficulty="medium"
            )
        ]
    
    def setup_model(self):
        """Initialize the model with the specified PE method."""
        print(f"🔧 Setting up model with {self.pe_method.upper()} PE...")
        self.device = get_best_device()
        try:
            self.model = create_mathematical_reasoning_model(
                pe_method=self.pe_method,
                base_model="microsoft/DialoGPT-small",  # Small model for testing
                load_in_4bit=False,
                use_lora=False,  # Disable LoRA for simpler testing
                torch_dtype=torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            self.tokenizer = self.model.tokenizer
            print(f"✅ Model created successfully with {self.pe_method} PE")
            print(f"   Device: {next(self.model.parameters()).device if not hasattr(self.model, 'module') else next(self.model.module.parameters()).device}")
            print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        except Exception as e:
            print(f"❌ Failed to create model: {e}")
            raise
    
    def test_forward_pass(self, problem: MathProblem) -> Dict[str, Any]:
        """Test forward pass with a mathematical problem."""
        try:
            input_text = f"Solve: {problem.problem}"
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
            device = next(self.model.parameters()).device if not hasattr(self.model, 'module') else next(self.model.module.parameters()).device
            input_ids = input_ids.to(device)
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
            forward_time = time.time() - start_time
            return {
                "success": True,
                "forward_time": forward_time,
                "output_shape": outputs['logits'].shape,
                "device": outputs['logits'].device
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "forward_time": 0.0
            }
    
    def test_generation(self, problem: MathProblem) -> Dict[str, Any]:
        """Test text generation with a mathematical problem."""
        try:
            input_text = f"Solve: {problem.problem}"
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
            device = next(self.model.parameters()).device if not hasattr(self.model, 'module') else next(self.model.module.parameters()).device
            input_ids = input_ids.to(device)
            start_time = time.time()
            generated = self.model.generate(
                input_ids=input_ids,
                max_length=100,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generation_time = time.time() - start_time
            generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            return {
                "success": True,
                "generation_time": generation_time,
                "generated_text": generated_text,
                "input_length": input_ids.shape[1],
                "output_length": generated.shape[1]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "generation_time": 0.0
            }
    
    def run_training_simulation(self, num_problems: int = 10):
        """Run a complete training simulation."""
        print(f"\n🚀 Starting Training Simulation with {self.pe_method.upper()} PE")
        print("=" * 60)
        
        # Setup model
        self.setup_model()
        
        # Select problems for testing
        test_problems = self.math_problems[:num_problems]
        
        print(f"\n📚 Testing with {len(test_problems)} mathematical problems:")
        for i, problem in enumerate(test_problems, 1):
            print(f"   {i}. {problem.category.upper()}: {problem.problem}")
        
        print(f"\n🧪 Running tests...")
        print("-" * 40)
        
        forward_times = []
        generation_times = []
        
        for i, problem in enumerate(test_problems, 1):
            print(f"\n🔍 Problem {i}/{len(test_problems)}: {problem.category.upper()}")
            print(f"   Problem: {problem.problem}")
            
            # Test forward pass
            print("   Testing forward pass...")
            forward_result = self.test_forward_pass(problem)
            
            if forward_result["success"]:
                print(f"   ✅ Forward pass: {forward_result['forward_time']:.3f}s")
                print(f"      Output shape: {forward_result['output_shape']}")
                print(f"      Device: {forward_result['device']}")
                self.training_stats["successful_forward_passes"] += 1
                forward_times.append(forward_result["forward_time"])
            else:
                print(f"   ❌ Forward pass failed: {forward_result['error']}")
                if "device" in forward_result["error"].lower():
                    self.training_stats["device_errors"] += 1
                else:
                    self.training_stats["other_errors"] += 1
            
            # Test generation
            print("   Testing generation...")
            generation_result = self.test_generation(problem)
            
            if generation_result["success"]:
                print(f"   ✅ Generation: {generation_result['generation_time']:.3f}s")
                print(f"      Generated: {generation_result['generated_text'][:100]}...")
                self.training_stats["successful_generations"] += 1
                generation_times.append(generation_result["generation_time"])
            else:
                print(f"   ❌ Generation failed: {generation_result['error']}")
                if "device" in generation_result["error"].lower():
                    self.training_stats["device_errors"] += 1
                else:
                    self.training_stats["other_errors"] += 1
            
            self.training_stats["total_problems"] += 1
        
        # Calculate statistics
        if forward_times:
            self.training_stats["avg_forward_time"] = sum(forward_times) / len(forward_times)
        if generation_times:
            self.training_stats["avg_generation_time"] = sum(generation_times) / len(generation_times)
        
        # Print summary
        self.print_training_summary()
    
    def print_training_summary(self):
        """Print comprehensive training simulation summary."""
        print(f"\n{'='*60}")
        print(f"📊 TRAINING SIMULATION SUMMARY - {self.pe_method.upper()} PE")
        print(f"{'='*60}")
        
        print(f"🎯 Overall Results:")
        print(f"   Total Problems Tested: {self.training_stats['total_problems']}")
        print(f"   Successful Forward Passes: {self.training_stats['successful_forward_passes']}")
        print(f"   Successful Generations: {self.training_stats['successful_generations']}")
        print(f"   Device Errors: {self.training_stats['device_errors']}")
        print(f"   Other Errors: {self.training_stats['other_errors']}")
        
        print(f"\n⏱️ Performance Metrics:")
        print(f"   Average Forward Time: {self.training_stats['avg_forward_time']:.3f}s")
        print(f"   Average Generation Time: {self.training_stats['avg_generation_time']:.3f}s")
        
        print(f"\n📈 Success Rates:")
        forward_success_rate = (self.training_stats['successful_forward_passes'] / self.training_stats['total_problems']) * 100
        generation_success_rate = (self.training_stats['successful_generations'] / self.training_stats['total_problems']) * 100
        
        print(f"   Forward Pass Success Rate: {forward_success_rate:.1f}%")
        print(f"   Generation Success Rate: {generation_success_rate:.1f}%")
        
        print(f"\n🎉 Conclusion:")
        if self.training_stats['device_errors'] == 0:
            print(f"   ✅ NO DEVICE MISMATCH ERRORS!")
        else:
            print(f"   ⚠️ {self.training_stats['device_errors']} device errors encountered")
        
        if forward_success_rate >= 90 and generation_success_rate >= 80:
            print(f"   ✅ {self.pe_method.upper()} PE is READY for Kaggle T4 X2 training!")
        else:
            print(f"   ⚠️ {self.pe_method.upper()} PE needs further investigation")


def main():
    """Run training simulation for all PE methods."""
    print("🧪 Comprehensive Training Pipeline Simulation")
    print("Testing all PE methods with real mathematical problems")
    print("=" * 60)
    
    # Check device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🚀 Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print(f"💻 Using CPU device")
    
    # Test all PE methods
    pe_methods = ["rope", "sinusoidal", "t5_relative", "diet", "alibi", "math_adaptive"]
    
    results = {}
    
    for pe_method in pe_methods:
        print(f"\n{'='*60}")
        print(f"Testing {pe_method.upper()} PE")
        print(f"{'='*60}")
        
        try:
            simulator = TrainingSimulator(pe_method=pe_method, device=device)
            simulator.run_training_simulation(num_problems=20)
            results[pe_method] = simulator.training_stats
        except Exception as e:
            print(f"❌ Failed to test {pe_method}: {e}")
            results[pe_method] = {"error": str(e)}
    
    # Final summary
    print(f"\n{'='*60}")
    print("🏆 FINAL COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for pe_method, stats in results.items():
        if "error" in stats:
            print(f"❌ {pe_method.upper()}: {stats['error']}")
        else:
            forward_rate = (stats['successful_forward_passes'] / stats['total_problems']) * 100
            generation_rate = (stats['successful_generations'] / stats['total_problems']) * 100
            device_errors = stats['device_errors']
            
            status = "✅ READY" if device_errors == 0 and forward_rate >= 90 else "⚠️ ISSUES"
            print(f"{status} {pe_method.upper()}: Forward={forward_rate:.1f}%, Gen={generation_rate:.1f}%, Device_Errors={device_errors}")
    
    print(f"\n🎯 RECOMMENDATION:")
    ready_methods = [pe for pe, stats in results.items() 
                    if "error" not in stats and stats['device_errors'] == 0 and 
                    (stats['successful_forward_passes'] / stats['total_problems']) >= 0.9]
    
    if ready_methods:
        print(f"✅ Use these PE methods for Kaggle T4 X2: {', '.join(ready_methods).upper()}")
    else:
        print(f"⚠️ All PE methods have issues, need further debugging")


if __name__ == "__main__":
    main() 