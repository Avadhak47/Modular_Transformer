"""
Evaluation script for the modular transformer.
Supports comprehensive model evaluation and comparison.
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import math
from typing import Dict, Any, List, Tuple
import time

from src.model import TransformerModel
from src.utils.metrics import calculate_perplexity, calculate_bleu
from config import get_config, ExperimentConfig
from train import DummyDataset, create_data_loaders


class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, model_path: str, config: ExperimentConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = TransformerModel(config.model.__dict__).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        print(f"Model loaded from {model_path}")
        print(f"Positional encoding: {config.model.positional_encoding}")
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on given data loader."""
        total_loss = 0
        total_samples = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                logits = self.model(src, tgt_input)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
                
                total_loss += loss.item() * src.size(0)
                total_samples += src.size(0)
                num_batches += 1
        
        avg_loss = total_loss / total_samples
        perplexity = math.exp(min(avg_loss, 100))
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'num_samples': total_samples,
            'num_batches': num_batches
        }
    
    def evaluate_length_generalization(self, vocab_size: int, test_lengths: List[int]) -> Dict[int, Dict[str, float]]:
        """Evaluate model's ability to generalize to different sequence lengths."""
        results = {}
        
        for length in test_lengths:
            print(f"Evaluating on length {length}...")
            
            # Create dataset with specific length
            test_dataset = DummyDataset(vocab_size, length, num_samples=500)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            
            # Evaluate
            metrics = self.evaluate(test_loader)
            results[length] = metrics
            
            print(f"Length {length}: Loss = {metrics['loss']:.4f}, Perplexity = {metrics['perplexity']:.4f}")
        
        return results
    
    def benchmark_inference_speed(self, batch_sizes: List[int], seq_len: int = 512) -> Dict[int, Dict[str, float]]:
        """Benchmark inference speed for different batch sizes."""
        results = {}
        
        for batch_size in batch_sizes:
            print(f"Benchmarking batch size {batch_size}...")
            
            # Create dummy data
            src = torch.randint(1, self.config.model.vocab_size, (batch_size, seq_len)).to(self.device)
            tgt = torch.randint(1, self.config.model.vocab_size, (batch_size, seq_len)).to(self.device)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = self.model(src, tgt)
            
            # Benchmark
            start_time = time.time()
            num_runs = 100
            
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = self.model(src, tgt)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            results[batch_size] = {
                'total_time': total_time,
                'avg_time_per_batch': total_time / num_runs,
                'samples_per_second': (batch_size * num_runs) / total_time,
                'tokens_per_second': (batch_size * seq_len * num_runs) / total_time
            }
            
            print(f"Batch size {batch_size}: {results[batch_size]['samples_per_second']:.2f} samples/sec")
        
        return results
    
    def analyze_attention_patterns(self, src: torch.Tensor, tgt: torch.Tensor) -> Dict[str, Any]:
        """Analyze attention patterns (simplified version)."""
        # This is a simplified version - in practice, you'd need to modify
        # the model to return attention weights
        self.model.eval()
        
        with torch.no_grad():
            logits = self.model(src, tgt)
        
        # Placeholder for attention analysis
        return {
            'output_shape': logits.shape,
            'max_attention': "Not implemented - would need model modification",
            'attention_entropy': "Not implemented - would need model modification"
        }


def compare_models(model_paths: List[str], test_data_path: str) -> Dict[str, Any]:
    """Compare multiple models on the same test data."""
    results = {}
    
    for model_path in model_paths:
        print(f"\nEvaluating {model_path}...")
        
        # Load checkpoint to get config
        checkpoint = torch.load(model_path, map_location='cpu')
        config = ExperimentConfig.from_dict(checkpoint['config'])
        
        # Create evaluator
        evaluator = ModelEvaluator(model_path, config)
        
        # Create test data loader
        test_dataset = DummyDataset(
            vocab_size=config.model.vocab_size,
            seq_len=config.model.max_seq_len,
            num_samples=1000
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Evaluate
        metrics = evaluator.evaluate(test_loader)
        
        # Store results
        pe_type = config.model.positional_encoding
        results[pe_type] = {
            'model_path': model_path,
            'config': config.to_dict(),
            'metrics': metrics
        }
        
        print(f"{pe_type}: Loss = {metrics['loss']:.4f}, Perplexity = {metrics['perplexity']:.4f}")
    
    return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Modular Transformer')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, default=None,
                       help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--compare_models', type=str, nargs='+', default=None,
                       help='Paths to multiple models for comparison')
    parser.add_argument('--length_generalization', action='store_true',
                       help='Test length generalization')
    parser.add_argument('--benchmark_speed', action='store_true',
                       help='Benchmark inference speed')
    parser.add_argument('--analyze_attention', action='store_true',
                       help='Analyze attention patterns')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.compare_models:
        # Compare multiple models
        print("Comparing multiple models...")
        results = compare_models(args.compare_models, args.test_data)
        
        # Save comparison results
        with open(os.path.join(args.output_dir, 'model_comparison.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n=== Model Comparison Summary ===")
        for pe_type, result in results.items():
            metrics = result['metrics']
            print(f"{pe_type:>12}: Loss = {metrics['loss']:.4f}, Perplexity = {metrics['perplexity']:.4f}")
        
    else:
        # Single model evaluation
        print(f"Evaluating single model: {args.model_path}")
        
        # Load checkpoint to get config
        checkpoint = torch.load(args.model_path, map_location='cpu')
        config = ExperimentConfig.from_dict(checkpoint['config'])
        
        # Create evaluator
        evaluator = ModelEvaluator(args.model_path, config)
        
        # Basic evaluation
        print("\n=== Basic Evaluation ===")
        test_dataset = DummyDataset(
            vocab_size=config.model.vocab_size,
            seq_len=config.model.max_seq_len,
            num_samples=1000
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        basic_metrics = evaluator.evaluate(test_loader)
        print(f"Test Loss: {basic_metrics['loss']:.4f}")
        print(f"Test Perplexity: {basic_metrics['perplexity']:.4f}")
        
        results = {'basic_evaluation': basic_metrics}
        
        # Length generalization test
        if args.length_generalization:
            print("\n=== Length Generalization Test ===")
            test_lengths = [128, 256, 512, 1024]
            length_results = evaluator.evaluate_length_generalization(config.model.vocab_size, test_lengths)
            results['length_generalization'] = length_results
        
        # Speed benchmark
        if args.benchmark_speed:
            print("\n=== Speed Benchmark ===")
            batch_sizes = [1, 4, 8, 16, 32]
            speed_results = evaluator.benchmark_inference_speed(batch_sizes)
            results['speed_benchmark'] = speed_results
        
        # Attention analysis
        if args.analyze_attention:
            print("\n=== Attention Analysis ===")
            src = torch.randint(1, config.model.vocab_size, (2, 64))
            tgt = torch.randint(1, config.model.vocab_size, (2, 64))
            attention_results = evaluator.analyze_attention_patterns(src, tgt)
            results['attention_analysis'] = attention_results
        
        # Save results
        results_file = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()