"""
Compare different positional encoding methods.
"""
import torch
import time
from src.model import TransformerModel
from config import get_config


def benchmark_model(pe_type: str, num_steps: int = 5):
    """Benchmark a model with specific positional encoding."""
    print(f"\n--- Benchmarking {pe_type.upper()} ---")
    
    # Get configuration and create model
    config = get_config(pe_type)
    model_config = config.model.__dict__
    model = TransformerModel(model_config)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
    # Create test data
    batch_size = 4
    seq_len = 128
    vocab_size = model_config['vocab_size']
    
    src = torch.randint(1, vocab_size, (batch_size, seq_len))
    tgt = torch.randint(1, vocab_size, (batch_size, seq_len))
    
    # Benchmark forward pass
    model.eval()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_steps):
            output = model(src, tgt)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_steps
    
    print(f"Average forward pass time: {avg_time:.4f}s")
    print(f"Output shape: {output.shape}")
    
    return {
        'pe_type': pe_type,
        'parameters': total_params,
        'avg_time': avg_time,
        'output_shape': output.shape
    }


def main():
    """Compare different positional encoding methods."""
    print("=== Positional Encoding Comparison ===")
    
    pe_types = ["sinusoidal", "rope", "alibi", "nope"]
    results = []
    
    for pe_type in pe_types:
        try:
            result = benchmark_model(pe_type)
            results.append(result)
        except Exception as e:
            print(f"Failed to benchmark {pe_type}: {e}")
    
    # Summary
    print("\n=== Summary ===")
    print(f"{'PE Type':<12} {'Parameters':<12} {'Avg Time (s)':<15}")
    print("-" * 45)
    
    for result in results:
        print(f"{result['pe_type']:<12} {result['parameters']:<12,} {result['avg_time']:<15.4f}")


if __name__ == "__main__":
    main()
