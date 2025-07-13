"""Main entry point for the modular transformer.
Demonstrates basic usage and provides CLI interface.
"""
import argparse
import torch
from typing import Dict, Any

from src.model import TransformerModel
from config import get_config, create_experiment_configs


def demo_basic_usage():
    """Demonstrate basic usage of the modular transformer."""
    print("=== Modular Transformer Demo ===\n")
    
    # Create a basic configuration
    config = get_config("sinusoidal")
    model_config = config.model.__dict__
    
    print("1. Creating model with sinusoidal positional encoding...")
    model = TransformerModel(model_config)
    print(f"   Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 4
    seq_len = 64
    src = torch.randint(1, model_config['vocab_size'], (batch_size, seq_len))
    tgt = torch.randint(1, model_config['vocab_size'], (batch_size, seq_len))
    
    model.eval()
    with torch.no_grad():
        output = model(src, tgt)
    
    print(f"   Input shape: {src.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output dtype: {output.dtype}")
    
    # Demonstrate switching positional encodings
    print("\n3. Switching positional encodings...")
    pe_types = ["rope", "alibi", "nope"]
    
    for pe_type in pe_types:
        try:
            model.switch_positional_encoding(pe_type)
            print(f"   ✓ Successfully switched to {pe_type}")
            
            # Test forward pass with new encoding
            with torch.no_grad():
                output = model(src, tgt)
            print(f"     Forward pass successful with {pe_type}")
        except Exception as e:
            print(f"   ✗ Failed to switch to {pe_type}: {e}")
    
    print("\n4. Model configuration:")
    for key, value in model_config.items():
        print(f"   {key}: {value}")


def demo_all_encodings():
    """Demonstrate all positional encoding types."""
    print("\n=== All Positional Encodings Demo ===\n")
    
    pe_types = ["sinusoidal", "rope", "alibi", "diet", "t5_relative", "nope"]
    
    for pe_type in pe_types:
        print(f"Testing {pe_type.upper()} positional encoding...")
        try:
            config = get_config(pe_type)
            model_config = config.model.__dict__
            model = TransformerModel(model_config)
            
            # Test forward pass
            batch_size = 2
            seq_len = 32
            src = torch.randint(1, model_config['vocab_size'], (batch_size, seq_len))
            tgt = torch.randint(1, model_config['vocab_size'], (batch_size, seq_len))
            
            model.eval()
            with torch.no_grad():
                output = model(src, tgt)
            
            print(f"   ✓ {pe_type}: Model created and tested successfully")
            print(f"     Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"     Output shape: {output.shape}")
            
        except Exception as e:
            print(f"   ✗ {pe_type}: Failed with error: {e}")
        print()


def compare_model_sizes():
    """Compare model sizes for different positional encodings."""
    print("\n=== Model Size Comparison ===\n")
    
    pe_types = ["sinusoidal", "rope", "alibi", "diet", "t5_relative", "nope"]
    sizes = {}
    
    for pe_type in pe_types:
        try:
            config = get_config(pe_type)
            model_config = config.model.__dict__
            model = TransformerModel(model_config)
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            sizes[pe_type] = {
                'total': total_params,
                'trainable': trainable_params,
                'size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
            }
            
        except Exception as e:
            print(f"Failed to create {pe_type} model: {e}")
    
    # Print comparison table
    print(f"{'PE Type':<12} {'Total Params':<12} {'Trainable':<12} {'Size (MB)':<10}")
    print("-" * 50)
    
    for pe_type, stats in sizes.items():
        print(f"{pe_type:<12} {stats['total']:<12,} {stats['trainable']:<12,} {stats['size_mb']:<10.2f}")


def test_device_compatibility():
    """Test device compatibility (CPU/GPU)."""
    print("\n=== Device Compatibility Test ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    config = get_config("sinusoidal")
    model_config = config.model.__dict__
    model = TransformerModel(model_config)
    
    try:
        # Move model to device
        model = model.to(device)
        print(f"✓ Model moved to {device}")
        
        # Create test data on device
        batch_size = 2
        seq_len = 32
        src = torch.randint(1, model_config['vocab_size'], (batch_size, seq_len)).to(device)
        tgt = torch.randint(1, model_config['vocab_size'], (batch_size, seq_len)).to(device)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(src, tgt)
        
        print(f"✓ Forward pass successful on {device}")
        print(f"   Output device: {output.device}")
        
        if torch.cuda.is_available():
            print(f"   GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"   GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            
    except Exception as e:
        print(f"✗ Device compatibility test failed: {e}")


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description="Modular Transformer Demo")
    parser.add_argument(
        "--demo", 
        choices=["basic", "all", "sizes", "device", "full"], 
        default="basic",
        help="Type of demo to run"
    )
    parser.add_argument(
        "--pe_type", 
        choices=["sinusoidal", "rope", "alibi", "diet", "t5_relative", "nope"],
        default="sinusoidal",
        help="Positional encoding type for basic demo"
    )
    
    args = parser.parse_args()
    
    if args.demo == "basic":
        demo_basic_usage()
    elif args.demo == "all":
        demo_all_encodings()
    elif args.demo == "sizes":
        compare_model_sizes()
    elif args.demo == "device":
        test_device_compatibility()
    elif args.demo == "full":
        demo_basic_usage()
        demo_all_encodings()
        compare_model_sizes()
        test_device_compatibility()


if __name__ == "__main__":
    main()