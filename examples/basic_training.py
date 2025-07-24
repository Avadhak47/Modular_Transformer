"""
Basic training example for the modular transformer.
"""
import torch
from src.model import TransformerModel
from config import get_config


def main():
    """Basic training example."""
    print("=== Basic Training Example ===")
    
    # Get configuration
    config = get_config("sinusoidal")
    model_config = config.model.__dict__
    
    # Create model
    model = TransformerModel(model_config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy data
    batch_size = 4
    seq_len = 64
    vocab_size = model_config['vocab_size']
    
    src = torch.randint(1, vocab_size, (batch_size, seq_len))
    tgt = torch.randint(1, vocab_size, (batch_size, seq_len))
    
    # Training setup
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    model.train()
    for step in range(10):
        optimizer.zero_grad()
        
        # Forward pass
        # Guard against short sequences
        if tgt.size(1) < 2:
            raise ValueError("Target sequence too short for teacher forcing (must be at least 2 timesteps)")
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Step {step + 1}: Loss = {loss.item():.4f}")
    
    print("Basic training completed!")


if __name__ == "__main__":
    main()
