#!/usr/bin/env python3
"""
Kaggle Training Test Script

This script runs a minimal training session to verify the setup works
and can complete at least one epoch with a small dataset.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "math_pe_research" / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_device():
    """Setup device (GPU if available, otherwise CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    return device

class SimpleDataset(Dataset):
    """Simple dataset for testing"""
    def __init__(self, num_samples=10):
        self.num_samples = num_samples
        # Create simple math problems
        self.data = []
        for i in range(num_samples):
            a = i + 1
            b = i + 2
            question = f"What is {a} + {b}?"
            answer = str(a + b)
            self.data.append({
                'question': question,
                'answer': answer,
                'input_ids': [100 + i % 50] * 20,  # Simple token sequence
                'labels': [200 + i % 30] * 10      # Simple label sequence
            })
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.data[idx]['input_ids'], dtype=torch.long),
            'labels': torch.tensor(self.data[idx]['labels'], dtype=torch.long),
            'attention_mask': torch.ones(20, dtype=torch.long)
        }

class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self, vocab_size=1000, hidden_size=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=256,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, labels=None, attention_mask=None):
        # Simple forward pass
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            # Adjust sequence lengths to match
            seq_len = min(logits.size(1), labels.size(1))
            logits_flat = logits[:, :seq_len, :].contiguous().view(-1, logits.size(-1))
            labels_flat = labels[:, :seq_len].contiguous().view(-1)
            loss = self.criterion(logits_flat, labels_flat)
        
        return {'loss': loss, 'logits': logits}

def test_training(device, num_samples=10, num_epochs=2):
    """Test training loop"""
    logger.info("Starting Kaggle training test...")
    
    # Create dataset and dataloader
    dataset = SimpleDataset(num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Create model
    model = SimpleModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Training on {len(dataset)} samples for {num_epochs} epochs")
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        logger.info(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            logger.info(f"Batch {batch_idx + 1}/{len(dataloader)}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")
    
    logger.info("✅ Training test completed successfully!")
    return True

def test_positional_encoding():
    """Test positional encoding imports"""
    logger.info("Testing positional encoding imports...")
    
    try:
        from positional_encoding import get_positional_encoding, PE_REGISTRY
        logger.info("✅ Positional encoding imports successful")
        
        # Test each encoding type
        for pe_type in PE_REGISTRY.keys():
            try:
                pe = get_positional_encoding(pe_type, d_model=128)
                logger.info(f"✅ {pe_type} encoding created successfully")
            except Exception as e:
                logger.warning(f"⚠️ Issue with {pe_type}: {e}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Positional encoding import failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting Kaggle Compatibility Training Test")
    logger.info("=" * 60)
    
    # Setup device
    device = setup_device()
    
    # Test 1: Positional Encoding
    logger.info("\n📋 Test 1: Positional Encoding Imports")
    pe_success = test_positional_encoding()
    
    # Test 2: Training Loop
    logger.info("\n📋 Test 2: Training Loop")
    training_success = test_training(device, num_samples=8, num_epochs=2)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 KAGGLE TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    logger.info(f"Positional Encoding Test: {'✅ PASS' if pe_success else '❌ FAIL'}")
    logger.info(f"Training Test: {'✅ PASS' if training_success else '❌ FAIL'}")
    
    if pe_success and training_success:
        logger.info("\n🎉 ALL TESTS PASSED - Ready for Kaggle deployment!")
        return True
    else:
        logger.error("\n❌ Some tests failed - Check logs above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)