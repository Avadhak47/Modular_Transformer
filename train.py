"""
Training script for the modular transformer.
Supports different positional encoding methods and comprehensive logging.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb
from tqdm import tqdm
import json
import math
from typing import Dict, Any, Optional, Tuple
import time

from src.model import TransformerModel
from src.utils.training_utils import (
    get_linear_schedule_with_warmup, 
    save_checkpoint, 
    load_checkpoint,
    AverageMeter,
    get_optimizer
)
from src.utils.metrics import calculate_perplexity, calculate_bleu
from config import get_config, ExperimentConfig


class DummyDataset(Dataset):
    """Dummy dataset for demonstration purposes."""
    
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random sequences
        src = torch.randint(1, self.vocab_size - 1, (self.seq_len,))
        tgt = torch.randint(1, self.vocab_size - 1, (self.seq_len,))
        
        # Add BOS and EOS tokens
        src = torch.cat([torch.tensor([1]), src, torch.tensor([2])])[:self.seq_len]
        tgt = torch.cat([torch.tensor([1]), tgt, torch.tensor([2])])[:self.seq_len]
        
        return {"src": src, "tgt": tgt}


class TransformerTrainer:
    """Main trainer class for the modular transformer."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = TransformerModel(config.model.__dict__).to(self.device)
        
        # Initialize optimizer
        self.optimizer = get_optimizer(self.model, config.training)
        
        # Initialize scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.training.warmup_steps,
            num_training_steps=config.training.max_steps
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Metrics tracking
        self.train_loss = AverageMeter()
        self.val_loss = AverageMeter()
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Initialize wandb if enabled
        if config.training.use_wandb:
            wandb.init(
                project=config.training.project_name,
                name=config.training.experiment_name,
                config=config.to_dict()
            )
        
        # Create checkpoint directory
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        
        print(f"Trainer initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Training on device: {self.device}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute a single training step."""
        self.model.train()
        
        src = batch['src'].to(self.device)
        tgt = batch['tgt'].to(self.device)
        
        # Create input and target sequences
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # Forward pass
        self.optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = self.model(src, tgt_input)
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
        
        # Update parameters
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Perform validation."""
        self.model.eval()
        val_loss = AverageMeter()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                logits = self.model(src, tgt_input)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
                
                val_loss.update(loss.item(), src.size(0))
        
        perplexity = math.exp(min(val_loss.avg, 100))  # Clip to prevent overflow
        
        return {
            'val_loss': val_loss.avg,
            'perplexity': perplexity
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.training.checkpoint_dir,
            f"{self.config.model.positional_encoding}_step_{self.step}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(
                self.config.training.checkpoint_dir,
                f"{self.config.model.positional_encoding}_best.pt"
            )
            torch.save(checkpoint, best_path)
            print(f"New best model saved at step {self.step}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded from step {self.step}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop."""
        print(f"Starting training with {self.config.model.positional_encoding} positional encoding")
        print(f"Total training steps: {self.config.training.max_steps}")
        
        self.model.train()
        start_time = time.time()
        
        while self.step < self.config.training.max_steps:
            epoch_loss = AverageMeter()
            
            for batch in tqdm(train_loader, desc=f"Epoch {self.epoch + 1}", leave=False):
                # Training step
                loss = self.train_step(batch)
                epoch_loss.update(loss)
                self.train_loss.update(loss)
                
                # Logging
                if self.step % self.config.training.log_interval == 0:
                    elapsed = time.time() - start_time
                    lr = self.scheduler.get_last_lr()[0]
                    
                    print(f"Step {self.step:6d} | Loss: {loss:.4f} | LR: {lr:.2e} | "
                          f"Time: {elapsed:.1f}s")
                    
                    if self.config.training.use_wandb:
                        wandb.log({
                            'train_loss': loss,
                            'learning_rate': lr,
                            'step': self.step,
                            'epoch': self.epoch
                        })
                
                # Validation
                if self.step % self.config.training.eval_interval == 0:
                    val_metrics = self.validate(val_loader)
                    
                    print(f"Validation | Loss: {val_metrics['val_loss']:.4f} | "
                          f"Perplexity: {val_metrics['perplexity']:.4f}")
                    
                    if self.config.training.use_wandb:
                        wandb.log({
                            'val_loss': val_metrics['val_loss'],
                            'perplexity': val_metrics['perplexity'],
                            'step': self.step
                        })
                    
                    # Early stopping check
                    if val_metrics['val_loss'] < self.best_val_loss - self.config.training.min_delta:
                        self.best_val_loss = val_metrics['val_loss']
                        self.patience_counter = 0
                        is_best = True
                    else:
                        self.patience_counter += 1
                        is_best = False
                    
                    if self.patience_counter >= self.config.training.patience:
                        print(f"Early stopping triggered after {self.patience_counter} evaluations without improvement")
                        return
                    
                    # Save checkpoint
                    self.save_checkpoint(is_best)
                
                # Regular checkpoint saving
                if self.step % self.config.training.save_interval == 0:
                    self.save_checkpoint()
                
                self.step += 1
                
                if self.step >= self.config.training.max_steps:
                    break
            
            self.epoch += 1
            print(f"Epoch {self.epoch} completed | Average Loss: {epoch_loss.avg:.4f}")
        
        print(f"Training completed after {self.step} steps")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def create_data_loaders(config: ExperimentConfig) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    # For demonstration, we use dummy datasets
    # In practice, you would load your actual data here
    
    train_dataset = DummyDataset(
        vocab_size=config.model.vocab_size,
        seq_len=config.model.max_seq_len,
        num_samples=10000
    )
    
    val_dataset = DummyDataset(
        vocab_size=config.model.vocab_size,
        seq_len=config.model.max_seq_len,
        num_samples=1000
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Modular Transformer')
    parser.add_argument('--pe_type', type=str, default='sinusoidal',
                       choices=['sinusoidal', 'rope', 'alibi', 'diet', 't5_relative', 'nope'],
                       help='Positional encoding type')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name for logging')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = ExperimentConfig.load(args.config)
    else:
        config = get_config(args.pe_type)
    
    # Override with command line arguments
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.max_steps = args.epochs * 1000  # Approximate steps per epoch
    config.training.use_wandb = args.wandb
    
    if args.experiment_name:
        config.training.experiment_name = args.experiment_name
    else:
        config.training.experiment_name = f"transformer_{args.pe_type}"
    
    # Create trainer
    trainer = TransformerTrainer(config)
    
    # Load checkpoint if resuming
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)
    
    # Start training
    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint()
    except Exception as e:
        print(f"Training failed with error: {e}")
        trainer.save_checkpoint()
        raise


if __name__ == "__main__":
    main()