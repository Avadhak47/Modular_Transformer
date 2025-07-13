"""
Training utility functions.
"""
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import os
from typing import Dict, Any


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Create a schedule with a learning rate that decreases linearly after warmup."""
    
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda)


def get_optimizer(model: torch.nn.Module, config) -> torch.optim.Optimizer:
    """Create optimizer based on configuration."""
    if config.optimizer == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def save_checkpoint(model, optimizer, scheduler, step: int, checkpoint_dir: str, filename: str):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, model, optimizer=None, scheduler=None):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    step = checkpoint.get('step', 0)
    print(f"Checkpoint loaded from {filepath}, step: {step}")
    
    return step
