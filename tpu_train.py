import os
import glob
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import sys

# Try to import PyTorch XLA with proper error handling
TPU_AVAILABLE = False
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.debug.metrics as met
    import torch_xla.utils.utils as xu
    TPU_AVAILABLE = True
    print("PyTorch XLA imported successfully")
except ImportError as e:
    print(f"PyTorch XLA import failed: {e}")
    print("This is likely due to PyTorch version incompatibility.")
    print("Falling back to CPU/GPU training mode.")
    TPU_AVAILABLE = False
except Exception as e:
    print(f"Unexpected error importing PyTorch XLA: {e}")
    TPU_AVAILABLE = False

from config import get_config, ExperimentConfig
from src.model import TransformerModel
from src.utils.training_utils import (
    get_linear_schedule_with_warmup,
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
    get_optimizer
)
from data.math_dataset_loader import MathematicalDatasetLoader, MathematicalProblem
from torch.utils.data import Dataset
import wandb
import json

def find_latest_checkpoint(checkpoint_dir, pe_type=None):
    pattern = os.path.join(checkpoint_dir, f'kaggle_epoch_*_{pe_type if pe_type else "*"}_*.pt') if pe_type else os.path.join(checkpoint_dir, 'kaggle_epoch_*.pt')
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None, 0
    def extract_epoch(path):
        fname = os.path.basename(path)
        try:
            return int(fname.split('_')[2])
        except Exception:
            return -1
    latest_ckpt = max(checkpoints, key=extract_epoch)
    latest_epoch = extract_epoch(latest_ckpt)
    return latest_ckpt, latest_epoch

def remove_old_checkpoints(checkpoint_dir, pe_type, keep_filename):
    pattern = os.path.join(checkpoint_dir, f'kaggle_epoch_*_{pe_type}_*.pt')
    for f in glob.glob(pattern):
        if os.path.basename(f) != keep_filename:
            try:
                os.remove(f)
                print(f"Deleted old checkpoint: {f}")
            except Exception as e:
                print(f"Failed to delete {f}: {e}")

def get_next_experiment_name(base_dir, prefix='train_'):
    import re
    import os
    existing = []
    if os.path.exists(base_dir):
        for name in os.listdir(base_dir):
            m = re.match(rf'{prefix}(\d+)', name)
            if m:
                existing.append(int(m.group(1)))
    next_num = max(existing) + 1 if existing else 1
    return f'{prefix}{next_num:02d}'

def upload_checkpoints_to_kaggle(checkpoint_dir, version_notes="Training completed"):
    """Upload all checkpoints to Kaggle dataset using dataset_create_version with zip mode."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        print(f"[KAGGLE_API] Creating dataset version with zip mode...")
        print(f"[KAGGLE_API] Notes: {version_notes}")
        
        # Create version with zip mode
        api.dataset_create_version(
            folder=checkpoint_dir,
            version_notes=version_notes,
            dir_mode="zip",  # This is the key improvement
            quiet=False
        )
        
        print(f"[KAGGLE_API] Successfully uploaded dataset version!")
        return True
    except Exception as e:
        print(f"[KAGGLE_API] Failed to upload dataset: {e}")
        return False

def setup_device(use_tpu=True):
    """Setup device with fallback options."""
    if use_tpu and TPU_AVAILABLE:
        try:
            device = xm.xla_device()
            print(f"Successfully initialized TPU device: {device}")
            return device, True
        except Exception as e:
            print(f"TPU initialization failed: {e}")
            print("Falling back to CPU/GPU")
            return setup_device(use_tpu=False)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print("Using CPU")
        return device, False

def main():
    parser = argparse.ArgumentParser(description='TPU/GPU Robust Training Script')
    parser.add_argument('--pe_type', type=str, default='sinusoidal',
                       choices=['sinusoidal', 'rope', 'alibi', 'diet', 't5_relative', 'nope'],
                       help='Positional encoding type')
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'medium', 'large'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint if available')
    parser.add_argument('--tokenizer_name', type=str, default='gpt2', help='Tokenizer name')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--experiment_name', type=str, default=None, help='Custom experiment name')
    parser.add_argument('--experiment_suffix', type=str, default='', help='Experiment name suffix')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU training even if TPU/GPU available')
    args = parser.parse_args()

    config = get_config(args.pe_type, model_size=args.model_size)
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.max_steps = args.epochs * 1000  # Approximate steps per epoch
    config.training.use_wandb = args.wandb
    config.training.experiment_suffix = args.experiment_suffix
    config.training.tokenizer_name = args.tokenizer_name

    # Set experiment_name
    if args.experiment_name:
        config.training.experiment_name = args.experiment_name
    else:
        config.training.experiment_name = get_next_experiment_name('../checkpoints', prefix='train_')

    checkpoint_dir = '../checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load GSM8k + MATH problems and tokenizer
    loader = MathematicalDatasetLoader(tokenizer_name=args.tokenizer_name, max_length=config.model.max_seq_len)
    train_problems, _ = loader.load_combined_train_test(test_size=0.1)

    class MathReasoningDataset(Dataset):
        def __init__(self, problems, tokenizer, max_length=1024):
            self.problems = problems
            self.tokenizer = tokenizer
            self.max_length = max_length
        def __len__(self):
            return len(self.problems)
        def __getitem__(self, idx):
            problem = self.problems[idx]
            input_text = f"Solve this step by step:\n\nProblem: {problem.problem}\n\nSolution:"
            target_text = problem.solution
            input_encoding = self.tokenizer(
                input_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt'
            )
            target_encoding = self.tokenizer(
                target_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt'
            )
            return {
                'src': input_encoding['input_ids'].squeeze(),
                'tgt': target_encoding['input_ids'].squeeze()
            }

    train_dataset = MathReasoningDataset(train_problems, loader.tokenizer, max_length=config.model.max_seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Device setup with fallback
    device, is_tpu = setup_device(use_tpu=not args.force_cpu)
    
    model = TransformerModel(config.model.__dict__)
    model = model.to(device)
    print(f"Model moved to {device} with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    optimizer = get_optimizer(model, config.training)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=config.training.max_steps
    )

    # Initialize wandb if enabled
    if args.wandb:
        device_type = 'TPU' if is_tpu else 'GPU' if torch.cuda.is_available() else 'CPU'
        wandb.init(
            project=getattr(config.training, 'project_name', 'mathematical_reasoning_transformers'),
            name=f"{config.training.experiment_name}{args.experiment_suffix}",
            config=config.training.__dict__,
            tags=['mathematical_reasoning', args.pe_type, device_type],
            mode='disabled' if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') else 'online'
        )

    start_epoch = 0
    if args.resume:
        latest_ckpt, latest_epoch = find_latest_checkpoint(checkpoint_dir, args.pe_type)
        if latest_ckpt:
            print(f"Resuming from checkpoint: {latest_ckpt} (epoch {latest_epoch})")
            step = load_checkpoint(latest_ckpt, model, optimizer, scheduler)
            start_epoch = latest_epoch + 1
        else:
            print("No checkpoint found, starting from scratch.")
    else:
        print("Starting training from scratch.")

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = AverageMeter()
        
        # Device-specific data loading
        if is_tpu:
            train_device_loader = pl.MpDeviceLoader(train_loader, device)
        else:
            train_device_loader = train_loader
        
        for batch in tqdm(train_device_loader, desc=f"Epoch {epoch+1}"):
            src = batch['src']
            tgt = batch['tgt']
            if tgt.size(0) == 0 or tgt.size(-1) < 2:
                continue
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            optimizer.zero_grad()
            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            
            # Device-specific optimizer step
            if is_tpu:
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()
            scheduler.step()
            epoch_loss.update(loss.item())
            
            if args.wandb:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': scheduler.get_last_lr()[0],
                    'epoch': epoch
                })
        
        print(f"Epoch {epoch+1} completed | Average Loss: {epoch_loss.avg:.4f}")
        if args.wandb:
            wandb.log({'epoch_avg_loss': epoch_loss.avg, 'epoch': epoch})
        
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        device_suffix = 'tpu' if is_tpu else 'gpu' if torch.cuda.is_available() else 'cpu'
        ckpt_filename = f"{device_suffix}_epoch_{epoch}_{args.pe_type}_{timestamp}{args.experiment_suffix}.pt"
        save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir, ckpt_filename)
        print(f"Checkpoint saved: {ckpt_filename}")
        remove_old_checkpoints(checkpoint_dir, args.pe_type, ckpt_filename)
        
        # Device-specific metrics
        if is_tpu:
            xm.mark_step()
            print(f"TPU Metrics: {met.metrics_report()}")

    print("Training completed.")
    
    # Upload all checkpoints to Kaggle at the end
    if os.environ.get('KAGGLE_API') == '1':
        print("Training completed. Uploading all checkpoints to Kaggle...")
        device_type = 'TPU' if is_tpu else 'GPU' if torch.cuda.is_available() else 'CPU'
        version_notes = f"{device_type} Training completed - {args.pe_type} model, {args.epochs} epochs, experiment: {config.training.experiment_name}{args.experiment_suffix}"
        upload_checkpoints_to_kaggle(checkpoint_dir, version_notes)
    else:
        print("To upload checkpoints to Kaggle, set KAGGLE_API=1")
    
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main() 