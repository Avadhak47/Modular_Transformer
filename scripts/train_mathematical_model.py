#!/usr/bin/env python3
"""
Mathematical Reasoning Model Training Script

Main training script for mathematical reasoning models with:
- Multi-node distributed training support
- Advanced positional encoding integration
- DeepSeekMath model fine-tuning
- Comprehensive evaluation and logging
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from accelerate import Accelerator
from peft import get_peft_model_state_dict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.mathematical_model import MathematicalReasoningModel, MathematicalModelFactory
from data.math_dataset_loader import MathematicalDatasetLoader, create_mathematical_dataloader
from src.utils.evaluation import MathematicalEvaluator
from src.utils.logging_utils import setup_logging, log_model_info

logger = logging.getLogger(__name__)


class MathematicalTrainer:
    """Enhanced trainer for mathematical reasoning models."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        node_id: int = 0,
        world_size: int = 1,
        local_rank: int = 0
    ):
        self.config = config
        self.node_id = node_id
        self.world_size = world_size
        self.local_rank = local_rank
        
        # Setup distributed training
        self.is_distributed = world_size > 1
        if self.is_distributed:
            self._setup_distributed()
        
        # Initialize accelerator
        self.accelerator = Accelerator()
        
        # Setup logging and directories
        self._setup_directories()
        self._setup_logging()
        
        # Initialize model and data
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.evaluator = None
        
        logger.info(f"Initialized trainer for Node {node_id}")
    
    def _setup_distributed(self):
        """Setup distributed training environment."""
        try:
            if 'RANK' in os.environ:
                rank = int(os.environ['RANK'])
                world_size = int(os.environ['WORLD_SIZE'])
                
                dist.init_process_group(
                    backend='nccl' if torch.cuda.is_available() else 'gloo',
                    rank=rank,
                    world_size=world_size
                )
                
                torch.cuda.set_device(self.local_rank)
                logger.info(f"Distributed training initialized: rank {rank}/{world_size}")
        except Exception as e:
            logger.warning(f"Failed to setup distributed training: {e}")
            self.is_distributed = False
    
    def _setup_directories(self):
        """Setup output directories."""
        self.output_dir = Path(self.config['experiment_config']['output_dir'])
        self.logging_dir = Path(self.config['experiment_config']['logging_dir'])
        
        for directory in [self.output_dir, self.logging_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Logging directory: {self.logging_dir}")
    
    def _setup_logging(self):
        """Setup comprehensive logging."""
        setup_logging(
            log_level=logging.INFO,
            log_file=self.logging_dir / f"training_node_{self.node_id}.log"
        )
        
        # Setup wandb if configured
        if self.config['training_config'].get('report_to') == 'wandb':
            try:
                wandb.init(
                    project=self.config['experiment_config']['wandb_project'],
                    name=self.config['experiment_config']['wandb_run_name'],
                    config=self.config,
                    dir=str(self.logging_dir)
                )
                logger.info("Weights & Biases logging initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
    
    def setup_model_and_data(self):
        """Initialize model, tokenizer, and datasets."""
        logger.info("Setting up model and data...")
        
        # Create model using factory
        self.model = MathematicalModelFactory.create_node_model(
            node_id=self.node_id,
            config=self.config
        )
        
        # Get tokenizer
        self.tokenizer = self.model.tokenizer
        
        # Log model information
        model_info = self.model.get_model_info()
        log_model_info(model_info, logger)
        
        # Setup datasets
        self._setup_datasets()
        
        # Setup evaluator
        self.evaluator = MathematicalEvaluator(
            tokenizer=self.tokenizer,
            device=self.accelerator.device
        )
        
        logger.info("Model and data setup complete")
    
    def _setup_datasets(self):
        """Setup training and evaluation datasets."""
        data_config = self.config['data_config']
        
        # Initialize dataset loader
        loader = MathematicalDatasetLoader(
            cache_dir=str(self.output_dir / "data_cache"),
            max_problems=data_config.get('max_train_samples')
        )
        
        # Load training datasets
        train_datasets = data_config.get('train_datasets', ['math', 'gsm8k'])
        train_problems = loader.load_combined_dataset(train_datasets, split='train')
        
        # Load evaluation datasets
        eval_datasets = data_config.get('eval_datasets', ['math_test', 'gsm8k_test'])
        eval_problems = []
        for dataset_name in eval_datasets:
            if dataset_name == 'math_test':
                eval_problems.extend(loader.load_math_dataset('test')[:500])
            elif dataset_name == 'gsm8k_test':
                eval_problems.extend(loader.load_gsm8k_dataset('test')[:200])
        
        # Create data loaders
        batch_size = self.config['training_config']['batch_size']
        max_length = self.config['model_config'].get('max_seq_len', 2048)
        
        self.train_dataset = create_mathematical_dataloader(
            problems=train_problems,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            include_solution=True,
            shuffle=True
        )
        
        self.eval_dataset = create_mathematical_dataloader(
            problems=eval_problems,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            include_solution=False,
            shuffle=False
        )
        
        logger.info(f"Training dataset: {len(train_problems)} problems")
        logger.info(f"Evaluation dataset: {len(eval_problems)} problems")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Prepare model and optimizer with accelerator
        self.model, self.train_dataset, self.eval_dataset = self.accelerator.prepare(
            self.model, self.train_dataset, self.eval_dataset
        )
        
        # Training configuration
        training_config = self.config['training_config']
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 0.01)
        )
        
        # Setup scheduler
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_config.get('warmup_steps', 500),
            num_training_steps=training_config['max_steps']
        )
        
        optimizer, scheduler = self.accelerator.prepare(optimizer, scheduler)
        
        # Training loop
        self.model.train()
        global_step = 0
        best_eval_loss = float('inf')
        
        progress_bar = tqdm(
            total=training_config['max_steps'],
            desc=f"Training Node {self.node_id}",
            disable=not self.accelerator.is_local_main_process
        )
        
        for epoch in range(100):  # Large number, will break on max_steps
            for batch in self.train_dataset:
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if training_config.get('max_grad_norm'):
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        training_config['max_grad_norm']
                    )
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Logging
                if global_step % training_config.get('logging_steps', 50) == 0:
                    self._log_training_metrics(loss, global_step, scheduler.get_last_lr()[0])
                
                # Evaluation
                if global_step % training_config.get('eval_steps', 500) == 0:
                    eval_results = self._evaluate()
                    
                    # Save best model
                    if eval_results['eval_loss'] < best_eval_loss:
                        best_eval_loss = eval_results['eval_loss']
                        self._save_model(global_step, is_best=True)
                
                # Save checkpoint
                if global_step % training_config.get('save_steps', 1000) == 0:
                    self._save_model(global_step)
                
                progress_bar.update(1)
                
                # Check if max steps reached
                if global_step >= training_config['max_steps']:
                    break
            
            if global_step >= training_config['max_steps']:
                break
        
        progress_bar.close()
        
        # Final evaluation
        final_results = self._evaluate()
        self._save_model(global_step, is_final=True)
        
        logger.info("Training completed!")
        return final_results
    
    def _evaluate(self):
        """Run evaluation on the model."""
        logger.info("Running evaluation...")
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_dataset:
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        # Mathematical reasoning evaluation
        math_metrics = self._evaluate_mathematical_reasoning()
        
        results = {
            'eval_loss': avg_loss,
            **math_metrics
        }
        
        # Log results
        if self.accelerator.is_local_main_process:
            logger.info(f"Evaluation results: {results}")
            if wandb.run:
                wandb.log(results)
        
        self.model.train()
        return results
    
    def _evaluate_mathematical_reasoning(self):
        """Evaluate mathematical reasoning capabilities."""
        # Sample a few problems for detailed evaluation
        sample_problems = [
            "What is 15 + 27?",
            "Solve for x: 2x + 5 = 13",
            "If a rectangle has length 8 and width 5, what is its area?"
        ]
        
        correct = 0
        total = len(sample_problems)
        
        for problem in sample_problems:
            try:
                result = self.model.generate_mathematical_solution(
                    problem,
                    max_length=256,
                    temperature=0.1
                )
                
                # Simple correctness check (would be more sophisticated in practice)
                if self._check_answer_correctness(problem, result['solution']):
                    correct += 1
                    
            except Exception as e:
                logger.warning(f"Failed to evaluate problem '{problem}': {e}")
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'math_accuracy': accuracy,
            'math_problems_evaluated': total
        }
    
    def _check_answer_correctness(self, problem: str, solution: str) -> bool:
        """Simple answer correctness check."""
        # This is a simplified check - in practice would use SymPy or other tools
        if "15 + 27" in problem:
            return "42" in solution
        elif "2x + 5 = 13" in problem:
            return "x = 4" in solution or "x=4" in solution
        elif "length 8 and width 5" in problem:
            return "40" in solution
        return False
    
    def _log_training_metrics(self, loss: torch.Tensor, step: int, learning_rate: float):
        """Log training metrics."""
        metrics = {
            'train_loss': loss.item(),
            'learning_rate': learning_rate,
            'step': step
        }
        
        if self.accelerator.is_local_main_process:
            logger.info(f"Step {step}: Loss={loss.item():.4f}, LR={learning_rate:.2e}")
            if wandb.run:
                wandb.log(metrics)
    
    def _save_model(self, step: int, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint."""
        if not self.accelerator.is_local_main_process:
            return
        
        # Create save directory
        if is_best:
            save_dir = self.output_dir / "best_model"
        elif is_final:
            save_dir = self.output_dir / "final_model"
        else:
            save_dir = self.output_dir / f"checkpoint-{step}"
        
        save_dir.mkdir(exist_ok=True)
        
        try:
            # Unwrap model if using DDP/accelerator
            model_to_save = self.accelerator.unwrap_model(self.model)
            
            # Save model
            model_to_save.save_model(str(save_dir))
            
            # Save training state
            training_state = {
                'step': step,
                'config': self.config,
                'model_info': model_to_save.get_model_info()
            }
            
            with open(save_dir / 'training_state.json', 'w') as f:
                json.dump(training_state, f, indent=2)
            
            logger.info(f"Model saved to {save_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def cleanup(self):
        """Cleanup resources."""
        if wandb.run:
            wandb.finish()
        
        if self.is_distributed:
            dist.destroy_process_group()
        
        logger.info("Training cleanup completed")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Mathematical Reasoning Model")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--node_id",
        type=int,
        default=0,
        help="Node ID for multi-node training"
    )
    
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank for distributed training"
    )
    
    parser.add_argument(
        "--resume_from",
        type=str,
        help="Path to checkpoint to resume from"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup environment variables for distributed training
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    try:
        # Initialize trainer
        trainer = MathematicalTrainer(
            config=config,
            node_id=args.node_id,
            world_size=world_size,
            local_rank=args.local_rank
        )
        
        # Setup model and data
        trainer.setup_model_and_data()
        
        # Start training
        results = trainer.train()
        
        # Log final results
        logger.info(f"Training completed successfully. Final results: {results}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        # Cleanup
        if 'trainer' in locals():
            trainer.cleanup()


if __name__ == "__main__":
    main()