"""
Complete training pipeline for mathematical reasoning with all specified metrics.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import re
import json
from tqdm import tqdm
import wandb
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
import math

from data.math_dataset_loader import MathematicalDatasetLoader, MathematicalProblem
from evaluation.mathematical_metrics import MathematicalReasoningEvaluator
from src.model import TransformerModel

class MathematicalReasoningDataset(Dataset):
    """PyTorch Dataset for mathematical reasoning problems."""
    
    def __init__(self, problems: List[MathematicalProblem], tokenizer, max_length: int = 1024):
        self.problems = problems
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.problems)
    
    def __getitem__(self, idx):
        problem = self.problems[idx]
        
        # Create chain-of-thought input
        input_text = f"Solve this step by step:\n\nProblem: {problem.problem}\n\nSolution:"
        target_text = problem.solution
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length // 2,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze(),
            'problem_obj': problem
        }

class MathematicalReasoningTrainer:
    """Complete trainer implementing all evaluation metrics from the proposal."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.logger.info(f"Initializing model with {config['positional_encoding']} positional encoding")
        self.model = TransformerModel(config['model']).to(self.device)
        
        # Initialize data loader
        self.data_loader = MathematicalDatasetLoader(
            tokenizer_name=config['tokenizer_name'],
            max_length=config['max_length']
        )
        
        # Initialize evaluator with all metrics
        self.evaluator = MathematicalReasoningEvaluator(config['tokenizer_name'])
        
        # Initialize optimizer with proper configuration
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.98),
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get('scheduler_t0', 1000),
            T_mult=2
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.data_loader.tokenizer.pad_token_id)
        
        # Initialize Weights & Biases
        if config.get('use_wandb', False):
            wandb.init(
                project=config['project_name'],
                name=f"math_reasoning_{config['positional_encoding']}_{config.get('experiment_suffix', '')}",
                config=config,
                tags=['mathematical_reasoning', config['positional_encoding']]
            )
    
    def load_datasets(self):
        """Load and prepare MATH and GSM8K datasets as specified in proposal."""
        self.logger.info("Loading MATH and GSM8K datasets...")
        
        # Load GSM8K dataset (8.5K problems)
        self.logger.info("Loading GSM8K dataset...")
        gsm8k_train = self.data_loader.load_gsm8k_dataset("train")
        gsm8k_test = self.data_loader.load_gsm8k_dataset("test")
        
        # Load MATH dataset (12.5K problems - limit for training efficiency)
        self.logger.info("Loading MATH dataset...")
        math_train = self.data_loader.load_math_dataset("train", max_samples=8000)
        
        # Combine datasets as specified in proposal
        self.train_problems = gsm8k_train + math_train
        self.test_problems = gsm8k_test  # Use GSM8K test for evaluation
        
        self.logger.info(f"Training set: {len(self.train_problems)} problems")
        self.logger.info(f"  - GSM8K: {len(gsm8k_train)} problems")
        self.logger.info(f"  - MATH: {len(math_train)} problems")
        self.logger.info(f"Test set: {len(self.test_problems)} problems")
        
        # Create PyTorch datasets
        self.train_dataset = MathematicalReasoningDataset(
            self.train_problems,
            self.data_loader.tokenizer,
            self.config['max_length']
        )
        
        self.test_dataset = MathematicalReasoningDataset(
            self.test_problems,
            self.data_loader.tokenizer,
            self.config['max_length']
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 2),
            pin_memory=True
        )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with mathematical reasoning optimization."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Prepare decoder input (shifted labels for autoregressive training)
            decoder_input = labels[:, :-1]
            target_labels = labels[:, 1:]
            
            # Forward pass
            outputs = self.model(input_ids, decoder_input)
            
            # Calculate loss
            loss = self.criterion(
                outputs.reshape(-1, outputs.size(-1)),
                target_labels.reshape(-1)
            )
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation and optimization step
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Accumulate metrics
            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # Log to wandb
            if self.config.get('use_wandb', False) and batch_idx % 50 == 0:
                wandb.log({
                    'train_loss_step': loss.item() * gradient_accumulation_steps,
                    'learning_rate': current_lr,
                    'epoch': epoch,
                    'step': epoch * len(self.train_loader) + batch_idx
                })
        
        avg_loss = total_loss / num_batches
        return {
            'train_loss': avg_loss,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def generate_response(self, problem_text: str, max_length: int = 512) -> str:
        """Generate response for a mathematical problem."""
        self.model.eval()
        
        # Prepare input
        input_text = f"Solve this step by step:\n\nProblem: {problem_text}\n\nSolution:"
        inputs = self.data_loader.tokenizer(
            input_text,
            return_tensors='pt',
            truncation=True,
            max_length=256
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.data_loader.tokenizer.pad_token_id,
                eos_token_id=self.data_loader.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_attentions=True
            )
        
        # Decode response
        generated_text = self.data_loader.tokenizer.decode(
            outputs.sequences[0],
            skip_special_tokens=True
        )
        
        # Extract solution part
        if "Solution:" in generated_text:
            solution = generated_text.split("Solution:")[-1].strip()
        else:
            solution = generated_text
        
        return solution, outputs.attentions if hasattr(outputs, 'attentions') else None
    
    def comprehensive_evaluation(self, epoch: int) -> Dict[str, Any]:
        """Comprehensive evaluation implementing all metrics from the proposal."""
        self.logger.info(f"Starting comprehensive evaluation for epoch {epoch + 1}")
        self.model.eval()
        
        predictions = []
        ground_truths = []
        reasoning_chains = []
        problems = []
        solutions = []
        all_attention_weights = []
        
        # Generate predictions for test set
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Generating predictions")):
                problem_objects = batch['problem_obj']
                
                for problem_obj in problem_objects:
                    # Generate model response
                    generated_solution, attention_weights = self.generate_response(problem_obj.problem)
                    
                    # Extract final answer from generated solution
                    predicted_answer = self._extract_final_answer(generated_solution)
                    
                    # Extract reasoning steps from generated solution
                    generated_steps = self._extract_reasoning_steps(generated_solution)
                    
                    # Collect data for evaluation
                    predictions.append(predicted_answer)
                    ground_truths.append(problem_obj.final_answer)
                    reasoning_chains.append(generated_steps)
                    problems.append(problem_obj.problem)
                    solutions.append(problem_obj.solution)
                    
                    # Store attention weights (limit to avoid memory issues)
                    if attention_weights and len(all_attention_weights) < 100:
                        # Take the last layer's attention from the last generation step
                        last_attention = attention_weights[-1][-1]  
                        all_attention_weights.append(last_attention)
                
                # Limit evaluation size for efficiency
                if len(predictions) >= 500:  
                    break
        
        # Combine attention weights into tensor
        attention_tensor = None
        if all_attention_weights:
            try:
                attention_tensor = torch.stack(all_attention_weights)
            except:
                self.logger.warning("Could not stack attention weights")
        
        # Run comprehensive evaluation with all metrics
        self.logger.info("Calculating all evaluation metrics...")
        evaluation_results = self.evaluator.comprehensive_evaluation(
            model=self.model,
            predictions=predictions,
            ground_truths=ground_truths,
            reasoning_chains=reasoning_chains,
            problems=problems,
            solutions=solutions,
            attention_weights=attention_tensor,
            device=self.device
        )
        
        # Add epoch information
        evaluation_results['epoch'] = epoch
        evaluation_results['num_evaluated'] = len(predictions)
        
        # Log results
        self.logger.info("Evaluation Results:")
        self.logger.info(f"  Exact Match Accuracy: {evaluation_results['summary']['overall_accuracy']:.4f}")
        self.logger.info(f"  Numerical Accuracy: {evaluation_results['summary']['numerical_accuracy']:.4f}")
        self.logger.info(f"  Step Correctness: {evaluation_results['summary']['step_correctness']:.4f}")
        self.logger.info(f"  Logical Validity: {evaluation_results['summary']['logical_validity']:.4f}")
        self.logger.info(f"  Perplexity: {evaluation_results['summary']['perplexity']:.4f}")
        if 'mean_attention_entropy' in evaluation_results['summary']:
            self.logger.info(f"  Attention Entropy: {evaluation_results['summary']['mean_attention_entropy']:.4f}")
        
        # Log to wandb
        if self.config.get('use_wandb', False):
            wandb.log({
                'eval/exact_match_accuracy': evaluation_results['summary']['overall_accuracy'],
                'eval/numerical_accuracy': evaluation_results['summary']['numerical_accuracy'],
                'eval/step_correctness': evaluation_results['summary']['step_correctness'],
                'eval/logical_validity': evaluation_results['summary']['logical_validity'],
                'eval/perplexity': evaluation_results['summary']['perplexity'],
                'eval/attention_entropy': evaluation_results['summary']['mean_attention_entropy'],
                'eval/normalized_attention_entropy': evaluation_results['summary']['normalized_attention_entropy'],
                'epoch': epoch
            })
        
        return evaluation_results
    
    def _extract_final_answer(self, solution: str) -> str:
        """Extract final answer from generated solution."""
        # Look for various answer patterns
        patterns = [
            r'####\s*([0-9,]+(?:\.[0-9]+)?)',  # GSM8K style
            r'\\boxed\{([^}]+)\}',             # MATH style
            r'(?:answer|result|solution)(?:\s*is)?:?\s*([0-9,]+(?:\.[0-9]+)?)',  # Natural language
            r'\b([0-9,]+(?:\.[0-9]+)?)\s*(?:\.|$)'  # Last number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, solution, re.IGNORECASE)
            if match:
                return match.group(1).replace(',', '')
        
        return ""
    
    def _extract_reasoning_steps(self, solution: str) -> List[str]:
        """Extract reasoning steps from generated solution."""
        # Split by sentences and filter for meaningful steps
        sentences = re.split(r'[.!?]+', solution)
        steps = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 10 and 
                any(char.isalpha() for char in sentence) and
                not sentence.startswith('####')):
                steps.append(sentence)
        
        return steps[:10]  # Limit steps for efficiency
    
    def train(self):
        """Complete training loop with comprehensive evaluation."""
        self.logger.info("Starting mathematical reasoning training...")
        
        # Load datasets
        self.load_datasets()
        
        # Training tracking
        best_accuracy = 0.0
        evaluation_history = []
        
        for epoch in range(self.config['num_epochs']):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Training phase
            train_results = self.train_epoch(epoch)
            self.logger.info(f"Training Loss: {train_results['train_loss']:.4f}")
            
            # Evaluation phase
            if (epoch + 1) % self.config['eval_interval'] == 0:
                eval_results = self.comprehensive_evaluation(epoch)
                
                # Track best model
                current_accuracy = eval_results['summary']['overall_accuracy']
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    self.save_model(f"best_model_{self.config['positional_encoding']}.pt")
                    self.logger.info(f"New best model saved with accuracy: {best_accuracy:.4f}")
                
                evaluation_history.append(eval_results)
        
        # Save final results
        self.save_evaluation_history(evaluation_history)
        
        self.logger.info(f"\nTraining completed!")
        self.logger.info(f"Best accuracy achieved: {best_accuracy:.4f}")
        
        return evaluation_history
    
    def save_model(self, filename: str):
        """Save model checkpoint."""
        save_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        save_dir.mkdir(exist_ok=True)
        
        torch.save({
            'epoch': self.config['num_epochs'],
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'positional_encoding': self.config['positional_encoding']
        }, save_dir / filename)
        
        self.logger.info(f"Model saved to {save_dir / filename}")
    
    def save_evaluation_history(self, history: List[Dict[str, Any]]):
        """Save evaluation results history."""
        results_dir = Path(self.config.get('results_dir', 'results'))
        results_dir.mkdir(exist_ok=True)
        
        filename = f"evaluation_history_{self.config['positional_encoding']}.json"
        with open(results_dir / filename, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        self.logger.info(f"Evaluation history saved to {results_dir / filename}")

# Configuration templates for different positional encodings
def get_mathematical_reasoning_config(pe_type: str = "sinusoidal") -> Dict[str, Any]:
    """Get configuration for mathematical reasoning training."""
    base_config = {
        'model': {
            'd_model': 512,
            'n_heads': 8,
            'd_ff': 2048,
            'n_encoder_layers': 6,
            'n_decoder_layers': 6,
            'vocab_size': 50257,  # GPT-2 vocab size
            'max_seq_len': 1024,
            'dropout': 0.1,
            'positional_encoding': pe_type
        },
        'tokenizer_name': 'gpt2',
        'max_length': 1024,
        'batch_size': 4,           # Small batch size for mathematical reasoning
        'eval_batch_size': 8,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'num_epochs': 10,
        'eval_interval': 2,
        'gradient_accumulation_steps': 4,
        'scheduler_t0': 1000,
        'num_workers': 2,
        'use_wandb': True,
        'project_name': 'mathematical_reasoning_transformers',
        'positional_encoding': pe_type,
        'checkpoint_dir': 'checkpoints',
        'results_dir': 'results'
    }
    
    return base_config

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train mathematical reasoning transformer')
    parser.add_argument('--pe_type', default='sinusoidal', 
                       choices=['sinusoidal', 'rope', 'alibi', 'diet', 't5_relative', 'nope'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--experiment_suffix', default='')
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_mathematical_reasoning_config(args.pe_type)
    config['num_epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['experiment_suffix'] = args.experiment_suffix
    
    # Initialize and run trainer
    trainer = MathematicalReasoningTrainer(config)
    trainer.train()
