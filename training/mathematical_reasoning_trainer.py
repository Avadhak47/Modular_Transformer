"""
Enhanced training pipeline for mathematical reasoning with improved dataset loading and Kaggle optimizations.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
import warnings

# Import project modules
from data.math_dataset_loader import MathematicalDatasetLoader, MathematicalProblem
from evaluation.mathematical_metrics import MathematicalReasoningEvaluator
from src.model import TransformerModel

print("all module imported succesfully")

class MathematicalReasoningDataset(Dataset):
    """Enhanced PyTorch Dataset for mathematical reasoning problems with better error handling."""
    
    def __init__(self, problems: List[MathematicalProblem], tokenizer, max_length: int = 1024):
        self.problems = problems
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Validate problems
        if not problems:
            raise ValueError("No problems provided to dataset")
        
        print(f"Dataset initialized with {len(problems)} problems")
    
    def __len__(self):
        return len(self.problems)
    
    def __getitem__(self, idx):
        try:
            problem = self.problems[idx]
            
            # Create chain-of-thought input
            input_text = f"Solve this step by step:\n\nProblem: {problem.problem}\n\nSolution:"
            target_text = problem.solution
            
            # Tokenize input with error handling
            try:
                input_encoding = self.tokenizer(
                    input_text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,   # FIX: use self.max_length
                    return_tensors='pt'
                )
                
                target_encoding = self.tokenizer(
                    target_text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
            except Exception as e:
                print(f"Tokenization error for problem {idx}: {e}")
                # Return dummy data to prevent crashes
                dummy_tensor = torch.zeros(self.max_length, dtype=torch.long)
                return {
                    'input_ids': dummy_tensor,
                    'attention_mask': torch.ones(self.max_length, dtype=torch.long),
                    'labels': dummy_tensor
                }
            
            return {
                'input_ids': input_encoding['input_ids'].squeeze(),
                'attention_mask': input_encoding['attention_mask'].squeeze(),
                'labels': target_encoding['input_ids'].squeeze()
            }
        except Exception as e:
            print(f"Error processing problem {idx}: {e}")
            # Return dummy data
            dummy_tensor = torch.zeros(self.max_length, dtype=torch.long)
            return {
                'input_ids': dummy_tensor,
                'attention_mask': torch.ones(self.max_length, dtype=torch.long),
                'labels': dummy_tensor
            }

class MathematicalReasoningTrainer:
    """Enhanced trainer with improved dataset loading and Kaggle optimizations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize model with error handling
        try:
            self.logger.info(f"Initializing model with {config['positional_encoding']} positional encoding")
            self.model = TransformerModel(config['model'])
            
            # Multi-GPU support
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs via DataParallel!")
                self.model = nn.DataParallel(self.model)
            
            self.model = self.model.to(self.device)
            print(f"Model initialized successfully with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            raise
        
        # Initialize data loader with enhanced error handling
        try:
            # Use SOTA tokenizer by default
            self.data_loader = MathematicalDatasetLoader(
                tokenizer_name=config.get('tokenizer_name', 'mistralai/Mistral-7B-v0.1'),
                max_length=config['max_length']
            )
            print("Data loader initialized successfully")
        except Exception as e:
            self.logger.error(f"Data loader initialization failed: {e}")
            raise
        
        # Initialize evaluator
        try:
            self.evaluator = MathematicalReasoningEvaluator(config.get('tokenizer_name', 'mistralai/Mistral-7B-v0.1'))
            print("Evaluator initialized successfully")
        except Exception as e:
            self.logger.error(f"Evaluator initialization failed: {e}")
            raise
        
        # Initialize optimizer with improved configuration
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
        
        # Initialize Weights & Biases with Kaggle-specific settings
        if config.get('use_wandb', False):
            try:
                wandb.init(
                    project=config['project_name'],
                    name=f"math_reasoning_{config['positional_encoding']}_{config.get('experiment_suffix', '')}",
                    config=config,
                    tags=['mathematical_reasoning', config['positional_encoding']],
                    mode='disabled' if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') else 'online'
                )
                print("WandB initialized successfully")
            except Exception as e:
                print(f"WandB initialization failed: {e}")
                config['use_wandb'] = False
    
    def load_datasets(self):
        """Load datasets from local files and perform train/test split at runtime."""
        self.logger.info("Loading datasets from local files...")
        print("=== Loading Datasets from Local Files ===")
        try:
            train_problems, test_problems = self.data_loader.load_combined_train_test(test_size=0.1)
            print(f"✓ Local train loaded: {len(train_problems)} problems")
            print(f"✓ Local test loaded: {len(test_problems)} problems")
        except Exception as e:
            print(f"✗ Local dataset loading failed: {e}")
            train_problems, test_problems = [], []
        self.train_problems = train_problems
        self.test_problems = test_problems
        print(f"Total training problems: {len(self.train_problems)}")
        print(f"Total test problems: {len(self.test_problems)}")
        if len(self.train_problems) == 0:
            print("ERROR: No training problems loaded! Exiting.")
            self.train_loader = None
            self.test_loader = None
            return
        try:
            print("Creating PyTorch datasets...")
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
            print(f"✓ Train dataset size: {len(self.train_dataset)}")
            print(f"✓ Test dataset size: {len(self.test_dataset)}")
        except Exception as e:
            print(f"✗ Dataset creation failed: {e}")
            self.train_loader = None
            self.test_loader = None
            return
        try:
            print("Creating data loaders...")
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=min(self.config.get('num_workers', 2), 4),
                pin_memory=True if torch.cuda.is_available() else False
            )
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.config['eval_batch_size'],
                shuffle=False,
                num_workers=min(self.config.get('num_workers', 2), 4),
                pin_memory=True if torch.cuda.is_available() else False
            )
            print(f"✓ Train loader batches: {len(self.train_loader) if self.train_loader else 0}")
            print(f"✓ Test loader batches: {len(self.test_loader) if self.test_loader else 0}")
        except Exception as e:
            print(f"✗ Data loader creation failed: {e}")
            self.train_loader = None
            self.test_loader = None
            return
        print("=== Dataset Loading Complete ===")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Enhanced training epoch with minimal terminal output."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                decoder_input = labels
                target_labels = labels
                self.optimizer.zero_grad()
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_ids, decoder_input)
                        loss = self.criterion(
                            outputs.reshape(-1, outputs.size(-1)),
                            target_labels.reshape(-1)
                        )
                        loss = loss / gradient_accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    outputs = self.model(input_ids, decoder_input)
                    loss = self.criterion(
                        outputs.reshape(-1, outputs.size(-1)),
                        target_labels.reshape(-1)
                    )
                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                total_loss += loss.item() * gradient_accumulation_steps
                num_batches += 1
                if (batch_idx + 1) % 100 == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({'loss': f"{loss.item() * gradient_accumulation_steps:.4f}", 'lr': f"{current_lr:.2e}"})
                if self.config.get('use_wandb', False) and batch_idx % 100 == 0:
                    wandb.log({
                        'train_loss_step': loss.item() * gradient_accumulation_steps,
                        'learning_rate': self.scheduler.get_last_lr()[0],
                        'epoch': epoch,
                        'step': epoch * (len(self.train_loader) if self.train_loader else 0) + batch_idx
                    })
            except Exception as e:
                # Only print error summary
                print(f"Error in training batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                print("Stopping training due to error.")
                break
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
        return {
            'train_loss': avg_loss,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def generate_response(self, problem_text: str, max_length: int = 512) -> tuple[str, None]:
        """Generate response for a mathematical problem using greedy decoding."""
        self.model.eval()
        
        try:
            # Prepare input
            input_text = f"Solve this step by step:\n\nProblem: {problem_text}\n\nSolution:"
            inputs = self.data_loader.tokenizer(
                input_text,
                return_tensors='pt',
                truncation=True,
                max_length=256
            )
            input_ids = inputs['input_ids'].to(self.device)
            
            # Start with BOS or first token of target
            generated = input_ids
            eos_token_id = self.data_loader.tokenizer.eos_token_id
            pad_token_id = self.data_loader.tokenizer.pad_token_id
            
            for _ in range(max_length):
                # For decoder input, use everything except last token as input, last token as target
                decoder_input = generated[:, -self.config['model']['max_seq_len']:]  # truncate to max_seq_len
                logits = self.model(input_ids, decoder_input)
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token_id], dim=1)
                if next_token_id.item() == eos_token_id or next_token_id.item() == pad_token_id:
                    break
            
            # Remove the input prompt from the generated sequence
            gen_tokens = generated[0, input_ids.shape[1]:]
            generated_text = self.data_loader.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            
            # Extract solution part
            if "Solution:" in generated_text:
                parts = generated_text.split("Solution:")
                solution = parts[-1].strip() if parts and len(parts) > 1 else generated_text.strip()
            else:
                solution = generated_text
                
            return solution, None
            
        except Exception as e:
            print(f"Generation error: {e}")
            return "", None
    
    def comprehensive_evaluation(self, epoch: int) -> Dict[str, Any]:
        """Enhanced comprehensive evaluation with better error handling."""
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
            if hasattr(self, 'test_problems') and self.test_problems:
                for idx, problem_obj in enumerate(tqdm(self.test_problems[:500], desc="Generating predictions")):
                    try:
                        # Generate model response
                        generated_solution, attention_weights = self.generate_response(problem_obj.problem)
                        
                        # Extract final answer from generated solution
                        predicted_answer = self._extract_final_answer(generated_solution) if generated_solution else ""
                        
                        # Extract reasoning steps from generated solution
                        generated_steps = self._extract_reasoning_steps(generated_solution) if generated_solution else []
                        
                        # Collect data for evaluation
                        predictions.append(predicted_answer)
                        ground_truths.append(problem_obj.final_answer)
                        reasoning_chains.append(generated_steps)
                        problems.append(problem_obj.problem)
                        solutions.append(problem_obj.solution)
                        
                        # Store attention weights (limit to avoid memory issues)
                        if attention_weights and len(all_attention_weights) < 100:
                            # Take the last layer's attention from the last generation step, guard against empty
                            if isinstance(attention_weights, list) and attention_weights and isinstance(attention_weights[-1], list) and attention_weights[-1]:
                            last_attention = attention_weights[-1][-1]  
                            all_attention_weights.append(last_attention)
                        
                        # Limit evaluation size for efficiency
                        if len(predictions) >= 500:  
                            break
                            
                    except Exception as e:
                        print(f"Error processing problem {idx}: {e}")
                        continue
        
        # Combine attention weights into tensor
        attention_tensor = None
        if all_attention_weights:
            try:
                attention_tensor = torch.stack(all_attention_weights)
            except:
                self.logger.warning("Could not stack attention weights")
        
        # Run comprehensive evaluation with all metrics
        try:
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
        except Exception as e:
            print(f"Evaluation failed: {e}")
            # Return default results
            evaluation_results = {
                'summary': {
                    'overall_accuracy': 0.0,
                    'numerical_accuracy': 0.0,
                    'step_correctness': 0.0,
                    'logical_validity': 0.0,
                    'perplexity': 0.0,
                    'mean_attention_entropy': 0.0,
                    'normalized_attention_entropy': 0.0
                }
            }
        
        # Add epoch information
        evaluation_results['epoch'] = epoch  # type: ignore
        evaluation_results['num_evaluated'] = len(predictions)  # type: ignore
        
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
            try:
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
            except Exception as e:
                print(f"WandB logging failed: {e}")
        
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
        """Enhanced training loop with comprehensive error handling."""
        self.logger.info("Starting mathematical reasoning training...")
        
        # Load datasets
        self.load_datasets()
        
        # Check if datasets were loaded successfully
        if not hasattr(self, 'train_loader') or self.train_loader is None:
            print("ERROR: No training data available. Exiting training.")
            return []
        
        # Training tracking
        best_accuracy = 0.0
        evaluation_history = []
        
        try:
            for epoch in range(self.config['num_epochs']):
                self.logger.info(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
                
                # Training phase
                train_results = self.train_epoch(epoch)
                
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
                    
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"Training error: {e}")
        
        # Save final results
        try:
            self.save_evaluation_history(evaluation_history)
        except Exception as e:
            print(f"Failed to save evaluation history: {e}")
        
        self.logger.info(f"\nTraining completed!")
        self.logger.info(f"Best accuracy achieved: {best_accuracy:.4f}")
        
        return evaluation_history
    
    def save_model(self, filename: str):
        """Save model checkpoint with error handling."""
        try:
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
        except Exception as e:
            print(f"Failed to save model: {e}")
    
    def save_evaluation_history(self, history: List[Dict[str, Any]]):
        """Save evaluation results history with error handling."""
        try:
            results_dir = Path(self.config.get('results_dir', 'results'))
            results_dir.mkdir(exist_ok=True)
            
            filename = f"evaluation_history_{self.config['positional_encoding']}.json"
            with open(results_dir / filename, 'w') as f:
                json.dump(history, f, indent=2, default=str)
            
            self.logger.info(f"Evaluation history saved to {results_dir / filename}")
        except Exception as e:
            print(f"Failed to save evaluation history: {e}")

def get_mathematical_reasoning_config(pe_type: str = "sinusoidal") -> Dict[str, Any]:
    """Get enhanced configuration for mathematical reasoning training."""
    base_config = {
        'model': {
            'd_model': 512,
            'n_heads': 8,
            'd_ff': 2048,
            'n_encoder_layers': 6,
            'n_decoder_layers': 6,
            'vocab_size': 32000,  # Mistral-7B vocab size
            'max_seq_len': 1024,
            'dropout': 0.1,
            'positional_encoding': pe_type
        },
        'tokenizer_name': 'mistralai/Mistral-7B-v0.1',
        'max_length': 1024,
        'batch_size': 4,
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
    print("=== Enhanced Mathematical Reasoning Trainer Starting ===")
    parser = argparse.ArgumentParser(description='Train mathematical reasoning transformer')
    parser.add_argument('--pe_type', default='sinusoidal', 
                       choices=['sinusoidal', 'rope', 'alibi', 'diet', 't5_relative', 'nope'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--experiment_suffix', default='')
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'medium', 'large'],
                        help='Model size. Use small for Kaggle or limited GPU environments.')
    args = parser.parse_args()
    print(f"Arguments: pe_type={args.pe_type}, epochs={args.epochs}, batch_size={args.batch_size}, model_size={args.model_size}")
    config = get_mathematical_reasoning_config(args.pe_type)
    config['num_epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['experiment_suffix'] = args.experiment_suffix
    if args.model_size == 'small':
        config['model'].update({
            'd_model': 256, 
            'n_heads': 8, 
            'd_ff': 1024, 
            'n_encoder_layers': 4, 
            'n_decoder_layers': 4
        })
        config['batch_size'] = 2
        config['eval_batch_size'] = 4
    elif args.model_size == 'medium':
        config['model'].update({
            'd_model': 384, 
            'n_heads': 8, 
            'd_ff': 1536, 
            'n_encoder_layers': 6, 
            'n_decoder_layers': 6
        })
        config['batch_size'] = 4
        config['eval_batch_size'] = 8
    elif args.model_size == 'large':
        config['model'].update({
            'd_model': 512, 
            'n_heads': 8, 
            'd_ff': 2048, 
            'n_encoder_layers': 8, 
            'n_decoder_layers': 8
        })
        config['batch_size'] = 2
        config['eval_batch_size'] = 4
    print(f"Model config: {config['model']}")
    print(f"Training config: batch_size={config['batch_size']}, eval_batch_size={config['eval_batch_size']}")
    try:
        print("Initializing trainer...")
        trainer = MathematicalReasoningTrainer(config)
        print("Starting training...")
        trainer.train()
        print("Training completed!")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
