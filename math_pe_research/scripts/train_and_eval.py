#!/usr/bin/env python3
"""
Training and evaluation script for mathematical reasoning model.
Updated to handle Kaggle bitsandbytes compatibility issues.
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set environment variables to avoid bitsandbytes issues
os.environ['BITSANDBYTES_DISABLE'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Handle transformers import issues
try:
    from transformers import (
        AutoTokenizer, 
        TrainingArguments, 
        Trainer,
        DataCollatorForLanguageModeling
    )
except ImportError as e:
    if "cannot import name 'requires'" in str(e):
        print("Transformers import issue detected. Trying to fix...")
        # Try to install a compatible version
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "transformers==4.35.0"])
        from transformers import (
            AutoTokenizer, 
            TrainingArguments, 
            Trainer,
            DataCollatorForLanguageModeling
        )
    else:
        raise e

from models.mathematical_reasoning_model import create_mathematical_reasoning_model
from data.math_dataset_loader import MathDatasetLoader
import wandb
from sklearn.metrics import accuracy_score
import random
import shutil
from typing import List, Dict, Optional, Tuple
import pickle

class AdaptiveCheckpointCallback:
    """
    Custom callback for adaptive checkpointing with model comparison and random initialization.
    Saves models every 10K samples, compares performance, keeps best 5, and randomly initializes.
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        save_every_samples: int = 10000,
        keep_best_models: int = 5,
        eval_metrics: List[str] = ['eval_loss', 'eval_accuracy'],
        random_seed: int = 42
    ):
        self.checkpoint_dir = checkpoint_dir
        self.save_every_samples = save_every_samples
        self.keep_best_models = keep_best_models
        self.eval_metrics = eval_metrics
        self.random_seed = random_seed
        self.best_models = []  # List of (model_path, metrics) tuples
        self.samples_trained = 0
        self.last_save_samples = 0
        
        # Create checkpoint subdirectories
        self.models_dir = checkpoint_dir / "adaptive_models"
        self.models_dir.mkdir(exist_ok=True)
        self.metrics_file = checkpoint_dir / "model_metrics.json"
        
        # Load existing metrics if available
        self._load_metrics()
        
        print(f"🔄 Adaptive checkpointing initialized:")
        print(f"   📁 Models directory: {self.models_dir}")
        print(f"   💾 Save every {self.save_every_samples:,} samples")
        print(f"   🏆 Keep best {self.keep_best_models} models")
        print(f"   📊 Tracking metrics: {self.eval_metrics}")
    
    def _load_metrics(self):
        """Load existing model metrics from file."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.best_models = data.get('best_models', [])
                    self.samples_trained = data.get('samples_trained', 0)
                    self.last_save_samples = data.get('last_save_samples', 0)
                print(f"📈 Loaded {len(self.best_models)} existing model metrics")
            except Exception as e:
                print(f"⚠️  Could not load existing metrics: {e}")
    
    def _save_metrics(self):
        """Save current model metrics to file."""
        data = {
            'best_models': self.best_models,
            'samples_trained': self.samples_trained,
            'last_save_samples': self.last_save_samples
        }
        with open(self.metrics_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _evaluate_model(self, model, eval_dataset, tokenizer) -> Dict[str, float]:
        """Evaluate model and return metrics."""
        model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Sample a subset for quick evaluation
        eval_subset = torch.utils.data.Subset(eval_dataset, 
                                             random.sample(range(len(eval_dataset)), 
                                                         min(100, len(eval_dataset))))
        
        with torch.no_grad():
            for batch in torch.utils.data.DataLoader(eval_subset, batch_size=1, shuffle=False):
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                labels = batch['labels'].to(model.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                
                # Simple accuracy calculation (you can enhance this)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.numel()
        
        model.train()
        
        return {
            'eval_loss': total_loss / len(eval_subset),
            'eval_accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0.0
        }
    
    def _save_model(self, model, tokenizer, metrics: Dict[str, float], step: int):
        """Save model with metrics."""
        model_path = self.models_dir / f"model_step_{step}_loss_{metrics.get('eval_loss', 0):.4f}"
        
        try:
            # Save model
            model.save_pretrained(str(model_path))
            tokenizer.save_pretrained(str(model_path))
            
            # Save metrics
            metrics_file = model_path / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Add to best models list
            self.best_models.append({
                'path': str(model_path),
                'step': step,
                'metrics': metrics,
                'samples_trained': self.samples_trained
            })
            
            # Sort by primary metric (eval_loss) and keep only best models
            self.best_models.sort(key=lambda x: x['metrics'].get('eval_loss', float('inf')))
            self.best_models = self.best_models[:self.keep_best_models]
            
            # Save updated metrics
            self._save_metrics()
            
            print(f"💾 Saved model at step {step}:")
            print(f"   📁 Path: {model_path}")
            print(f"   📊 Loss: {metrics.get('eval_loss', 0):.4f}")
            print(f"   🎯 Accuracy: {metrics.get('eval_accuracy', 0):.4f}")
            print(f"   🏆 Best models kept: {len(self.best_models)}")
            
            return str(model_path)
            
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
            return None
    
    def _load_random_best_model(self, model_class, **kwargs):
        """Load a random model from the best models list."""
        if not self.best_models:
            print("⚠️  No best models available, starting fresh")
            return None
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        
        # Select random model from best models
        selected_model = random.choice(self.best_models)
        model_path = selected_model['path']
        
        try:
            print(f"🔄 Loading random best model:")
            print(f"   📁 Path: {model_path}")
            print(f"   📊 Loss: {selected_model['metrics'].get('eval_loss', 0):.4f}")
            print(f"   🎯 Accuracy: {selected_model['metrics'].get('eval_accuracy', 0):.4f}")
            
            # Load model
            model = model_class.from_pretrained(model_path, **kwargs)
            return model
            
        except Exception as e:
            print(f"❌ Failed to load model from {model_path}: {e}")
            return None
    
    def on_step_end(self, args, state, control, model, tokenizer, eval_dataset, **kwargs):
        """Called at the end of each training step."""
        # Update samples trained
        self.samples_trained += args.per_device_train_batch_size * args.gradient_accumulation_steps
        
        # Check if we should save a checkpoint
        if (self.samples_trained - self.last_save_samples) >= self.save_every_samples:
            print(f"\n🔄 Adaptive checkpointing triggered at {self.samples_trained:,} samples")
            
            # Evaluate current model
            metrics = self._evaluate_model(model, eval_dataset, tokenizer)
            
            # Save model
            saved_path = self._save_model(model, tokenizer, metrics, state.global_step)
            
            if saved_path:
                self.last_save_samples = self.samples_trained
                
                # Optionally load a random best model for next phase
                if len(self.best_models) > 1:  # Only if we have multiple models
                    print("🎲 Randomly selecting best model for next phase...")
                    # Note: In a real implementation, you'd restart training with the selected model
                    # For now, we just log the selection
                    selected = random.choice(self.best_models)
                    print(f"   🎯 Selected: {selected['path']}")
    
    def get_best_model_path(self) -> Optional[str]:
        """Get the path of the best performing model."""
        if not self.best_models:
            return None
        
        # Return the model with lowest eval_loss
        best_model = min(self.best_models, key=lambda x: x['metrics'].get('eval_loss', float('inf')))
        return best_model['path']

class AdaptiveTrainer(Trainer):
    """
    Custom Trainer that integrates with adaptive checkpointing.
    """
    
    def __init__(self, adaptive_callback=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive_callback = adaptive_callback
    
    def training_step(self, model, inputs):
        """Override training step to integrate adaptive checkpointing."""
        result = super().training_step(model, inputs)
        
        # Call adaptive checkpointing callback if available
        if self.adaptive_callback:
            self.adaptive_callback.on_step_end(
                args=self.args,
                state=self.state,
                control=None,
                model=model,
                tokenizer=self.tokenizer,
                eval_dataset=self.eval_dataset
            )
        
        return result

def print_memory_usage():
    """Print current memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"💾 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
    else:
        print("💾 Using CPU")

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a mathematical reasoning model.")
    parser.add_argument('--fp16', action='store_true', help='Use FP16 mixed precision training')
    parser.add_argument('--model_size', type=str, default="deepseek-ai/deepseek-math-7b-instruct",
                        help="Base model size/name (e.g., deepseek-ai/deepseek-math-7b-instruct, deepseek-ai/deepseek-math-1.3b-instruct, etc.)")
    parser.add_argument('--pe', type=str, default="rope", choices=["rope", "alibi", "sinusoidal", "diet", "t5_relative", "math_adaptive"],
                        help="Positional encoding method")
    parser.add_argument('--experiment_name', type=str, required=True, help="Experiment name")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument('--result_dir', type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument('--max_steps', type=int, default=1000, help="Max training steps")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size (reduced for GPU memory)")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="Learning rate")
    parser.add_argument('--max_length', type=int, default=2048, help="Max sequence length (reduced for memory)")
    parser.add_argument('--datasets', type=str, default="gsm8k,math", help="Comma-separated list of datasets")
    parser.add_argument('--wandb_project', type=str, default="math_pe_research", help="wandb project name")
    parser.add_argument('--wandb_entity', type=str, default=None, help="wandb entity (optional)")
    parser.add_argument('--cache_dir', type=str, default=None, help='Directory to cache downloaded models and datasets')
    parser.add_argument('--load_in_4bit', action='store_true', help='Enable 4-bit quantization (requires bitsandbytes, not supported on Kaggle by default)')
    parser.add_argument('--use_lora', action='store_true', default=True, help='Enable LoRA fine-tuning (enabled by default for training)')
    parser.add_argument('--no_lora', action='store_true', help='Disable LoRA fine-tuning')
    parser.add_argument('--enable_gradient_checkpointing', action='store_true', help='Enable gradient checkpointing (disabled by default to prevent training issues)')
    parser.add_argument('--memory_efficient', action='store_true', default=True, help='Enable memory efficient settings')
    parser.add_argument('--large_scale_training', action='store_true', help='Enable large-scale training for OpenMathInstruct-1M')
    parser.add_argument('--max_train_samples', type=int, default=None, help='Maximum training samples per dataset (None for full dataset)')
    parser.add_argument('--max_eval_samples', type=int, default=None, help='Maximum evaluation samples per dataset (None for full dataset)')
    parser.add_argument('--adaptive_checkpointing', action='store_true', help='Enable adaptive checkpointing with model comparison')
    parser.add_argument('--save_every_samples', type=int, default=10000, help='Save model every N samples')
    parser.add_argument('--keep_best_models', type=int, default=5, help='Number of best models to keep')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for model selection')
    args = parser.parse_args()

    # Handle LoRA argument conflict
    if args.no_lora:
        args.use_lora = False

    # Detect Kaggle environment and adjust defaults
    is_kaggle = '/kaggle/' in os.getcwd() or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    if is_kaggle:
        print("🔍 Kaggle environment detected - applying compatibility settings...")
        # Keep LoRA enabled by default for training
        if args.use_lora:
            print("   ✅ LoRA enabled for training")
        if args.load_in_4bit:
            print("   ⚠️  4-bit quantization may cause issues on Kaggle")
    
    # Memory optimization settings
    if args.memory_efficient:
        print("🔧 Applying memory efficient settings...")
        # Set PyTorch memory allocation settings
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        # Enable memory efficient attention if available
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        print("   ✅ Memory optimization enabled")
    
    # Large-scale training optimizations
    if args.large_scale_training:
        print("🚀 Applying large-scale training optimizations...")
        # Reduce batch size for large datasets
        if args.batch_size > 1:
            args.batch_size = max(1, args.batch_size // 2)
            print(f"   📉 Reduced batch size to {args.batch_size} for memory efficiency")
        
        # Increase gradient accumulation to maintain effective batch size
        if args.gradient_accumulation_steps < 32:
            args.gradient_accumulation_steps = min(64, args.gradient_accumulation_steps * 2)
            print(f"   📈 Increased gradient accumulation to {args.gradient_accumulation_steps}")
        
        # Enable more aggressive memory optimizations
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        print("   ✅ Large-scale optimizations enabled")
    
    # Setup directories
    checkpoint_dir = Path(args.checkpoint_dir)
    result_dir = Path(args.result_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    else:
        cache_dir = checkpoint_dir / "data_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

    # wandb setup
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.experiment_name,
        config=vars(args)
    )

    # Load model
    print("🔄 Loading model...")
    model = create_mathematical_reasoning_model(
        pe_method=args.pe,
        base_model=args.model_size,
        load_in_4bit=args.load_in_4bit,
        use_lora=args.use_lora,
        enable_gradient_checkpointing=args.enable_gradient_checkpointing,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu"
    )
    
    # Memory cleanup after model loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"💾 GPU memory after model loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Robust tokenizer access for DataParallel
    tokenizer = model.module.tokenizer if hasattr(model, 'module') else model.tokenizer
    # Robust config access for DataParallel
    config = model.module.config if hasattr(model, 'module') else model.config
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for decoder-only models
    
    # Create data collator for causal language modeling
    base_data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def safe_data_collator(features):
        tensor_fields = {"input_ids", "attention_mask", "labels"}
        filtered_features = []
        for feature in features:
            filtered_feature = {k: v for k, v in feature.items() if k in tensor_fields or isinstance(v, (list, int, float))}
            filtered_features.append(filtered_feature)
        return base_data_collator(filtered_features)
    
    # Data loading with large-scale support
    dataset_names = [d.strip() for d in args.datasets.split(",")]
    # Enable streaming for large datasets
    streaming_enabled = args.large_scale_training and any('openmath' in name.lower() for name in dataset_names)
    loader = MathDatasetLoader(
        tokenizer=tokenizer, 
        max_length=args.max_length, 
        cache_dir=str(cache_dir),
        streaming=streaming_enabled
    )
    
    if streaming_enabled:
        print("📡 Streaming mode enabled for large dataset loading")
    
    # Configure dataset sizes based on training mode
    if args.large_scale_training:
        # For OpenMathInstruct-1M: use 70-80% for training
        max_train_samples = args.max_train_samples or 800000  # 800K for training
        max_eval_samples = args.max_eval_samples or 200000    # 200K for evaluation
        print(f"🚀 Large-scale training mode: {max_train_samples:,} train, {max_eval_samples:,} eval samples")
    else:
        # Memory-efficient mode for smaller datasets
        max_train_samples = args.max_train_samples or (1000 if args.memory_efficient else 10000)
        max_eval_samples = args.max_eval_samples or (200 if args.memory_efficient else 1000)
        print(f"📊 Standard training mode: {max_train_samples:,} train, {max_eval_samples:,} eval samples")
    
    print(f"📊 Loading datasets with max {max_train_samples:,} train and {max_eval_samples:,} eval samples...")
    train_problems = loader.load_multiple_datasets(dataset_names, split='train', max_samples_per_dataset=max_train_samples)
    eval_problems = loader.load_multiple_datasets(dataset_names, split='test', max_samples_per_dataset=max_eval_samples)
    train_dataset = loader.create_pytorch_dataset(train_problems, is_training=True)
    eval_dataset = loader.create_pytorch_dataset(eval_problems, is_training=False)
    
    print(f"📈 Dataset sizes: Train={len(train_dataset)}, Eval={len(eval_dataset)}")

    # Training arguments with version compatibility
    import transformers
    from packaging import version
    
    ta_params = {
        'output_dir': str(checkpoint_dir),
        'logging_dir': str(checkpoint_dir / "logs"),
        'num_train_epochs': 10,
        'max_steps': args.max_steps,
        'per_device_train_batch_size': args.batch_size,
        'per_device_eval_batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'logging_steps': 50,
        'fp16': True,
        'remove_unused_columns': False,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'warmup_steps': 100,
        'weight_decay': 0.01,
        'report_to': "wandb",
        # Memory optimization
        'dataloader_pin_memory': False,
        'dataloader_num_workers': 0,
        'max_grad_norm': 1.0,
        # Disable automatic saving to avoid safetensors shared tensor issue
        'save_strategy': 'no',
        'save_steps': None,
    }
    
    # Add version-specific arguments
    transformers_version = version.parse(transformers.__version__)
    if transformers_version >= version.parse("4.0.0"):
        # Modern transformers (>=4.0.0)
        ta_params.update({
            'eval_strategy': "steps",
            'eval_steps': 250,
            'save_strategy': "steps",
            'save_steps': 500,
            'save_total_limit': 3,
            'load_best_model_at_end': True,
            'metric_for_best_model': "eval_loss",
            'greater_is_better': False,
        })
    else:
        # Older transformers (<4.0.0)
        ta_params.update({
            'do_eval': True,
            'eval_steps': 250,
            'save_steps': 500,
            'save_total_limit': 3,
        })
    
    training_args = TrainingArguments(**ta_params)

    # Trainer with version-compatible callbacks
    callbacks = []
    if transformers_version >= version.parse("4.0.0"):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.00001))
    
    # Add adaptive checkpointing callback
    if args.adaptive_checkpointing:
        adaptive_callback = AdaptiveCheckpointCallback(
            checkpoint_dir=checkpoint_dir,
            save_every_samples=args.save_every_samples,
            keep_best_models=args.keep_best_models,
            random_seed=args.random_seed
        )
        callbacks.append(adaptive_callback)
        print(f"🔄 Adaptive checkpointing enabled:")
        print(f"   💾 Save every {args.save_every_samples:,} samples")
        print(f"   🏆 Keep best {args.keep_best_models} models")
        print(f"   🎲 Random seed: {args.random_seed}")
    
    # Use custom trainer for adaptive checkpointing
    if args.adaptive_checkpointing:
        trainer = AdaptiveTrainer(
            adaptive_callback=adaptive_callback,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=safe_data_collator,
            callbacks=callbacks,
            compute_metrics=None
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=safe_data_collator,
            callbacks=callbacks,
            compute_metrics=None
        )

    # Double-check: Ensure model is in train mode before training
    model.train()
    
    # Check and ensure trainable parameters exist
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total ({100 * trainable_params / total_params:.2f}%)")
    print_memory_usage()
    
    if trainable_params == 0:
        print("⚠️  WARNING: No trainable parameters found! This will cause training to fail.")
        print("Attempting to unfreeze parameters...")
        
        # Try to unfreeze some parameters
        for name, param in model.named_parameters():
            if any(key in name.lower() for key in ['lora', 'adapter', 'pe', 'positional', 'embedding']):
                param.requires_grad = True
                print(f"Unfroze parameter: {name}")
        
        # Recheck
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"After unfreezing: {trainable_params:,} trainable parameters")
        
        if trainable_params == 0:
            raise RuntimeError("No trainable parameters found even after unfreezing. Check LoRA configuration.")
    
    # Double-check: Ensure all trainable parameters have requires_grad=True
    for n, p in model.named_parameters():
        if p.requires_grad is False and p.grad is not None:
            print(f"Warning: Parameter {n} does not require grad but has a gradient!")

    # Training
    print("Starting training...")
    print_memory_usage()
    train_result = trainer.train()
    print("Training complete.")
    print_memory_usage()

    # Get best model path if using adaptive checkpointing
    best_model_path = None
    if args.adaptive_checkpointing:
        best_model_path = adaptive_callback.get_best_model_path()
        if best_model_path:
            print(f"🏆 Best model found: {best_model_path}")
        else:
            print("⚠️  No best model found in adaptive checkpointing")

    # Save final model and tokenizer using custom method
    if args.checkpoint_dir:
        save_path = os.path.join(args.checkpoint_dir, "final_model")
        try:
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"✅ Final model saved to {save_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save final model: {e}")
            print("   Model weights are still updated and available for inference")

    # Evaluation
    print("Running evaluation...")
    eval_result = trainer.evaluate()
    print("Evaluation complete.")

    # Save evaluation results
    results = {
        'experiment_name': args.experiment_name,
        'pe_method': args.pe,
        'model_size': args.model_size,
        'train_loss': train_result.training_loss,
        'eval_loss': eval_result.get('eval_loss', None),
        'train_samples': len(train_dataset),
        'eval_samples': len(eval_dataset),
        'total_steps': train_result.global_step
    }
    results_file = result_dir / "final_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")

    # wandb log and finish
    wandb.log(results)
    wandb.finish()

if __name__ == "__main__":
    main() 