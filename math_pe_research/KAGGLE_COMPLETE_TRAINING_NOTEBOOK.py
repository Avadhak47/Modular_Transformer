# =============================================================================
# KAGGLE COMPLETE TRAINING NOTEBOOK
# =============================================================================
# Mathematical Reasoning LLM Training with Adaptive Checkpointing
# 
# This notebook provides a complete training pipeline for mathematical reasoning
# models with support for OpenMathInstruct-1M and other datasets.
# 
# Cell Structure:
# 1. Environment Setup & Dependencies
# 2. Training Parameters & Configuration
# 3. Model Training Execution
# 4. Model Evaluation & Testing
# 5. Results Visualization & Analysis
# =============================================================================

# =============================================================================
# CELL 1: ENVIRONMENT SETUP & DEPENDENCIES
# =============================================================================

# Install required packages
!pip install -q transformers>=4.35.0,<4.55.0
!pip install -q peft>=0.7.0
!pip install -q accelerate>=0.25.0
!pip install -q datasets>=2.15.0
!pip install -q wandb>=0.16.0
!pip install -q safetensors>=0.4.0
!pip install -q einops>=0.7.0
!pip install -q sympy>=1.12
!pip install -q scipy>=1.11.0
!pip install -q matplotlib>=3.7.0
!pip install -q pandas>=2.1.0
!pip install -q scikit-learn>=1.3.0
!pip install -q seaborn>=0.12.0
!pip install -q tiktoken>=0.5.0
!pip install -q sentencepiece>=0.1.99

# Import required libraries
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project paths
project_root = Path('/kaggle/working/Transformer')
sys.path.insert(0, str(project_root / 'math_pe_research' / 'src'))

# Set environment variables for Kaggle compatibility
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Disable bitsandbytes to avoid CUDA issues
os.environ['BITSANDBYTES_DISABLE'] = '1'

# Check GPU availability
print("Checking GPU availability...")
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name()}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("No GPU available, using CPU")

# Create necessary directories
os.makedirs('/kaggle/working/Transformer/math_pe_research/outputs', exist_ok=True)
os.makedirs('/kaggle/working/Transformer/math_pe_research/checkpoints', exist_ok=True)
os.makedirs('/kaggle/working/Transformer/math_pe_research/logs', exist_ok=True)

print("Environment setup complete!")

# =============================================================================
# CELL 2: TRAINING PARAMETERS & CONFIGURATION
# =============================================================================

# =============================================================================
# COMPREHENSIVE TRAINING PARAMETERS GUIDE
# =============================================================================

class TrainingConfig:
    """
    Comprehensive training configuration with detailed parameter explanations.
    """
    
    def __init__(self):
        # =====================================================================
        # MODEL CONFIGURATION
        # =====================================================================
        self.model_config = {
            'base_model': 'deepseek-ai/deepseek-math-7b-instruct',  # Base model to fine-tune
            'pe_method': 'rope',  # Positional encoding method
            'use_lora': True,  # Enable LoRA for parameter-efficient fine-tuning
            'load_in_4bit': False,  # 4-bit quantization (saves memory but may reduce performance)
            'load_in_8bit': False,  # 8-bit quantization
            'trust_remote_code': True,  # Trust custom model code
            'torch_dtype': torch.float16,  # Use FP16 for memory efficiency
            'device_map': 'auto',  # Automatic device mapping
            'enable_gradient_checkpointing': False,  # Memory optimization (can cause issues)
        }
        
        # =====================================================================
        # DATASET CONFIGURATION
        # =====================================================================
        self.dataset_config = {
            'datasets': 'openmath_instruct',  # Dataset to use
            'large_scale_training': True,  # Enable large-scale optimizations
            'max_train_samples': 100000,  # Training samples (None for full dataset)
            'max_eval_samples': 20000,  # Evaluation samples
            'max_length': 2048,  # Maximum sequence length
            'streaming': True,  # Enable streaming for large datasets
        }
        
        # =====================================================================
        # TRAINING HYPERPARAMETERS
        # =====================================================================
        self.training_config = {
            'batch_size': 1,  # Batch size per device
            'gradient_accumulation_steps': 32,  # Effective batch size = batch_size * gradient_accumulation_steps
            'learning_rate': 2e-5,  # Learning rate
            'max_steps': 10000,  # Maximum training steps
            'warmup_steps': 100,  # Learning rate warmup steps
            'weight_decay': 0.01,  # Weight decay for regularization
            'max_grad_norm': 1.0,  # Gradient clipping
            'fp16': True,  # Mixed precision training
            'logging_steps': 50,  # Log every N steps
            'eval_steps': 500,  # Evaluate every N steps
            'save_steps': 1000,  # Save every N steps
        }
        
        # =====================================================================
        # ADAPTIVE CHECKPOINTING CONFIGURATION
        # =====================================================================
        self.checkpoint_config = {
            'adaptive_checkpointing': True,  # Enable adaptive checkpointing
            'save_every_samples': 10000,  # Save model every N samples
            'keep_best_models': 5,  # Number of best models to keep
            'random_seed': 42,  # Random seed for reproducibility
        }
        
        # =====================================================================
        # MEMORY OPTIMIZATION CONFIGURATION
        # =====================================================================
        self.memory_config = {
            'memory_efficient': True,  # Enable memory optimizations
            'dataloader_pin_memory': False,  # Disable pin memory for GPU memory
            'dataloader_num_workers': 0,  # Single-threaded data loading
            'remove_unused_columns': False,  # Keep all columns
        }
        
        # =====================================================================
        # EXPERIMENT CONFIGURATION
        # =====================================================================
        self.experiment_config = {
            'experiment_name': 'kaggle_math_training',
            'checkpoint_dir': '/kaggle/working/checkpoints',
            'result_dir': '/kaggle/working/results',
            'cache_dir': '/kaggle/working/data_cache',
            'wandb_project': 'kaggle_math_reasoning',
            'wandb_entity': None,  # Set your wandb username
        }

    def print_parameter_guide(self):
        """Print comprehensive parameter guide with explanations."""
        print("=" * 80)
        print("🎯 COMPREHENSIVE TRAINING PARAMETERS GUIDE")
        print("=" * 80)
        
        print("\n📊 MODEL CONFIGURATION:")
        print("-" * 40)
        for param, value in self.model_config.items():
            print(f"  {param}: {value}")
        
        print("\n📈 DATASET CONFIGURATION:")
        print("-" * 40)
        for param, value in self.dataset_config.items():
            print(f"  {param}: {value}")
        
        print("\n⚙️  TRAINING HYPERPARAMETERS:")
        print("-" * 40)
        for param, value in self.training_config.items():
            print(f"  {param}: {value}")
        
        print("\n🔄 ADAPTIVE CHECKPOINTING:")
        print("-" * 40)
        for param, value in self.checkpoint_config.items():
            print(f"  {param}: {value}")
        
        print("\n💾 MEMORY OPTIMIZATION:")
        print("-" * 40)
        for param, value in self.memory_config.items():
            print(f"  {param}: {value}")
        
        print("\n🧪 EXPERIMENT CONFIGURATION:")
        print("-" * 40)
        for param, value in self.experiment_config.items():
            print(f"  {param}: {value}")

    def print_parameter_explanations(self):
        """Print detailed explanations of each parameter and its effects."""
        print("\n" + "=" * 80)
        print("📚 DETAILED PARAMETER EXPLANATIONS")
        print("=" * 80)
        
        explanations = {
            # Model Configuration
            'base_model': {
                'description': 'Pre-trained model to fine-tune',
                'options': ['deepseek-ai/deepseek-math-7b-instruct', 'deepseek-ai/deepseek-math-1.3b-instruct', 'microsoft/DialoGPT-medium'],
                'effect': 'Larger models have more capacity but require more memory and compute',
                'recommendation': 'Use 7B for best performance, 1.3B for memory constraints'
            },
            'pe_method': {
                'description': 'Positional encoding method for transformer',
                'options': ['rope', 't5_relative', 'alibi', 'sinusoidal', 'diet', 'math_adaptive'],
                'effect': 'Affects how the model understands sequence position and mathematical relationships',
                'recommendation': 'RoPE works well for most cases, T5-relative for long sequences'
            },
            'use_lora': {
                'description': 'Enable Low-Rank Adaptation for parameter-efficient fine-tuning',
                'options': [True, False],
                'effect': 'Reduces trainable parameters by 95%+, enables training on limited hardware',
                'recommendation': 'Always enable for memory-constrained environments'
            },
            'load_in_4bit': {
                'description': 'Load model in 4-bit precision for memory efficiency',
                'options': [True, False],
                'effect': 'Reduces memory usage by ~75% but may reduce performance',
                'recommendation': 'Use only if you run out of memory with 16-bit'
            },
            
            # Dataset Configuration
            'datasets': {
                'description': 'Dataset(s) to use for training',
                'options': ['openmath_instruct', 'gsm8k', 'math', 'metamath', 'mathinstruct'],
                'effect': 'Different datasets have different problem types and difficulty levels',
                'recommendation': 'OpenMathInstruct-1M for large-scale training, GSM8K for evaluation'
            },
            'max_train_samples': {
                'description': 'Maximum number of training samples',
                'options': ['None (full dataset)', '100000', '500000', '800000'],
                'effect': 'More samples = better generalization but longer training time',
                'recommendation': 'Start with 100K, increase based on results and time'
            },
            'max_length': {
                'description': 'Maximum sequence length for tokenization',
                'options': [512, 1024, 2048, 4096],
                'effect': 'Longer sequences capture more context but use more memory',
                'recommendation': '2048 for most mathematical problems, 4096 for complex proofs'
            },
            
            # Training Hyperparameters
            'batch_size': {
                'description': 'Batch size per device',
                'options': [1, 2, 4, 8],
                'effect': 'Larger batches = more stable gradients but more memory',
                'recommendation': 'Use 1 for memory constraints, increase if possible'
            },
            'gradient_accumulation_steps': {
                'description': 'Number of steps to accumulate gradients',
                'options': [16, 32, 64, 128],
                'effect': 'Effective batch size = batch_size * gradient_accumulation_steps',
                'recommendation': 'Use 32-64 for stable training with small batch sizes'
            },
            'learning_rate': {
                'description': 'Learning rate for optimization',
                'options': [1e-5, 2e-5, 5e-5, 1e-4],
                'effect': 'Higher LR = faster convergence but risk of instability',
                'recommendation': 'Start with 2e-5, adjust based on loss curve'
            },
            'max_steps': {
                'description': 'Maximum number of training steps',
                'options': [1000, 5000, 10000, 50000],
                'effect': 'More steps = potentially better performance but longer training',
                'recommendation': 'Start with 10K steps, monitor loss curve'
            },
            
            # Adaptive Checkpointing
            'save_every_samples': {
                'description': 'Save model every N samples',
                'options': [5000, 10000, 20000, 50000],
                'effect': 'More frequent saves = better model selection but more disk usage',
                'recommendation': 'Use 10K for good balance of frequency and efficiency'
            },
            'keep_best_models': {
                'description': 'Number of best models to keep',
                'options': [3, 5, 10],
                'effect': 'More models = better diversity but more disk usage',
                'recommendation': 'Use 5 for good diversity without excessive storage'
            }
        }
        
        for param, info in explanations.items():
            print(f"\n🔧 {param.upper()}:")
            print(f"   Description: {info['description']}")
            print(f"   Options: {info['options']}")
            print(f"   Effect: {info['effect']}")
            print(f"   Recommendation: {info['recommendation']}")

    def get_training_command(self) -> str:
        """Generate training command with current configuration."""
        cmd_parts = [
            'python', 'math_pe_research/scripts/train_and_eval.py',
            '--pe', self.model_config['pe_method'],
            '--datasets', self.dataset_config['datasets'],
            '--max_train_samples', str(self.dataset_config['max_train_samples']),
            '--max_eval_samples', str(self.dataset_config['max_eval_samples']),
            '--batch_size', str(self.training_config['batch_size']),
            '--gradient_accumulation_steps', str(self.training_config['gradient_accumulation_steps']),
            '--learning_rate', str(self.training_config['learning_rate']),
            '--max_steps', str(self.training_config['max_steps']),
            '--experiment_name', self.experiment_config['experiment_name'],
            '--checkpoint_dir', self.experiment_config['checkpoint_dir'],
            '--result_dir', self.experiment_config['result_dir'],
            '--adaptive_checkpointing',
            '--save_every_samples', str(self.checkpoint_config['save_every_samples']),
            '--keep_best_models', str(self.checkpoint_config['keep_best_models']),
            '--memory_efficient',
            '--large_scale_training'
        ]
        
        if self.model_config['use_lora']:
            cmd_parts.append('--use_lora')
        
        if self.model_config['load_in_4bit']:
            cmd_parts.append('--load_in_4bit')
        
        return ' '.join(cmd_parts)

# Initialize configuration
config = TrainingConfig()

# Print parameter guide
config.print_parameter_guide()

# Print detailed explanations
config.print_parameter_explanations()

# Show training command
print("\n" + "=" * 80)
print("🚀 GENERATED TRAINING COMMAND")
print("=" * 80)
print(config.get_training_command())

# =============================================================================
# CELL 3: MODEL TRAINING EXECUTION
# =============================================================================

# Import training modules
from models.mathematical_reasoning_model import create_mathematical_reasoning_model
from data.math_dataset_loader import MathDatasetLoader
from transformers import AutoTokenizer, TrainingArguments, Trainer
import wandb

def start_training():
    """Execute the training process with current configuration."""
    
    print("🚀 Starting Mathematical Reasoning Model Training")
    print("=" * 60)
    
    # Initialize wandb
    try:
        wandb.init(
            project=config.experiment_config['wandb_project'],
            entity=config.experiment_config['wandb_entity'],
            name=config.experiment_config['experiment_name'],
            config={
                **config.model_config,
                **config.dataset_config,
                **config.training_config,
                **config.checkpoint_config
            }
        )
        print("✅ Wandb initialized successfully")
    except Exception as e:
        print(f"⚠️  Wandb initialization failed: {e}")
    
    # Create model
    print("\n📦 Creating model...")
    model = create_mathematical_reasoning_model(
        pe_method=config.model_config['pe_method'],
        base_model=config.model_config['base_model'],
        use_lora=config.model_config['use_lora'],
        load_in_4bit=config.model_config['load_in_4bit'],
        enable_gradient_checkpointing=config.model_config['enable_gradient_checkpointing'],
        torch_dtype=config.model_config['torch_dtype'],
        device_map=config.model_config['device_map']
    )
    
    # Get tokenizer
    tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else model.module.tokenizer
    
    # Load dataset
    print("\n📊 Loading dataset...")
    loader = MathDatasetLoader(
        tokenizer=tokenizer,
        max_length=config.dataset_config['max_length'],
        cache_dir=config.experiment_config['cache_dir'],
        streaming=config.dataset_config['streaming']
    )
    
    # Load training and evaluation data
    train_problems = loader.load_multiple_datasets(
        [config.dataset_config['datasets']], 
        split='train', 
        max_samples_per_dataset=config.dataset_config['max_train_samples']
    )
    
    eval_problems = loader.load_multiple_datasets(
        [config.dataset_config['datasets']], 
        split='test', 
        max_samples_per_dataset=config.dataset_config['max_eval_samples']
    )
    
    # Create PyTorch datasets
    train_dataset = loader.create_pytorch_dataset(train_problems, is_training=True)
    eval_dataset = loader.create_pytorch_dataset(eval_problems, is_training=False)
    
    print(f"📈 Dataset sizes: Train={len(train_dataset)}, Eval={len(eval_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.experiment_config['checkpoint_dir'],
        logging_dir=f"{config.experiment_config['checkpoint_dir']}/logs",
        num_train_epochs=10,
        max_steps=config.training_config['max_steps'],
        per_device_train_batch_size=config.training_config['batch_size'],
        per_device_eval_batch_size=config.training_config['batch_size'],
        learning_rate=config.training_config['learning_rate'],
        logging_steps=config.training_config['logging_steps'],
        fp16=config.training_config['fp16'],
        remove_unused_columns=config.memory_config['remove_unused_columns'],
        gradient_accumulation_steps=config.training_config['gradient_accumulation_steps'],
        warmup_steps=config.training_config['warmup_steps'],
        weight_decay=config.training_config['weight_decay'],
        report_to="wandb",
        dataloader_pin_memory=config.memory_config['dataloader_pin_memory'],
        dataloader_num_workers=config.memory_config['dataloader_num_workers'],
        max_grad_norm=config.training_config['max_grad_norm'],
        eval_strategy="steps",
        eval_steps=config.training_config['eval_steps'],
        save_strategy="steps",
        save_steps=config.training_config['save_steps'],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda features: {
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
            'labels': torch.stack([f['labels'] for f in features])
        }
    )
    
    # Start training
    print("\n🎯 Starting training...")
    print(f"📊 Training for {config.training_config['max_steps']} steps")
    print(f"💾 Effective batch size: {config.training_config['batch_size'] * config.training_config['gradient_accumulation_steps']}")
    
    train_result = trainer.train()
    
    # Save final model
    final_model_path = f"{config.experiment_config['checkpoint_dir']}/final_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"✅ Training completed! Final model saved to: {final_model_path}")
    
    return trainer, train_result

# Execute training
trainer, train_result = start_training()

# =============================================================================
# CELL 4: MODEL EVALUATION & TESTING
# =============================================================================

def evaluate_model():
    """Evaluate the trained model on test data."""
    
    print("🧪 MODEL EVALUATION & TESTING")
    print("=" * 50)
    
    # Load the best model
    model_path = f"{config.experiment_config['checkpoint_dir']}/final_model"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return None
    
    print(f"📦 Loading model from: {model_path}")
    
    # Load model and tokenizer
    model = create_mathematical_reasoning_model(
        pe_method=config.model_config['pe_method'],
        base_model=model_path,
        use_lora=False,  # Don't use LoRA for inference
        load_in_4bit=False,  # Use full precision for evaluation
        torch_dtype=torch.float16
    )
    
    tokenizer = model.tokenizer
    
    # Load test dataset
    loader = MathDatasetLoader(
        tokenizer=tokenizer,
        max_length=config.dataset_config['max_length'],
        cache_dir=config.experiment_config['cache_dir']
    )
    
    test_problems = loader.load_multiple_datasets(
        [config.dataset_config['datasets']], 
        split='test', 
        max_samples_per_dataset=min(1000, config.dataset_config['max_eval_samples'])
    )
    
    test_dataset = loader.create_pytorch_dataset(test_problems, is_training=False)
    
    print(f"📊 Evaluating on {len(test_dataset)} test samples")
    
    # Evaluation metrics
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataset):
            if i >= 100:  # Limit evaluation for speed
                break
                
            input_ids = batch['input_ids'].unsqueeze(0).to(model.device)
            attention_mask = batch['attention_mask'].unsqueeze(0).to(model.device)
            labels = batch['labels'].unsqueeze(0).to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            # Generate predictions
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1)
            
            # Calculate accuracy
            correct_predictions += (pred == labels).sum().item()
            total_predictions += labels.numel()
            
            # Store predictions for analysis
            predictions.extend(pred.cpu().numpy().flatten())
            actuals.extend(labels.cpu().numpy().flatten())
    
    # Calculate metrics
    avg_loss = total_loss / min(100, len(test_dataset))
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    # Save evaluation results
    eval_results = {
        'test_loss': avg_loss,
        'test_accuracy': accuracy,
        'total_samples': min(100, len(test_dataset)),
        'correct_predictions': correct_predictions,
        'total_predictions': total_predictions
    }
    
    # Save results
    results_file = f"{config.experiment_config['result_dir']}/evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\n📊 EVALUATION RESULTS:")
    print(f"   Test Loss: {avg_loss:.4f}")
    print(f"   Test Accuracy: {accuracy:.4f}")
    print(f"   Correct Predictions: {correct_predictions}")
    print(f"   Total Predictions: {total_predictions}")
    print(f"   Results saved to: {results_file}")
    
    return eval_results

# Run evaluation
eval_results = evaluate_model()

# =============================================================================
# CELL 5: RESULTS VISUALIZATION & ANALYSIS
# =============================================================================

def visualize_results():
    """Create comprehensive visualizations of training and evaluation results."""
    
    print("📊 RESULTS VISUALIZATION & ANALYSIS")
    print("=" * 50)
    
    # Load training logs
    log_file = f"{config.experiment_config['checkpoint_dir']}/logs/trainer_state.json"
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            training_logs = json.load(f)
        
        # Extract training metrics
        log_history = training_logs.get('log_history', [])
        
        if log_history:
            # Create training curves
            steps = [log.get('step', 0) for log in log_history]
            train_losses = [log.get('loss', 0) for log in log_history]
            eval_losses = [log.get('eval_loss', 0) for log in log_history]
            learning_rates = [log.get('learning_rate', 0) for log in log_history]
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Training Loss
            ax1.plot(steps, train_losses, 'b-', label='Training Loss')
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss Over Time')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Evaluation Loss
            ax2.plot(steps, eval_losses, 'r-', label='Evaluation Loss')
            ax2.set_xlabel('Training Steps')
            ax2.set_ylabel('Loss')
            ax2.set_title('Evaluation Loss Over Time')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Learning Rate
            ax3.plot(steps, learning_rates, 'g-', label='Learning Rate')
            ax3.set_xlabel('Training Steps')
            ax3.set_ylabel('Learning Rate')
            ax3.set_title('Learning Rate Schedule')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # Combined Loss Plot
            ax4.plot(steps, train_losses, 'b-', label='Training Loss', alpha=0.7)
            ax4.plot(steps, eval_losses, 'r-', label='Evaluation Loss', alpha=0.7)
            ax4.set_xlabel('Training Steps')
            ax4.set_ylabel('Loss')
            ax4.set_title('Training vs Evaluation Loss')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            plt.tight_layout()
            
            # Save plot
            plot_file = f"{config.experiment_config['result_dir']}/training_curves.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📈 Training curves saved to: {plot_file}")
            
            # Show plot
            plt.show()
    
    # Create summary statistics
    print("\n📋 TRAINING SUMMARY:")
    print("-" * 30)
    
    if 'eval_results' in locals():
        print(f"Final Test Loss: {eval_results['test_loss']:.4f}")
        print(f"Final Test Accuracy: {eval_results['test_accuracy']:.4f}")
        print(f"Total Test Samples: {eval_results['total_samples']}")
    
    # Model configuration summary
    print(f"\n🔧 MODEL CONFIGURATION:")
    print("-" * 30)
    print(f"Base Model: {config.model_config['base_model']}")
    print(f"PE Method: {config.model_config['pe_method']}")
    print(f"LoRA Enabled: {config.model_config['use_lora']}")
    print(f"4-bit Quantization: {config.model_config['load_in_4bit']}")
    
    print(f"\n📊 DATASET CONFIGURATION:")
    print("-" * 30)
    print(f"Dataset: {config.dataset_config['datasets']}")
    print(f"Max Train Samples: {config.dataset_config['max_train_samples']}")
    print(f"Max Eval Samples: {config.dataset_config['max_eval_samples']}")
    print(f"Max Length: {config.dataset_config['max_length']}")
    
    print(f"\n⚙️  TRAINING CONFIGURATION:")
    print("-" * 30)
    print(f"Batch Size: {config.training_config['batch_size']}")
    print(f"Gradient Accumulation: {config.training_config['gradient_accumulation_steps']}")
    print(f"Effective Batch Size: {config.training_config['batch_size'] * config.training_config['gradient_accumulation_steps']}")
    print(f"Learning Rate: {config.training_config['learning_rate']}")
    print(f"Max Steps: {config.training_config['max_steps']}")
    
    # Save comprehensive report
    report = {
        'model_config': config.model_config,
        'dataset_config': config.dataset_config,
        'training_config': config.training_config,
        'checkpoint_config': config.checkpoint_config,
        'evaluation_results': eval_results if 'eval_results' in locals() else None,
        'training_summary': {
            'total_steps': config.training_config['max_steps'],
            'effective_batch_size': config.training_config['batch_size'] * config.training_config['gradient_accumulation_steps'],
            'total_samples': config.dataset_config['max_train_samples']
        }
    }
    
    report_file = f"{config.experiment_config['result_dir']}/comprehensive_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Comprehensive report saved to: {report_file}")
    
    # Create model performance comparison
    if os.path.exists(f"{config.experiment_config['checkpoint_dir']}/model_metrics.json"):
        print("\n🏆 MODEL PERFORMANCE COMPARISON:")
        print("-" * 40)
        
        with open(f"{config.experiment_config['checkpoint_dir']}/model_metrics.json", 'r') as f:
            metrics = json.load(f)
        
        best_models = metrics.get('best_models', [])
        
        if best_models:
            print(f"{'Rank':<4} {'Step':<8} {'Loss':<10} {'Accuracy':<10}")
            print("-" * 35)
            
            for i, model in enumerate(best_models[:5], 1):
                loss = model['metrics'].get('eval_loss', 0)
                accuracy = model['metrics'].get('eval_accuracy', 0)
                step = model['step']
                print(f"{i:<4} {step:<8} {loss:<10.4f} {accuracy:<10.4f}")

# Run visualization
visualize_results()

print("\n🎉 TRAINING PIPELINE COMPLETE!")
print("=" * 50)
print("📁 Check the following directories for outputs:")
print(f"   Models: {config.experiment_config['checkpoint_dir']}")
print(f"   Results: {config.experiment_config['result_dir']}")
print(f"   Cache: {config.experiment_config['cache_dir']}")
print("\n🚀 You can now use the trained model for mathematical reasoning tasks!") 