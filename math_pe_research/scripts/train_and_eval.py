# ===============================
# Recent Fixes Summary Table
# ===============================
# | Issue/Request                                 | Fix/Change                                                                                 |
# |-----------------------------------------------|------------------------------------------------------------------------------------------|
# | DataParallel attribute error                  | Use model.module.tokenizer if hasattr(model, 'module') else model.tokenizer everywhere   |
# | AttributeError: 'Namespace' has no fp16       | Add --fp16 argument to ArgumentParser                                                     |
# | RecursionError in data_collator               | Rename wrapper to safe_data_collator, use in Trainer                                      |
# | Device mismatch for PE/model                  | All PE/model .to(device), DataParallel for multi-GPU, input tensors moved to device      |
# | All PE parameters on correct device           | .to(device) methods for all PE classes                                                   |
# | Model/PE multi-GPU support                    | get_best_device(), DataParallel, robust device handling                                  |
# | Robust config/tokenizer/model access          | Always check hasattr(model, 'module') for attribute access                               |
# ===============================
# Note on PyTorch Warning:
# /usr/local/lib/python3.11/dist-packages/torch/utils/checkpoint.py:87: UserWarning: None of the inputs have requires_grad=True. Gradients will be None
# - This warning is usually harmless if you are not training, or if your model/LoRA parameters are set up correctly.
# - If you are training and see this warning, ensure your model is in .train() mode and all trainable parameters have requires_grad=True.
# - The HuggingFace Trainer should handle this for you, so you can usually ignore this unless your model is not learning.
# ===============================

import argparse
import os
import sys
import json
from pathlib import Path

# Add the src directory to Python path for module imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

import torch
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, DataCollatorForLanguageModeling
import wandb

from models.mathematical_reasoning_model import create_mathematical_reasoning_model
from data.math_dataset_loader import MathDatasetLoader

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
    
    # Data loading
    dataset_names = [d.strip() for d in args.datasets.split(",")]
    loader = MathDatasetLoader(tokenizer=tokenizer, max_length=args.max_length, cache_dir=str(cache_dir))
    
    # Reduce dataset sizes for memory efficiency
    max_train_samples = 1000 if args.memory_efficient else 10000
    max_eval_samples = 200 if args.memory_efficient else 1000
    
    print(f"📊 Loading datasets with max {max_train_samples} train and {max_eval_samples} eval samples...")
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