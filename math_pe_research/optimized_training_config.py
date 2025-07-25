#!/usr/bin/env python3
"""
Optimized training configuration for Pythia-2.8B on 16GB GPU.
"""

import os
import sys
from pathlib import Path

# Memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def get_optimized_config():
    """Get optimized configuration for Pythia-2.8B on 16GB GPU."""
    return {
        'model_size': 'EleutherAI/pythia-2.8b',
        'pe': 'rope',
        'batch_size': 1,
        'gradient_accumulation_steps': 8,  # Effective batch size = 8
        'max_length': 512,
        'max_steps': 100,
        'learning_rate': 2e-5,
        'fp16': True,
        'memory_efficient': True,
        'enable_gradient_checkpointing': True,  # Enable for memory savings
        'datasets': 'gsm8k',  # Use only one dataset initially
        'max_train_samples': 500,  # Reduce dataset size
        'max_eval_samples': 100,
        'warmup_steps': 10,
        'logging_steps': 10,
        'save_steps': 50,
        'eval_steps': 50,
        'dataloader_pin_memory': False,
        'dataloader_num_workers': 0,
        'max_grad_norm': 1.0,
        'weight_decay': 0.01,
    }

def get_memory_breakdown():
    """Show memory breakdown for Pythia-2.8B."""
    print("📊 Memory Breakdown for Pythia-2.8B (16GB GPU):")
    print("   Model weights (FP16): ~5.6GB")
    print("   Activations (512 seq): ~2.5GB")
    print("   Gradients (LoRA only): ~0.5GB")
    print("   Optimizer states (LoRA): ~0.2GB")
    print("   CUDA overhead: ~1.5GB")
    print("   Total estimated: ~10.3GB")
    print("   Available buffer: ~5.6GB")

def get_training_command():
    """Generate optimized training command."""
    config = get_optimized_config()
    
    cmd = f"""python scripts/train_and_eval.py \\
  --model_size {config['model_size']} \\
  --pe {config['pe']} \\
  --batch_size {config['batch_size']} \\
  --gradient_accumulation_steps {config['gradient_accumulation_steps']} \\
  --max_length {config['max_length']} \\
  --max_steps {config['max_steps']} \\
  --learning_rate {config['learning_rate']} \\
  --fp16 \\
  --memory_efficient \\
  --enable_gradient_checkpointing \\
  --datasets {config['datasets']} \\
  --experiment_name pythia_2_8b_optimized \\
  --checkpoint_dir /kaggle/working/checkpoints \\
  --result_dir /kaggle/working/results \\
  --cache_dir /tmp/model_cache \\
  --wandb_project kaggle_math_reasoning \\
  --use_lora"""
    
    return cmd

if __name__ == "__main__":
    print("🚀 Optimized Configuration for Pythia-2.8B")
    print("=" * 50)
    
    config = get_optimized_config()
    print("\n📋 Recommended Settings:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print("\n💾 Memory Analysis:")
    get_memory_breakdown()
    
    print("\n🔧 Training Command:")
    print(get_training_command())
    
    print("\n💡 Tips:")
    print("   - Start with single dataset (gsm8k)")
    print("   - Use gradient checkpointing for memory savings")
    print("   - Monitor memory with --memory_efficient flag")
    print("   - Consider reducing max_length to 256 if needed") 