#!/usr/bin/env python3
"""
Memory-efficient training script for mathematical reasoning models.
Optimized for GPU memory constraints.
"""

import os
import sys
import torch
from pathlib import Path

# Set memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Add the src directory to Python path for module imports
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))

from scripts.train_and_eval import main

if __name__ == "__main__":
    # Override sys.argv with memory-efficient defaults
    sys.argv = [
        'train_memory_efficient.py',
        '--experiment_name', 'memory_efficient_test',
        '--checkpoint_dir', '/kaggle/working/checkpoints',
        '--result_dir', '/kaggle/working/results',
        '--pe', 'rope',
        '--model_size', 'microsoft/DialoGPT-small',  # Use smaller model
        '--max_steps', '50',  # Reduce steps for testing
        '--batch_size', '1',  # Minimal batch size
        '--gradient_accumulation_steps', '16',  # Increase accumulation
        '--max_length', '1024',  # Reduce sequence length
        '--learning_rate', '2e-5',
        '--fp16',  # Enable mixed precision
        '--memory_efficient',  # Enable memory optimizations
        '--datasets', 'gsm8k',  # Use only one dataset
    ]
    
    print("🚀 Starting memory-efficient training...")
    print("📋 Settings:")
    print("   - Batch size: 1")
    print("   - Gradient accumulation: 16")
    print("   - Max length: 1024")
    print("   - FP16: enabled")
    print("   - Memory efficient: enabled")
    
    main() 