#!/usr/bin/env python3
"""
Kaggle Setup Script for Mathematical Reasoning Model Training

This script handles all the environment setup and compatibility issues
for running the mathematical reasoning model training on Kaggle.

Usage:
    python kaggle_setup.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_status(message, status="INFO"):
    """Print colored status messages."""
    colors = {
        "INFO": "\033[94m",     # Blue
        "SUCCESS": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",    # Red
        "RESET": "\033[0m"      # Reset
    }
    print(f"{colors.get(status, '')}{status}: {message}{colors['RESET']}")

def run_command(cmd, check=True, capture_output=False):
    """Run a command and handle errors."""
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=True, check=check)
            return True
    except subprocess.CalledProcessError as e:
        print_status(f"Command failed: {cmd}", "ERROR")
        print_status(f"Error: {e}", "ERROR")
        return False

def check_kaggle_environment():
    """Detect if running on Kaggle."""
    is_kaggle = (
        '/kaggle/' in os.getcwd() or 
        'KAGGLE_KERNEL_RUN_TYPE' in os.environ or
        Path('/kaggle/working').exists()
    )
    
    if is_kaggle:
        print_status("Kaggle environment detected! ✅", "SUCCESS")
        print_status(f"Working directory: {os.getcwd()}", "INFO")
        print_status(f"Python version: {sys.version}", "INFO")
    else:
        print_status("Local environment detected", "INFO")
    
    return is_kaggle

def fix_numpy_compatibility():
    """Fix NumPy 2.x compatibility issues."""
    print_status("🔧 Fixing NumPy compatibility...", "INFO")
    
    # Check current numpy version
    try:
        import numpy as np
        current_version = np.__version__
        print_status(f"Current NumPy version: {current_version}", "INFO")
        
        if current_version.startswith('2.'):
            print_status("NumPy 2.x detected - downgrading for compatibility", "WARNING")
            run_command("pip install 'numpy<2.0' --upgrade --quiet")
            
            # Restart Python to load new numpy
            print_status("⚠️  Please restart your Kaggle kernel for NumPy changes to take effect!", "WARNING")
            return False
        else:
            print_status("NumPy version is compatible ✅", "SUCCESS")
            return True
            
    except ImportError:
        print_status("NumPy not found, installing compatible version", "WARNING")
        run_command("pip install 'numpy<2.0' --quiet")
        return True

def install_dependencies():
    """Install required dependencies with compatibility fixes."""
    print_status("📦 Installing dependencies...", "INFO")
    
    # Core dependencies with version constraints
    core_deps = [
        "'numpy<2.0'",
        "'torch>=2.0.0,<2.2.0'",
        "'transformers>=4.35.0,<4.46.0'",
        "'accelerate>=0.25.0'",
        "'peft>=0.7.0'",
        "'datasets>=2.15.0'",
        "'wandb>=0.16.0'",
        "'scikit-learn>=1.3.0'",
        "'matplotlib<3.8.0'",  # Pin for numpy compatibility
        "'pandas>=2.0.0'",
        "'tqdm>=4.66.0'"
    ]
    
    # Install core dependencies
    for dep in core_deps:
        print_status(f"Installing {dep}...", "INFO")
        success = run_command(f"pip install {dep} --upgrade --quiet")
        if not success:
            print_status(f"Failed to install {dep}", "ERROR")
    
    # Optional dependencies (don't fail if these don't work)
    optional_deps = [
        "'bitsandbytes>=0.41.0'",  # May not work on all Kaggle instances
    ]
    
    for dep in optional_deps:
        print_status(f"Installing optional {dep}...", "INFO")
        success = run_command(f"pip install {dep} --upgrade --quiet", check=False)
        if success:
            print_status(f"✅ {dep} installed successfully", "SUCCESS")
        else:
            print_status(f"⚠️  {dep} installation failed (optional)", "WARNING")

def setup_directories():
    """Setup required directories for Kaggle."""
    print_status("📁 Setting up directories...", "INFO")
    
    dirs = [
        '/kaggle/working/checkpoints',
        '/kaggle/working/evaluation_results', 
        '/kaggle/working/data_cache',
        '/kaggle/working/logs'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print_status(f"Created directory: {dir_path}", "SUCCESS")

def check_gpu():
    """Check GPU availability."""
    print_status("🔍 Checking GPU availability...", "INFO")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print_status(f"✅ GPU available: {gpu_name} (Count: {gpu_count})", "SUCCESS")
            print_status(f"CUDA version: {torch.version.cuda}", "INFO")
            return True
        else:
            print_status("❌ No GPU available", "WARNING")
            return False
    except ImportError:
        print_status("PyTorch not available for GPU check", "WARNING")
        return False

def setup_environment_variables():
    """Set important environment variables."""
    print_status("🌍 Setting environment variables...", "INFO")
    
    env_vars = {
        'TOKENIZERS_PARALLELISM': 'false',  # Prevent tokenizer warnings
        'WANDB_SILENT': 'true',             # Reduce wandb verbosity
        'HF_HOME': '/kaggle/working/data_cache',  # Hugging Face cache
        'TRANSFORMERS_CACHE': '/kaggle/working/data_cache',
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print_status(f"Set {key}={value}", "SUCCESS")

def main():
    """Main setup function."""
    print_status("🚀 Starting Kaggle Environment Setup for Mathematical Reasoning Model", "INFO")
    print("=" * 80)
    
    # Step 1: Check environment
    is_kaggle = check_kaggle_environment()
    
    # Step 2: Fix NumPy compatibility (critical!)
    numpy_ok = fix_numpy_compatibility()
    if not numpy_ok:
        print_status("⚠️  NumPy compatibility fix required. Please restart kernel and run again.", "WARNING")
        return
    
    # Step 3: Install dependencies
    install_dependencies()
    
    # Step 4: Setup directories
    if is_kaggle:
        setup_directories()
    
    # Step 5: Check GPU
    gpu_available = check_gpu()
    
    # Step 6: Set environment variables
    setup_environment_variables()
    
    print("=" * 80)
    print_status("🎉 Setup completed successfully!", "SUCCESS")
    
    # Provide next steps
    print("\n" + "=" * 80)
    print_status("📋 NEXT STEPS:", "INFO")
    print_status("1. Copy your project files to /kaggle/working/", "INFO")
    print_status("2. Run the training command:", "INFO")
    
    training_cmd = """
cd /kaggle/working/your-project-folder/math_pe_research
python scripts/train_and_eval.py \\
    --pe rope \\
    --batch_size 4 \\
    --experiment_name kaggle_run \\
    --checkpoint_dir /kaggle/working/checkpoints \\
    --result_dir /kaggle/working/evaluation_results \\
    --cache_dir /kaggle/working/data_cache \\
    --max_steps 500 \\
    --learning_rate 2e-5
"""
    
    print(training_cmd)
    
    print_status("3. Monitor training progress in W&B dashboard", "INFO")
    print_status("4. Results will be saved to /kaggle/working/evaluation_results/", "INFO")
    
    if not gpu_available:
        print_status("⚠️  NO GPU DETECTED - Training will be very slow!", "WARNING")
        print_status("   Make sure GPU is enabled in Kaggle notebook settings", "WARNING")

if __name__ == "__main__":
    main() 