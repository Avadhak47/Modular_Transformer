#!/usr/bin/env python3
"""
Setup script for Kaggle TPU environment
This script ensures proper PyTorch XLA installation for TPU training
"""

import os
import subprocess
import sys

def check_tpu_availability():
    # ""if TPU is available in the environment"" 
    return os.environ.get('TPU_NAME') is not None or os.environ.get('COLAB_TPU_ADDR') is not None

def install_torch_xla():
    # Install PyTorch XLA for TPU
    print("Installing PyTorch XLA for TPU...")
    
    # Get current PyTorch version
    try:
        import torch
        torch_version = torch.__version__
        print(f"Current PyTorch version: {torch_version}")
    except ImportError:
        print("PyTorch not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.0.1"])
        import torch
        torch_version = torch.__version__
    
    # Install PyTorch XLA
    try:
        # Try to install the compatible version
        cmd = [
            sys.executable, "-m", "pip", "install", "torch_xla[tpu]==2.0.1", "--index-url", "https://download.pytorch.org/whl/cpu"
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        print("PyTorch XLA installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install torch_xla: {e}")
        return False
    
    return True

def test_torch_xla():
    # Test if PyTorch XLA can be imported""
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        print("PyTorch XLA import successful!")
        
        # Test TPU device creation
        if check_tpu_availability():
            device = xm.xla_device()
            print(f"TPU device created: {device}")
        else:
            print("No TPU detected in environment")
        
        return True
    except Exception as e:
        print(f"PyTorch XLA test failed: {e}")
        return False

def main():
    print("=== Kaggle TPU Setup ===")
    
    # Check if we're in a TPU environment
    if check_tpu_availability():
        print("TPU environment detected!")
    else:
        print("No TPU environment detected. This script is for TPU setup.")
        return
    
    # Install PyTorch XLA
    if install_torch_xla():
        # Test the installation
        if test_torch_xla():
            print("✅ TPU setup completed successfully!")
            print("You can now run your TPU training script.")
        else:
            print("❌ TPU setup failed. Please check the error messages above.")
    else:
        print("❌ Failed to install PyTorch XLA")

if __name__ == "__main__":
    main() 