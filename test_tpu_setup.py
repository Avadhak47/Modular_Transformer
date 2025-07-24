#!/usr/bin/env python3

import os
import sys
import torch

def check_environment():
    print("=== Environment Check ===")
    print(f"Python version: {sys.version}")
    print(f"TPU_NAME: {os.environ.get('TPU_NAME', 'Not set')}")
    print(f"COLAB_TPU_ADDR: {os.environ.get('COLAB_TPU_ADDR', 'Not set')}")
    
    # Check if we're on Kaggle
    if os.path.exists('/kaggle/input'):
        print("✅ Running on Kaggle")
    else:
        print("❌ Not running on Kaggle")

def test_pytorch():
    print("\n=== PyTorch Check ===")
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
        return True
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False

def test_torch_xla():
    print("\n=== PyTorch XLA Check ===")
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        print(f"✅ PyTorch XLA version: {torch_xla.__version__}")
        
        # Test TPU device creation
        try:
            device = xm.xla_device()
            print(f"✅ TPU device created: {device}")
            
            # Test basic operation
            x = torch.randn(2, 2).to(device)
            y = x + x
            print(f"✅ TPU tensor operation successful")
            return True
        except Exception as e:
            print(f"❌ TPU device creation failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ PyTorch XLA import failed: {e}")
        print("\n=== Installation Instructions ===\n")
        print("To fix this, run the following in your Kaggle notebook:\n")
        print("!pip uninstall torch torch_xla -y\n")
        print("!pip install torch==201 torch_xla[tpu]==2.0.1 --index-url https://download.pytorch.org/whl/cpu")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    print("Kaggle TPU Setup Test\n" + "="*30)
    check_environment()
    pytorch_ok = test_pytorch()
    tpu_ok = test_torch_xla()
    
    print("\n=== Summary ===\n")
    if pytorch_ok and tpu_ok:
        print("✅ Everything is working! You can run TPU training.")
    elif pytorch_ok and not tpu_ok:
        print("⚠️ PyTorch works but TPU needs setup. Follow the installation instructions above.")
    else:
        print("❌ Setup issues detected. Check the error messages above.")

if __name__ == "__main__":
    main() 