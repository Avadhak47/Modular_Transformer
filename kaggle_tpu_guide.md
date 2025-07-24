# Kaggle TPU Setup Guide

## The Problem
The PyTorch XLA import error you're seeing is due to version incompatibility between PyTorch and PyTorch XLA on Kaggle.

## Solution

### Step 1: Use Kaggle's TPU Runtime
Make sure youre using a Kaggle notebook with TPU runtime enabled:
- Go to your Kaggle notebook
- Click on "Accelerator" in the right panel
- Select TPU instead ofGPU" or None"

### Step2Install Compatible PyTorch XLA
Run this in your Kaggle notebook cell:

```python
# Uninstall existing torch and torch_xla
!pip uninstall torch torch_xla -y

# Install TPU-compatible versions
!pip install torch==201 torch_xla[tpu]==2.0.1 --index-url https://download.pytorch.org/whl/cpu
```

### Step 3Test TPU Setup
Run this to verify TPU is working:

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm

print(f"PyTorch version: {torch.__version__})
print(f"PyTorch XLA version: {torch_xla.__version__}")

# Test TPU device
device = xm.xla_device()
print(f"TPU device: {device}")

# Test basic tensor operations
x = torch.randn(22).to(device)
y = x + x
print(f"TPU tensor operation successful: {y})
```

### Step 4: Run Your Training Script
Once TPU is working, you can run your training script:

```python
!python tpu_train.py --pe_type sinusoidal --epochs5--batch_size 32``

## Alternative: Use GPU Instead
If TPU continues to have issues, you can use GPU training instead:

```python
!python kaggle_train.py --pe_type sinusoidal --epochs5--batch_size 32ndb
```

## Common Issues and Solutions
1 **Import Error**: Make sure you're using the TPU runtime and have installed the correct versions
2**Memory Issues**: Reduce batch size for TPU training3**Version Mismatch**: Always use compatible PyTorch and PyTorch XLA versions

## Recommended TPU Configuration
- PyTorch:2.0.1
- PyTorch XLA: 2.01
- Batch size: 16depending on model size)
- Use gradient accumulation for larger effective batch sizes 