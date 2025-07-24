#!/bin/bash
set -e

echo "🚀 Setting up Mathematical Reasoning Repository for Kaggle"
echo "========================================================="

# Check if we're in a Kaggle environment
if [ -d "/kaggle" ]; then
    echo "✅ Kaggle environment detected"
    KAGGLE_ENV=true
else
    echo "ℹ️  Running in local/other environment"
    KAGGLE_ENV=false
fi

# Install system dependencies
echo "📦 Installing system dependencies..."
if command -v apt &> /dev/null; then
    sudo apt update
    sudo apt install -y python3-venv python3-pip python3-dev
fi

# Create virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv kaggle_env
source kaggle_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Check GPU availability and install PyTorch accordingly
echo "🔍 Checking GPU availability..."
python3 -c "
import subprocess
import sys

# Check if nvidia-smi exists and can detect GPUs
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0 and 'Tesla' in result.stdout:
        print('GPU detected: Installing CUDA PyTorch')
        gpu_available = True
    else:
        print('No compatible GPU detected: Installing CPU PyTorch')
        gpu_available = False
except FileNotFoundError:
    print('nvidia-smi not found: Installing CPU PyTorch')
    gpu_available = False

# Install PyTorch based on GPU availability
if gpu_available:
    # For Kaggle T4 GPUs (CUDA 11.8)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118'], check=True)
    print('✅ GPU PyTorch installed')
else:
    # CPU-only version
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cpu'], check=True)
    print('✅ CPU PyTorch installed')
"

# Install other dependencies
echo "📚 Installing ML dependencies..."
pip install transformers accelerate datasets huggingface-hub
pip install wandb peft scipy matplotlib scikit-learn seaborn
pip install -r requirements_kaggle.txt

# Create necessary directories
echo "📁 Creating project structure..."
mkdir -p math_pe_research/src/utils math_pe_research/configs
mkdir -p checkpoints results cache

# Create symbolic links for compatibility
echo "🔗 Creating symbolic links..."
if [ ! -L "src" ]; then
    ln -sf math_pe_research/src src
fi
if [ ! -L "scripts" ]; then
    ln -sf math_pe_research/scripts scripts
fi
if [ ! -L "configs" ]; then
    ln -sf math_pe_research/configs configs
fi

# Test the installation
echo "🧪 Testing installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')
"

echo "🎯 Running comprehensive test..."
python3 kaggle_training_test.py

echo ""
echo "✅ Setup completed successfully!"
echo "========================================================="
echo "📋 Quick Start Commands:"
echo "  source kaggle_env/bin/activate"
echo "  python3 simple_simulation.py"
echo "  python3 kaggle_training_test.py"
echo ""
echo "🚀 For training:"
echo "  python3 train_and_eval.py --experiment_name 'my_experiment' \\"
echo "    --checkpoint_dir './checkpoints' --result_dir './results' \\"
echo "    --max_steps 100 --batch_size 2 --datasets 'gsm8k'"
echo ""
echo "🔧 Kaggle Environment: $KAGGLE_ENV"