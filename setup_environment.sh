#!/bin/bash
# Environment Setup Script for Mathematical Reasoning HPC Deployment
# Author: Research Team, IIT Delhi

set -e

echo "🚀 Setting up Mathematical Reasoning Environment on HPC"
echo "======================================================="

# Configuration
PROJECT_ROOT="/scratch/$USER/math_reasoning"
VENV_PATH="$PROJECT_ROOT/venv"

# Create project directory
echo "📁 Creating project directories..."
mkdir -p $PROJECT_ROOT/{data,logs,results,configs}
cd $PROJECT_ROOT

# Copy project files
echo "📋 Copying project files..."
if [[ -d "/workspace" ]]; then
    cp -r /workspace/* .
    echo "✅ Project files copied from /workspace"
else
    echo "⚠️ Warning: /workspace not found. Please copy project files manually to $PROJECT_ROOT"
fi

# Setup Python virtual environment
echo "🐍 Setting up Python virtual environment..."
module load python/3.9.0

if [[ ! -d "$VENV_PATH" ]]; then
    python3 -m venv $VENV_PATH
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source $VENV_PATH/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate datasets
pip install peft bitsandbytes
pip install wandb
pip install sympy numpy pandas matplotlib seaborn
pip install tqdm einops safetensors
pip install omegaconf hydra-core
pip install psutil

echo "✅ Dependencies installed"

# Setup Weights & Biases
echo "📊 Setting up Weights & Biases..."
if [[ -z "$WANDB_API_KEY" ]]; then
    echo "⚠️ Warning: WANDB_API_KEY not set. Please run: wandb login"
else
    wandb login $WANDB_API_KEY
    echo "✅ Weights & Biases configured"
fi

# Download datasets (cache them)
echo "📚 Pre-downloading datasets..."
python3 -c "
from datasets import load_dataset
import os

cache_dir = '$PROJECT_ROOT/data_cache'
os.makedirs(cache_dir, exist_ok=True)

print('Downloading MATH dataset...')
try:
    dataset = load_dataset('hendrycks/competition_math', cache_dir=cache_dir)
    print('✅ MATH dataset cached')
except Exception as e:
    print(f'⚠️ MATH dataset failed: {e}')

print('Downloading GSM8K dataset...')
try:
    dataset = load_dataset('gsm8k', 'main', cache_dir=cache_dir)
    print('✅ GSM8K dataset cached')
except Exception as e:
    print(f'⚠️ GSM8K dataset failed: {e}')

print('Dataset caching completed.')
"

# Set up model cache directory
echo "🤖 Setting up model cache..."
export HF_HOME="$PROJECT_ROOT/model_cache"
mkdir -p $HF_HOME

# Test imports
echo "🧪 Testing imports..."
python3 -c "
import torch
import transformers
import datasets
print(f'✅ PyTorch {torch.__version__}')
print(f'✅ Transformers {transformers.__version__}')
print(f'✅ Datasets {datasets.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA devices: {torch.cuda.device_count()}')
"

# Create launch script
echo "📜 Creating launch script..."
cat > launch_experiment.sh << 'EOF'
#!/bin/bash
# Quick launch script for the experiment

cd /scratch/$USER/math_reasoning
source venv/bin/activate

echo "Launching Mathematical Reasoning Experiment..."
echo "Submit job with: qsub scripts/submit_hpc_job.sh"
echo "Monitor with: qstat -u $USER"

# Submit the job
qsub scripts/submit_hpc_job.sh
EOF

chmod +x launch_experiment.sh

# Create monitoring script
echo "📊 Creating monitoring script..."
cat > monitor_experiment.sh << 'EOF'
#!/bin/bash
# Monitor experiment progress

PROJECT_ROOT="/scratch/$USER/math_reasoning"
LOG_DIR="$PROJECT_ROOT/logs"

echo "📊 Mathematical Reasoning Experiment Monitor"
echo "============================================"

# Check job status
echo "📋 Job Status:"
qstat -u $USER

echo -e "\n📁 Log Files:"
if [[ -d "$LOG_DIR" ]]; then
    ls -la $LOG_DIR/
else
    echo "Log directory not found: $LOG_DIR"
fi

echo -e "\n🏃 Training Progress:"
for i in {0..4}; do
    log_file="$LOG_DIR/node_${i}_training.log"
    if [[ -f "$log_file" ]]; then
        echo "Node $i: $(tail -n 1 $log_file 2>/dev/null || echo 'No recent activity')"
    else
        echo "Node $i: Log file not found"
    fi
done

echo -e "\n💾 Disk Usage:"
du -sh $PROJECT_ROOT/* 2>/dev/null || echo "Unable to check disk usage"

echo -e "\n🔍 GPU Usage (if available):"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
EOF

chmod +x monitor_experiment.sh

# Create cleanup script
echo "🧹 Creating cleanup script..."
cat > cleanup_experiment.sh << 'EOF'
#!/bin/bash
# Cleanup experiment files

PROJECT_ROOT="/scratch/$USER/math_reasoning"

echo "🧹 Cleaning up experiment files..."
echo "⚠️ This will remove temporary files and caches"
read -p "Continue? (y/N): " confirm

if [[ $confirm == [yY] ]]; then
    echo "Cleaning temporary files..."
    rm -rf $PROJECT_ROOT/data_cache
    rm -rf $PROJECT_ROOT/model_cache
    rm -f $PROJECT_ROOT/logs/*.pid
    echo "✅ Cleanup completed"
else
    echo "Cleanup cancelled"
fi
EOF

chmod +x cleanup_experiment.sh

# Final checks
echo "🔍 Running final checks..."

# Check disk space
df -h /scratch/$USER/ | head -2

# Check permissions
if [[ -w "$PROJECT_ROOT" ]]; then
    echo "✅ Write permissions confirmed"
else
    echo "❌ No write permissions to $PROJECT_ROOT"
    exit 1
fi

# Summary
echo ""
echo "🎉 Environment setup completed successfully!"
echo "======================================================="
echo "📍 Project location: $PROJECT_ROOT"
echo "🐍 Virtual environment: $VENV_PATH"
echo "🚀 Launch experiment: ./launch_experiment.sh"
echo "📊 Monitor progress: ./monitor_experiment.sh"
echo "🧹 Cleanup: ./cleanup_experiment.sh"
echo ""
echo "📝 Next steps:"
echo "1. Verify all dependencies are working: source venv/bin/activate && python3 -c 'import torch; print(torch.cuda.is_available())'"
echo "2. Submit the job: qsub scripts/submit_hpc_job.sh"
echo "3. Monitor progress: watch -n 30 './monitor_experiment.sh'"
echo ""
echo "✅ Ready for HPC deployment!"