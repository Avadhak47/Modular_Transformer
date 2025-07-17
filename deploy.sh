#!/bin/bash
# deploy.sh - One-command deployment for IITD HPC Multi-Node Mathematical Reasoning
# Author: Avadhesh Kumar (2024EET2799)
# Usage: ./deploy.sh [full|containers-only|training-only] [experiment_name]

set -e  # Exit on any error

# Configuration
DEPLOYMENT_MODE="${1:-full}"
EXPERIMENT_NAME="${2:-mathematical_reasoning_$(date +%Y%m%d_%H%M%S)}"
BASE_DIR="/scratch/$USER/math_reasoning"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Banner
print_banner() {
    echo -e "${BLUE}"
    echo "████████████████████████████████████████████████████████████████████"
    echo "█                                                                  █"
    echo "█    IITD HPC Multi-Node Mathematical Reasoning Deployment         █"
    echo "█                                                                  █"
    echo "█    Experiment: $EXPERIMENT_NAME"
    echo "█    Mode: $DEPLOYMENT_MODE"
    echo "█    Base Dir: $BASE_DIR"
    echo "█                                                                  █"
    echo "████████████████████████████████████████████████████████████████████"
    echo -e "${NC}"
}

# Prerequisites check
check_prerequisites() {
    log "🔍 Checking prerequisites..."
    
    # Check HPC access
    if ! command -v qstat &> /dev/null; then
        error "PBS qstat command not found. Are you on the HPC cluster?"
    fi
    
    # Check GPU availability
    log "Checking GPU node availability..."
    AVAILABLE_GPUS=$(sinfo -p gpu --noheader --format="%D %T" | awk '$2=="idle" {print $1}' | head -1)
    if [ -z "$AVAILABLE_GPUS" ] || [ "$AVAILABLE_GPUS" -lt 5 ]; then
        warn "Less than 5 GPU nodes available. Current: $AVAILABLE_GPUS"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Check storage quota
    log "Checking storage quota..."
    QUOTA_INFO=$(lfs quota -hu $USER /scratch 2>/dev/null || echo "Cannot check quota")
    echo "Storage info: $QUOTA_INFO"
    
    # Create base directory structure
    log "Creating directory structure..."
    mkdir -p "$BASE_DIR"/{logs,results,configs,containers,models,scripts,monitoring}
    
    log "✅ Prerequisites check completed"
}

# Setup HPC environment
setup_hpc_environment() {
    log "🔧 Setting up HPC environment..."
    
    # Load required modules
    log "Loading HPC modules..."
    cat > "$BASE_DIR/scripts/load_modules.sh" << 'EOF'
#!/bin/bash
module purge
module load enroot/3.4.1
module load cuda/11.8
module load nccl/2.15.5
module load python/3.9
export ENROOT_CACHE_PATH=/scratch/$USER/.enroot_cache
export ENROOT_DATA_PATH=/scratch/$USER/.enroot_data
mkdir -p $ENROOT_CACHE_PATH $ENROOT_DATA_PATH
EOF
    chmod +x "$BASE_DIR/scripts/load_modules.sh"
    
    # Source modules
    source "$BASE_DIR/scripts/load_modules.sh"
    
    log "✅ HPC environment setup completed"
}

# Build and distribute containers
build_containers() {
    log "🐳 Building NVIDIA Enroot containers..."
    
    # Create enhanced Dockerfile for HPC
    cat > "$BASE_DIR/containers/math_reasoning.Dockerfile" << 'EOF'
# Multi-stage build for HPC optimization
FROM nvcr.io/nvidia/pytorch:24.01-py3 as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git build-essential cmake \
    libssl-dev libffi-dev \
    libnuma1 libnuma-dev \
    infiniband-diags \
    && rm -rf /var/lib/apt/lists/*

# Install HPC-specific Python packages
COPY requirements_hpc.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements_hpc.txt

# Install SOTA model dependencies with optimizations
RUN pip install --no-cache-dir \
    transformers[torch]==4.35.0 \
    accelerate==0.23.0 \
    deepspeed==0.10.3 \
    wandb==0.15.12 \
    flash-attn==2.3.6 \
    bitsandbytes==0.41.1 \
    peft==0.6.0 \
    datasets==2.14.0 \
    scipy==1.11.3 \
    sympy==1.12 \
    matplotlib==3.7.2 \
    seaborn==0.12.2

# Copy source code
WORKDIR /workspace
COPY . /workspace/

# Set environment variables for HPC
ENV PYTHONPATH="/workspace:$PYTHONPATH"
ENV CUDA_VISIBLE_DEVICES="0"
ENV NCCL_DEBUG=INFO
ENV NCCL_IB_DISABLE=0
ENV NCCL_NET_GDR_LEVEL=2
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8

# Create necessary directories
RUN mkdir -p /workspace/{checkpoints,results,logs,data,configs}

# Make scripts executable
RUN find /workspace/scripts -name "*.sh" -exec chmod +x {} \;

ENTRYPOINT ["python"]
EOF

    # Create HPC-specific requirements
    cat > "$BASE_DIR/requirements_hpc.txt" << 'EOF'
# Core PyTorch ecosystem (HPC optimized)
torch==2.1.0+cu118
torchvision==0.16.0+cu118
torchaudio==2.1.0+cu118

# Scientific computing (HPC versions)
numpy==1.24.3
scipy==1.11.3
pandas==2.0.3

# ML libraries (version locked for stability)
transformers==4.35.0
datasets==2.14.0
tokenizers==0.14.0
huggingface-hub==0.16.4
accelerate==0.23.0

# Training optimizations
deepspeed==0.10.3
flash-attn==2.3.6
bitsandbytes==0.41.1
peft==0.6.0

# SOTA model integrations
einops==0.7.0
rotary-embedding-torch==0.3.0

# Monitoring and logging
wandb==0.15.12
tensorboard==2.14.1
tqdm==4.66.1

# Mathematical verification
sympy==1.12
latex2sympy2==2.0.1

# Visualization
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.17.0

# Utilities
psutil==5.9.5
GPUtil==1.4.0
nvidia-ml-py3==11.525.112
editdistance==0.6.2

# Development
pytest==7.4.0
black==23.7.0
EOF

    # Build container locally if Docker available
    if command -v docker &> /dev/null; then
        log "Building Docker container locally..."
        cd "$SCRIPT_DIR"
        docker build -t math-reasoning:hpc -f "$BASE_DIR/containers/math_reasoning.Dockerfile" .
        docker save math-reasoning:hpc -o "$BASE_DIR/containers/math-reasoning-hpc.tar"
        log "Container saved to: $BASE_DIR/containers/math-reasoning-hpc.tar"
    else
        warn "Docker not available. Container will be built on HPC cluster."
    fi
    
    # Create container distribution script
    cat > "$BASE_DIR/scripts/build_and_distribute_containers.sh" << 'EOF'
#!/bin/bash
# build_and_distribute_containers.sh

BASE_DIR="/scratch/$USER/math_reasoning"
source "$BASE_DIR/scripts/load_modules.sh"

log() { echo "[$(date)] $1"; }

log "Building Enroot container on HPC..."

# Build from Dockerfile if tar doesn't exist
if [ ! -f "$BASE_DIR/containers/math-reasoning-hpc.tar" ]; then
    log "Building container from Dockerfile..."
    enroot import -o "$BASE_DIR/containers/math-reasoning-hpc.sqsh" \
        dockerd://nvcr.io/nvidia/pytorch:24.01-py3
else
    log "Importing container from tar..."
    enroot import -o "$BASE_DIR/containers/math-reasoning-hpc.sqsh" \
        "$BASE_DIR/containers/math-reasoning-hpc.tar"
fi

# Test container
log "Testing container..."
enroot start --rw "$BASE_DIR/containers/math-reasoning-hpc.sqsh" \
    -- python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

if [ $? -eq 0 ]; then
    log "✅ Container build and test successful"
else
    log "❌ Container build failed"
    exit 1
fi

log "Container ready: $BASE_DIR/containers/math-reasoning-hpc.sqsh"
EOF
    chmod +x "$BASE_DIR/scripts/build_and_distribute_containers.sh"
    
    # Run container build
    "$BASE_DIR/scripts/build_and_distribute_containers.sh"
    
    log "✅ Container build completed"
}

# Download SOTA models
download_sota_models() {
    log "🧠 Setting up SOTA model access..."
    
    # Create model download script
    cat > "$BASE_DIR/scripts/download_sota_models.sh" << 'EOF'
#!/bin/bash
# download_sota_models.sh

BASE_DIR="/scratch/$USER/math_reasoning"
MODELS_DIR="$BASE_DIR/models"

log() { echo "[$(date)] $1"; }

# Create models directory
mkdir -p "$MODELS_DIR"

# Check HuggingFace authentication
if [ -z "$HF_TOKEN" ]; then
    log "WARNING: HF_TOKEN not set. Some models may not be accessible."
    log "Set HF_TOKEN environment variable or login with 'huggingface-cli login'"
fi

# Download model configurations and tokenizers (not full models to save space)
log "Downloading model configurations..."

models=(
    "deepseek-ai/deepseek-math-7b-base"
    "microsoft/Orca-Math-7B-Preview"
    "internlm/internlm-math-7b"
    "meta-llama/Llama-2-7b-hf"
    "mistralai/Mistral-7B-v0.1"
)

for model in "${models[@]}"; do
    log "Downloading config for $model..."
    model_dir="$MODELS_DIR/$(basename $model)"
    mkdir -p "$model_dir"
    
    # Download only config and tokenizer files
    python3 -c "
from transformers import AutoConfig, AutoTokenizer
import os
model_name = '$model'
save_dir = '$model_dir'
try:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.save_pretrained(save_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(save_dir)
    print(f'Downloaded config and tokenizer for {model_name}')
except Exception as e:
    print(f'Failed to download {model_name}: {e}')
"
done

log "Model configurations downloaded to: $MODELS_DIR"
log "Full models will be downloaded during training as needed."
EOF
    chmod +x "$BASE_DIR/scripts/download_sota_models.sh"
    
    # Run model download
    "$BASE_DIR/scripts/download_sota_models.sh"
    
    log "✅ SOTA model setup completed"
}

# Generate node-specific configurations
generate_node_configs() {
    log "⚙️ Generating node-specific configurations..."
    
    # Create configuration generator
    cat > "$BASE_DIR/scripts/generate_node_configs.py" << 'EOF'
#!/usr/bin/env python3
"""Generate node-specific configurations for multi-node training."""

import json
import os
from pathlib import Path

BASE_DIR = Path(f"/scratch/{os.environ['USER']}/math_reasoning")
CONFIGS_DIR = BASE_DIR / "configs" / "node_configs"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

# Node configuration mapping
NODE_CONFIGS = {
    0: {
        'pe_type': 'sinusoidal',
        'sota_method': 'deepseekmath',
        'base_model': 'deepseek-ai/deepseek-math-7b-base',
        'specialization': 'continued_pretraining'
    },
    1: {
        'pe_type': 'rope',
        'sota_method': 'internlm_math',
        'base_model': 'internlm/internlm-math-7b',
        'specialization': 'verifiable_reasoning'
    },
    2: {
        'pe_type': 'alibi',
        'sota_method': 'orca_math',
        'base_model': 'microsoft/Orca-Math-7B-Preview',
        'specialization': 'multi_agent_learning'
    },
    3: {
        'pe_type': 'diet',
        'sota_method': 'dotamath',
        'base_model': 'meta-llama/Llama-2-7b-hf',
        'specialization': 'decomposition_reasoning'
    },
    4: {
        'pe_type': 't5_relative',
        'sota_method': 'mindstar',
        'base_model': 'mistralai/Mistral-7B-v0.1',
        'specialization': 'inference_optimization'
    }
}

def generate_node_config(node_id: int) -> dict:
    """Generate configuration for specific node."""
    node_config = NODE_CONFIGS[node_id]
    
    config = {
        "node_id": node_id,
        "experiment_name": f"node_{node_id}_{node_config['pe_type']}_{node_config['sota_method']}",
        
        "model_config": {
            "base_model": node_config['base_model'],
            "d_model": 4096,
            "n_heads": 32,
            "n_encoder_layers": 32,
            "n_decoder_layers": 32,
            "d_ff": 11008,
            "vocab_size": 50000,
            "max_seq_len": 4096,
            "dropout": 0.1,
            "positional_encoding": node_config['pe_type'],
            "use_flash_attention": True,
            "gradient_checkpointing": True
        },
        
        "sota_integration": {
            "method": node_config['sota_method'],
            "specialization": node_config['specialization'],
            "use_lora": True,
            "lora_rank": 64,
            "lora_alpha": 128,
            "use_4bit_quantization": True,
            "use_grpo": node_config['sota_method'] == 'deepseekmath',
            "use_chain_of_thought": True,
            "use_self_correction": True,
            "use_code_assistance": node_config['sota_method'] in ['dotamath', 'internlm_math']
        },
        
        "training_config": {
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-5,
            "warmup_steps": 1000,
            "max_steps": 50000,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "fp16": False,
            "bf16": True,
            "dataloader_num_workers": 4,
            "save_steps": 2000,
            "eval_steps": 1000,
            "logging_steps": 100
        },
        
        "data_config": {
            "train_datasets": ["math", "gsm8k", node_config['sota_method']],
            "eval_datasets": ["math_test", "gsm8k_test"],
            "max_train_samples": 50000,
            "max_eval_samples": 2000,
            "preprocessing": {
                "chain_of_thought": True,
                "max_reasoning_steps": 10,
                "augment_with_code": node_config['sota_method'] in ['dotamath', 'internlm_math'],
                "multi_agent_generation": node_config['sota_method'] == 'orca_math'
            }
        },
        
        "evaluation_config": {
            "metrics": [
                "exact_match_accuracy",
                "numerical_accuracy", 
                "reasoning_step_accuracy",
                "mathematical_correctness",
                "perplexity",
                "attention_entropy",
                "inference_time"
            ],
            "benchmarks": ["math", "gsm8k", "aime"],
            "generate_explanations": True,
            "verify_with_sympy": True
        },
        
        "distributed_config": {
            "backend": "nccl",
            "init_method": "env://",
            "world_size": 5,
            "rank": node_id,
            "local_rank": 0,
            "master_port": 29500
        },
        
        "monitoring_config": {
            "use_wandb": True,
            "wandb_project": "mathematical_reasoning_transformers",
            "wandb_entity": "iitd_hpc_research",
            "track_gpu_memory": True,
            "track_attention_patterns": True,
            "save_intermediate_outputs": True
        },
        
        "output_config": {
            "output_dir": f"/scratch/{os.environ['USER']}/math_reasoning/results/node_{node_id}_{node_config['pe_type']}_{node_config['sota_method']}",
            "checkpoint_dir": f"/scratch/{os.environ['USER']}/math_reasoning/checkpoints/node_{node_id}",
            "log_dir": f"/scratch/{os.environ['USER']}/math_reasoning/logs/node_{node_id}",
            "save_total_limit": 3,
            "load_best_model_at_end": True
        }
    }
    
    return config

def main():
    """Generate all node configurations."""
    print("Generating node-specific configurations...")
    
    for node_id in range(5):
        config = generate_node_config(node_id)
        config_file = CONFIGS_DIR / f"node_{node_id}_config.json"
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Generated config for Node {node_id}: {config_file}")
        
        # Create output directories
        for dir_key in ['output_dir', 'checkpoint_dir', 'log_dir']:
            Path(config['output_config'][dir_key]).mkdir(parents=True, exist_ok=True)
    
    print("✅ All node configurations generated successfully!")

if __name__ == "__main__":
    main()
EOF
    chmod +x "$BASE_DIR/scripts/generate_node_configs.py"
    
    # Generate configurations
    python3 "$BASE_DIR/scripts/generate_node_configs.py"
    
    log "✅ Node configurations generated"
}

# Create monitoring dashboard
setup_monitoring() {
    log "📊 Setting up monitoring dashboard..."
    
    # Create monitoring script
    cat > "$BASE_DIR/scripts/monitor_multi_node_dashboard.py" << 'EOF'
#!/usr/bin/env python3
"""Real-time monitoring dashboard for multi-node training."""

import os
import time
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class MultiNodeMonitor:
    """Multi-node training monitor with dashboard."""
    
    def __init__(self, experiment_name: str, base_dir: str = None):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir or f"/scratch/{os.environ['USER']}/math_reasoning")
        self.results_dir = self.base_dir / "results"
        self.logs_dir = self.base_dir / "logs"
        self.dashboard_dir = self.base_dir / "monitoring" / experiment_name
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        
    def get_job_status(self):
        """Get PBS job status."""
        try:
            result = subprocess.run(['qstat', '-u', os.environ['USER']], 
                                  capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            
            jobs = []
            for line in lines[2:]:  # Skip header
                if 'math_reasoning' in line:
                    parts = line.split()
                    jobs.append({
                        'job_id': parts[0],
                        'name': parts[1],
                        'user': parts[2], 
                        'status': parts[9],
                        'queue': parts[7]
                    })
            return jobs
        except Exception as e:
            return [{"error": str(e)}]
    
    def get_node_status(self):
        """Get status of all training nodes."""
        node_status = {}
        
        for node_id in range(5):
            node_dir = list(self.results_dir.glob(f"node_{node_id}_*"))
            if node_dir:
                node_dir = node_dir[0]
                status = self.get_single_node_status(node_dir, node_id)
                node_status[f"node_{node_id}"] = status
            else:
                node_status[f"node_{node_id}"] = {"status": "not_started"}
                
        return node_status
    
    def get_single_node_status(self, node_dir: Path, node_id: int):
        """Get status of single node."""
        try:
            # Check for trainer state
            trainer_state_files = list(node_dir.glob("**/trainer_state.json"))
            if trainer_state_files:
                latest_state = max(trainer_state_files, key=lambda x: x.stat().st_mtime)
                with open(latest_state) as f:
                    state = json.load(f)
                
                progress = state.get('global_step', 0) / state.get('max_steps', 1) * 100
                current_loss = None
                if state.get('log_history'):
                    latest_log = state['log_history'][-1]
                    current_loss = latest_log.get('train_loss', latest_log.get('eval_loss'))
                
                return {
                    "status": "training",
                    "progress": f"{progress:.1f}%",
                    "step": state.get('global_step', 0),
                    "max_steps": state.get('max_steps', 0),
                    "loss": current_loss,
                    "epoch": state.get('epoch', 0),
                    "last_update": datetime.fromtimestamp(latest_state.stat().st_mtime).isoformat()
                }
            
            # Check for log files
            log_files = list(self.logs_dir.glob(f"node_{node_id}*.log"))
            if log_files:
                latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                return {
                    "status": "running",
                    "last_log": latest_log.name,
                    "last_update": datetime.fromtimestamp(latest_log.stat().st_mtime).isoformat()
                }
            
            return {"status": "unknown"}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_gpu_utilization(self):
        """Get GPU utilization across cluster."""
        # This would require SSH access to nodes or cluster monitoring
        # For now, return placeholder
        return {
            f"node_{i}": {
                "gpu_utilization": "N/A",
                "memory_used": "N/A", 
                "memory_total": "N/A"
            } for i in range(5)
        }
    
    def generate_dashboard_html(self):
        """Generate HTML dashboard."""
        job_status = self.get_job_status()
        node_status = self.get_node_status()
        gpu_status = self.get_gpu_utilization()
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multi-Node Training Dashboard - {self.experiment_name}</title>
            <meta http-equiv="refresh" content="30">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #2E86AB; color: white; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .node-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }}
                .node-card {{ border: 1px solid #ccc; padding: 15px; border-radius: 5px; }}
                .status-training {{ background-color: #d4edda; }}
                .status-not-started {{ background-color: #f8d7da; }}
                .status-completed {{ background-color: #d1ecf1; }}
                .progress-bar {{ width: 100%; height: 20px; background: #f0f0f0; border-radius: 10px; }}
                .progress-fill {{ height: 100%; background: #28a745; border-radius: 10px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Multi-Node Training Dashboard</h1>
                <p>Experiment: {self.experiment_name}</p>
                <p>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📋 Job Status</h2>
                <table>
                    <tr><th>Job ID</th><th>Name</th><th>Status</th><th>Queue</th></tr>
        """
        
        for job in job_status:
            if 'error' not in job:
                html_template += f"<tr><td>{job.get('job_id', 'N/A')}</td><td>{job.get('name', 'N/A')}</td><td>{job.get('status', 'N/A')}</td><td>{job.get('queue', 'N/A')}</td></tr>"
        
        html_template += """
                </table>
            </div>
            
            <div class="section">
                <h2>🖥️ Node Status</h2>
                <div class="node-grid">
        """
        
        for node_name, status in node_status.items():
            status_class = f"status-{status.get('status', 'unknown')}"
            progress = 0
            if 'progress' in status:
                progress = float(status['progress'].replace('%', ''))
            
            html_template += f"""
                    <div class="node-card {status_class}">
                        <h3>{node_name.upper()}</h3>
                        <p><strong>Status:</strong> {status.get('status', 'Unknown')}</p>
            """
            
            if 'progress' in status:
                html_template += f"""
                        <p><strong>Progress:</strong> {status['progress']}</p>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {progress}%"></div>
                        </div>
                        <p><strong>Step:</strong> {status.get('step', 0)} / {status.get('max_steps', 0)}</p>
                """
            
            if 'loss' in status and status['loss']:
                html_template += f"<p><strong>Loss:</strong> {status['loss']:.4f}</p>"
            
            if 'last_update' in status:
                html_template += f"<p><strong>Last Update:</strong> {status['last_update']}</p>"
            
            html_template += "</div>"
        
        html_template += """
                </div>
            </div>
        </body>
        </html>
        """
        
        dashboard_file = self.dashboard_dir / "dashboard.html"
        with open(dashboard_file, 'w') as f:
            f.write(html_template)
        
        return dashboard_file
    
    def print_status(self):
        """Print status to console."""
        print(f"\n{'='*80}")
        print(f"🚀 Multi-Node Training Dashboard - {self.experiment_name}")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # Job status
        print(f"\n📋 Job Status:")
        job_status = self.get_job_status()
        for job in job_status:
            if 'error' not in job:
                print(f"  Job {job.get('job_id', 'N/A')}: {job.get('status', 'N/A')} ({job.get('queue', 'N/A')})")
        
        # Node status
        print(f"\n🖥️ Node Status:")
        node_status = self.get_node_status()
        for node_name, status in node_status.items():
            status_emoji = {"training": "🟢", "not_started": "🔴", "completed": "🔵"}.get(status.get('status'), "⚪")
            print(f"  {status_emoji} {node_name}: {status.get('status', 'Unknown')}")
            if 'progress' in status:
                print(f"    Progress: {status['progress']} (Step {status.get('step', 0)}/{status.get('max_steps', 0)})")
            if 'loss' in status and status['loss']:
                print(f"    Loss: {status['loss']:.4f}")
    
    def run_dashboard(self, interval: int = 30):
        """Run dashboard with auto-refresh."""
        print(f"Starting dashboard for experiment: {self.experiment_name}")
        print(f"Refresh interval: {interval} seconds")
        print(f"Dashboard files will be saved to: {self.dashboard_dir}")
        
        try:
            while True:
                self.print_status()
                dashboard_file = self.generate_dashboard_html()
                print(f"\n📊 Dashboard updated: {dashboard_file}")
                print(f"View in browser: file://{dashboard_file.absolute()}")
                
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped by user")

def main():
    parser = argparse.ArgumentParser(description="Multi-node training dashboard")
    parser.add_argument("--experiment", required=True, help="Experiment name")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval (seconds)")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--base-dir", help="Base directory")
    
    args = parser.parse_args()
    
    monitor = MultiNodeMonitor(args.experiment, args.base_dir)
    
    if args.once:
        monitor.print_status()
        dashboard_file = monitor.generate_dashboard_html()
        print(f"Dashboard generated: {dashboard_file}")
    else:
        monitor.run_dashboard(args.interval)

if __name__ == "__main__":
    main()
EOF
    chmod +x "$BASE_DIR/scripts/monitor_multi_node_dashboard.py"
    
    log "✅ Monitoring setup completed"
}

# Submit multi-node training
submit_training() {
    log "🎯 Submitting multi-node training jobs..."
    
    # Create multi-node submission script
    cat > "$BASE_DIR/scripts/submit_multi_node_training.sh" << 'EOF'
#!/bin/bash
#PBS -N math_reasoning_multi_node
#PBS -q gpuq
#PBS -l select=5:ncpus=8:ngpus=1:mem=128GB
#PBS -l walltime=48:00:00
#PBS -o /scratch/$USER/math_reasoning/logs/multi_node_${PBS_JOBID}.out
#PBS -e /scratch/$USER/math_reasoning/logs/multi_node_${PBS_JOBID}.err
#PBS -j oe

# Set up environment
BASE_DIR="/scratch/$USER/math_reasoning"
cd $BASE_DIR

# Load modules
source "$BASE_DIR/scripts/load_modules.sh"

# Set distributed training environment
export MASTER_ADDR=$(head -1 $PBS_NODEFILE)
export MASTER_PORT=29500
export WORLD_SIZE=5
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO

echo "Starting multi-node training..."
echo "Master node: $MASTER_ADDR"
echo "World size: $WORLD_SIZE"
echo "Job ID: $PBS_JOBID"

# Create job log
echo "$(date): Multi-node training started" >> "$BASE_DIR/logs/job_${PBS_JOBID}.log"

# Launch training on each node
declare -a PE_METHODS=("sinusoidal" "rope" "alibi" "diet" "t5_relative")
declare -a SOTA_MODELS=("deepseekmath" "internlm_math" "orca_math" "dotamath" "mindstar")

for i in {0..4}; do
    NODE_RANK=$i
    PE_METHOD=${PE_METHODS[$i]}
    SOTA_MODEL=${SOTA_MODELS[$i]}
    
    # Get node hostname
    NODE_HOST=$(sed -n "$((i+1))p" $PBS_NODEFILE)
    
    echo "Launching training on node $NODE_HOST (rank $NODE_RANK) with $PE_METHOD + $SOTA_MODEL"
    
    # Launch training on specific node
    ssh $NODE_HOST "cd $BASE_DIR && \
        ./scripts/node_training_launcher.sh \
        --node_rank=$NODE_RANK \
        --pe_method=$PE_METHOD \
        --sota_model=$SOTA_MODEL \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --world_size=$WORLD_SIZE \
        --job_id=$PBS_JOBID" &
done

echo "All nodes launched. Waiting for completion..."

# Wait for all background jobs
wait

echo "$(date): Multi-node training completed" >> "$BASE_DIR/logs/job_${PBS_JOBID}.log"
echo "Multi-node training completed!"

# Run final aggregation
python3 "$BASE_DIR/scripts/aggregate_results.py" --job_id="$PBS_JOBID"
EOF
    chmod +x "$BASE_DIR/scripts/submit_multi_node_training.sh"
    
    # Create node training launcher
    cat > "$BASE_DIR/scripts/node_training_launcher.sh" << 'EOF'
#!/bin/bash
# node_training_launcher.sh - Launch training on individual node

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --node_rank=*) NODE_RANK="${1#*=}"; shift ;;
        --pe_method=*) PE_METHOD="${1#*=}"; shift ;;
        --sota_model=*) SOTA_MODEL="${1#*=}"; shift ;;
        --master_addr=*) MASTER_ADDR="${1#*=}"; shift ;;
        --master_port=*) MASTER_PORT="${1#*=}"; shift ;;
        --world_size=*) WORLD_SIZE="${1#*=}"; shift ;;
        --job_id=*) JOB_ID="${1#*=}"; shift ;;
        *) shift ;;
    esac
done

# Setup environment
BASE_DIR="/scratch/$USER/math_reasoning"
source "$BASE_DIR/scripts/load_modules.sh"

# Set environment variables for this node
export RANK=$NODE_RANK
export LOCAL_RANK=0
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$WORLD_SIZE
export CUDA_VISIBLE_DEVICES=0

# Create output directory
OUTPUT_DIR="$BASE_DIR/results/node_${NODE_RANK}_${PE_METHOD}_${SOTA_MODEL}"
mkdir -p "$OUTPUT_DIR"

# Log file for this node
LOG_FILE="$BASE_DIR/logs/node_${NODE_RANK}_${PE_METHOD}_${SOTA_MODEL}.log"

echo "$(date): Starting node $NODE_RANK training with $PE_METHOD + $SOTA_MODEL" >> "$LOG_FILE"

# Launch Enroot container with training
enroot start \
    --mount "$BASE_DIR:/workspace" \
    --mount "/home/$USER/.cache/huggingface:/root/.cache/huggingface" \
    --env RANK=$RANK \
    --env LOCAL_RANK=$LOCAL_RANK \
    --env MASTER_ADDR=$MASTER_ADDR \
    --env MASTER_PORT=$MASTER_PORT \
    --env WORLD_SIZE=$WORLD_SIZE \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    "$BASE_DIR/containers/math-reasoning-hpc.sqsh" \
    /workspace/training/sota_mathematical_reasoning_trainer.py \
    --config "/workspace/configs/node_configs/node_${NODE_RANK}_config.json" \
    --output_dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"

echo "$(date): Node $NODE_RANK training completed" >> "$LOG_FILE"
EOF
    chmod +x "$BASE_DIR/scripts/node_training_launcher.sh"
    
    if [ "$DEPLOYMENT_MODE" != "containers-only" ]; then
        # Submit the job
        JOB_ID=$(qsub "$BASE_DIR/scripts/submit_multi_node_training.sh")
        log "✅ Multi-node training job submitted: $JOB_ID"
        
        # Save job ID for monitoring
        echo "$JOB_ID" > "$BASE_DIR/logs/current_job_id.txt"
        
        log "Monitor progress with:"
        log "  python3 $BASE_DIR/scripts/monitor_multi_node_dashboard.py --experiment $EXPERIMENT_NAME"
    fi
}

# Main deployment function
main() {
    print_banner
    
    case $DEPLOYMENT_MODE in
        "full")
            check_prerequisites
            setup_hpc_environment
            build_containers
            download_sota_models
            generate_node_configs
            setup_monitoring
            submit_training
            ;;
        "containers-only")
            check_prerequisites
            setup_hpc_environment
            build_containers
            ;;
        "training-only")
            check_prerequisites
            generate_node_configs
            setup_monitoring
            submit_training
            ;;
        *)
            error "Invalid deployment mode: $DEPLOYMENT_MODE. Use: full, containers-only, or training-only"
            ;;
    esac
    
    log "🎉 Deployment completed successfully!"
    log ""
    log "Next steps:"
    log "  1. Monitor training: python3 $BASE_DIR/scripts/monitor_multi_node_dashboard.py --experiment $EXPERIMENT_NAME"
    log "  2. Check logs: tail -f $BASE_DIR/logs/*.log"
    log "  3. View results: ls -la $BASE_DIR/results/"
    log ""
    log "Experiment: $EXPERIMENT_NAME"
    log "Base directory: $BASE_DIR"
}

# Run main function
main "$@"