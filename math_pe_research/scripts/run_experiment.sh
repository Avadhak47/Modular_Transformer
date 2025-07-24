#!/bin/bash
# run_experiment.sh - Main experiment automation for positional encoding research
# Usage: ./run_experiment.sh --pe_method rope --node_id 0 --experiment_name my_experiment

set -e

# Default parameters
PE_METHOD="rope"
NODE_ID=0
EXPERIMENT_NAME="pe_comparison_$(date +%Y%m%d_%H%M%S)"
BASE_MODEL="deepseek-ai/deepseek-math-7b-instruct"
DATASETS="math,gsm8k,openmath_instruct"
MAX_STEPS=5000
BATCH_SIZE=4
LEARNING_RATE=2e-5
SAVE_STEPS=500
EVAL_STEPS=250
MAX_LENGTH=4096
USE_LORA=true
LOAD_IN_4BIT=true
OUTPUT_DIR="/scratch/$USER/math_pe_research"
WANDB_PROJECT="math_pe_research"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pe_method)
            PE_METHOD="$2"
            shift 2
            ;;
        --node_id)
            NODE_ID="$2"
            shift 2
            ;;
        --experiment_name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --base_model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --pe_method       Positional encoding method (rope, alibi, sinusoidal, diet, t5_relative, math_adaptive)"
            echo "  --node_id         Node ID for this experiment (0-4)"
            echo "  --experiment_name Name for this experiment"
            echo "  --base_model      Base model to use"
            echo "  --datasets        Comma-separated list of datasets"
            echo "  --max_steps       Maximum training steps"
            echo "  --batch_size      Training batch size"
            echo "  --learning_rate   Learning rate"
            echo "  --output_dir      Output directory"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "████████████████████████████████████████████████████████████████"
    echo "█                                                              █"
    echo "█    Mathematical Reasoning PE Research Experiment             █"
    echo "█                                                              █"
    echo "█    PE Method: $PE_METHOD"
    echo "█    Node ID: $NODE_ID"
    echo "█    Experiment: $EXPERIMENT_NAME"
    echo "█    Base Model: $BASE_MODEL"
    echo "█                                                              █"
    echo "████████████████████████████████████████████████████████████████"
    echo -e "${NC}"
}

print_banner

# Validate PE method
VALID_PE_METHODS=("rope" "alibi" "sinusoidal" "diet" "t5_relative" "math_adaptive")
if [[ ! " ${VALID_PE_METHODS[*]} " =~ " ${PE_METHOD} " ]]; then
    error "Invalid PE method: $PE_METHOD. Valid options: ${VALID_PE_METHODS[*]}"
fi

# Set up experiment directory
EXPERIMENT_DIR="$OUTPUT_DIR/experiments/$EXPERIMENT_NAME/node_$NODE_ID"
mkdir -p "$EXPERIMENT_DIR"
mkdir -p "$EXPERIMENT_DIR/logs"
mkdir -p "$EXPERIMENT_DIR/checkpoints"
mkdir -p "$EXPERIMENT_DIR/results"

log "Experiment directory: $EXPERIMENT_DIR"

# Set up environment
log "Setting up environment..."

# Load modules (HPC specific)
if command -v module &> /dev/null; then
    module load python/3.9
    module load cuda/11.8
    module load gcc/9.3.0
fi

# Create Python virtual environment if not exists
VENV_DIR="$OUTPUT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    log "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install dependencies
log "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set environment variables
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="$NODE_ID"
export WANDB_PROJECT="$WANDB_PROJECT"
export WANDB_RUN_NAME="${EXPERIMENT_NAME}_node_${NODE_ID}_${PE_METHOD}"
export TRANSFORMERS_CACHE="$OUTPUT_DIR/cache/transformers"
export HF_DATASETS_CACHE="$OUTPUT_DIR/cache/datasets"

# Create cache directories
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "$HF_DATASETS_CACHE"

# Create experiment configuration
EXPERIMENT_CONFIG="$EXPERIMENT_DIR/experiment_config.json"
log "Creating experiment configuration: $EXPERIMENT_CONFIG"

cat > "$EXPERIMENT_CONFIG" << EOF
{
  "experiment_name": "$EXPERIMENT_NAME",
  "node_id": $NODE_ID,
  "pe_method": "$PE_METHOD",
  "base_model": "$BASE_MODEL",
  "datasets": "$DATASETS",
  "training_config": {
    "max_steps": $MAX_STEPS,
    "batch_size": $BATCH_SIZE,
    "learning_rate": $LEARNING_RATE,
    "save_steps": $SAVE_STEPS,
    "eval_steps": $EVAL_STEPS,
    "max_length": $MAX_LENGTH,
    "use_lora": $USE_LORA,
    "load_in_4bit": $LOAD_IN_4BIT,
    "gradient_accumulation_steps": 8,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "fp16": true,
    "dataloader_num_workers": 4
  },
  "output_config": {
    "output_dir": "$EXPERIMENT_DIR",
    "logging_dir": "$EXPERIMENT_DIR/logs",
    "save_total_limit": 3,
    "load_best_model_at_end": true,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": false
  },
  "hardware_config": {
    "device": "cuda:0",
    "world_size": 1,
    "local_rank": 0
  }
}
EOF

# Create training script
TRAINING_SCRIPT="$EXPERIMENT_DIR/train.py"
log "Creating training script: $TRAINING_SCRIPT"

cat > "$TRAINING_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""
Training script for positional encoding comparison experiment.
"""

import os
import sys
import json
import logging
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
import wandb

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from models.mathematical_reasoning_model import create_mathematical_reasoning_model
from data.math_dataset_loader import MathDatasetLoader, load_math_datasets

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_experiment_config():
    """Load experiment configuration."""
    config_path = Path(__file__).parent / "experiment_config.json"
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    """Main training function."""
    # Load configuration
    config = load_experiment_config()
    
    logger.info(f"Starting experiment: {config['experiment_name']}")
    logger.info(f"PE Method: {config['pe_method']}")
    logger.info(f"Node ID: {config['node_id']}")
    
    # Initialize wandb
    wandb.init(
        project=os.getenv('WANDB_PROJECT', 'math_pe_research'),
        name=os.getenv('WANDB_RUN_NAME'),
        config=config
    )
    
    # Load tokenizer
    logger.info(f"Loading tokenizer for: {config['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info(f"Creating model with PE method: {config['pe_method']}")
    model = create_mathematical_reasoning_model(
        pe_method=config['pe_method'],
        base_model=config['base_model'],
        use_lora=config['training_config']['use_lora'],
        load_in_4bit=config['training_config']['load_in_4bit']
    )
    
    # Load datasets
    logger.info(f"Loading datasets: {config['datasets']}")
    dataset_names = config['datasets'].split(',')
    
    data_loader = MathDatasetLoader(
        tokenizer=tokenizer,
        max_length=config['training_config']['max_length']
    )
    
    # Load training data
    train_problems = data_loader.load_multiple_datasets(
        dataset_names, split='train', max_samples_per_dataset=10000
    )
    train_dataset = data_loader.create_pytorch_dataset(train_problems, is_training=True)
    
    # Load evaluation data  
    eval_problems = data_loader.load_multiple_datasets(
        dataset_names, split='test', max_samples_per_dataset=1000
    )
    eval_dataset = data_loader.create_pytorch_dataset(eval_problems, is_training=False)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config['output_config']['output_dir'],
        logging_dir=config['output_config']['logging_dir'],
        
        # Training parameters
        num_train_epochs=1,  # We use max_steps instead
        max_steps=config['training_config']['max_steps'],
        per_device_train_batch_size=config['training_config']['batch_size'],
        per_device_eval_batch_size=config['training_config']['batch_size'],
        gradient_accumulation_steps=config['training_config']['gradient_accumulation_steps'],
        
        # Optimization
        learning_rate=config['training_config']['learning_rate'],
        weight_decay=config['training_config']['weight_decay'],
        warmup_steps=config['training_config']['warmup_steps'],
        
        # Precision and performance
        fp16=config['training_config']['fp16'],
        dataloader_num_workers=config['training_config']['dataloader_num_workers'],
        
        # Evaluation and saving
        evaluation_strategy="steps",
        eval_steps=config['training_config']['eval_steps'],
        save_strategy="steps",
        save_steps=config['training_config']['save_steps'],
        save_total_limit=config['output_config']['save_total_limit'],
        
        # Monitoring
        logging_steps=50,
        report_to="wandb",
        
        # Best model
        load_best_model_at_end=config['output_config']['load_best_model_at_end'],
        metric_for_best_model=config['output_config']['metric_for_best_model'],
        greater_is_better=config['output_config']['greater_is_better'],
        
        # Memory optimization
        remove_unused_columns=False,
        group_by_length=True,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Start training
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    trainer.save_state()
    
    # Log final results
    logger.info(f"Training completed!")
    logger.info(f"Final train loss: {train_result.training_loss}")
    
    # Final evaluation
    logger.info("Running final evaluation...")
    eval_result = trainer.evaluate()
    logger.info(f"Final eval loss: {eval_result['eval_loss']}")
    
    # Save results
    results_file = Path(config['output_config']['output_dir']) / "results" / "final_results.json"
    results_file.parent.mkdir(exist_ok=True)
    
    results = {
        'experiment_name': config['experiment_name'],
        'pe_method': config['pe_method'],
        'node_id': config['node_id'],
        'train_loss': train_result.training_loss,
        'eval_loss': eval_result['eval_loss'],
        'train_samples': len(train_dataset),
        'eval_samples': len(eval_dataset),
        'total_steps': train_result.global_step
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    wandb.finish()

if __name__ == "__main__":
    main()
EOF

# Create job submission script for HPC
JOB_SCRIPT="$EXPERIMENT_DIR/submit_job.pbs"
log "Creating HPC job script: $JOB_SCRIPT"

cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#PBS -N math_pe_${PE_METHOD}_${NODE_ID}
#PBS -l select=1:ncpus=8:mem=32GB:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o $EXPERIMENT_DIR/logs/job_output.log
#PBS -q gpu

# Change to experiment directory
cd $EXPERIMENT_DIR

# Load environment
source $VENV_DIR/bin/activate

# Set environment variables
export PYTHONPATH="$PWD/../../../src:\$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="$WANDB_PROJECT"
export WANDB_RUN_NAME="${EXPERIMENT_NAME}_node_${NODE_ID}_${PE_METHOD}"

# Run training
python train.py
EOF

# Make scripts executable
chmod +x "$TRAINING_SCRIPT"
chmod +x "$JOB_SCRIPT"

# Option to run locally or submit to queue
if [[ "${HPC_MODE:-true}" == "true" ]]; then
    log "Submitting job to HPC queue..."
    cd "$EXPERIMENT_DIR"
    job_id=$(qsub submit_job.pbs)
    log "Job submitted with ID: $job_id"
    
    # Create monitoring script
    MONITOR_SCRIPT="$EXPERIMENT_DIR/monitor.sh"
    cat > "$MONITOR_SCRIPT" << EOF
#!/bin/bash
echo "Monitoring job: $job_id"
echo "Job status:"
qstat $job_id
echo ""
echo "Recent log output:"
tail -20 "$EXPERIMENT_DIR/logs/job_output.log"
EOF
    chmod +x "$MONITOR_SCRIPT"
    
    log "Monitor job with: $MONITOR_SCRIPT"
    log "View logs with: tail -f $EXPERIMENT_DIR/logs/job_output.log"
    
else
    log "Running experiment locally..."
    cd "$EXPERIMENT_DIR"
    python train.py
fi

log "Experiment setup completed!"
log "Experiment directory: $EXPERIMENT_DIR"
log "Configuration: $EXPERIMENT_CONFIG"
log "Training script: $TRAINING_SCRIPT"

# Create summary
SUMMARY_FILE="$EXPERIMENT_DIR/experiment_summary.md"
cat > "$SUMMARY_FILE" << EOF
# Experiment Summary

## Configuration
- **Experiment Name**: $EXPERIMENT_NAME
- **Node ID**: $NODE_ID
- **PE Method**: $PE_METHOD
- **Base Model**: $BASE_MODEL
- **Datasets**: $DATASETS

## Training Parameters
- **Max Steps**: $MAX_STEPS
- **Batch Size**: $BATCH_SIZE
- **Learning Rate**: $LEARNING_RATE
- **Max Length**: $MAX_LENGTH
- **Use LoRA**: $USE_LORA
- **Load in 4-bit**: $LOAD_IN_4BIT

## Files
- **Config**: \`experiment_config.json\`
- **Training Script**: \`train.py\`
- **Job Script**: \`submit_job.pbs\`
- **Logs**: \`logs/\`
- **Checkpoints**: \`checkpoints/\`
- **Results**: \`results/\`

## Commands
- **Monitor**: \`./monitor.sh\`
- **View Logs**: \`tail -f logs/job_output.log\`
- **Check Results**: \`cat results/final_results.json\`
EOF

log "Experiment summary saved to: $SUMMARY_FILE"
EOF