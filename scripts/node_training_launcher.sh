#!/bin/bash
# scripts/node_training_launcher.sh
# Node-specific training launcher for mathematical reasoning models

set -e  # Exit on any error

# Function to log messages with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Starting node training launcher..."

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --node_rank=*) NODE_RANK="${1#*=}"; shift ;;
        --pe_method=*) PE_METHOD="${1#*=}"; shift ;;
        --sota_model=*) SOTA_MODEL="${1#*=}"; shift ;;
        --model_name=*) MODEL_NAME="${1#*=}"; shift ;;
        --master_addr=*) MASTER_ADDR="${1#*=}"; shift ;;
        --master_port=*) MASTER_PORT="${1#*=}"; shift ;;
        --world_size=*) WORLD_SIZE="${1#*=}"; shift ;;
        *) shift ;;
    esac
done

# Validate required arguments
if [[ -z "$NODE_RANK" || -z "$PE_METHOD" || -z "$SOTA_MODEL" ]]; then
    log "ERROR: Missing required arguments"
    log "Usage: $0 --node_rank=X --pe_method=Y --sota_model=Z [other options]"
    exit 1
fi

log "Configuration:"
log "  Node Rank: $NODE_RANK"
log "  PE Method: $PE_METHOD"
log "  SOTA Model: $SOTA_MODEL"
log "  Model Name: $MODEL_NAME"
log "  Master Address: $MASTER_ADDR"
log "  Master Port: $MASTER_PORT"
log "  World Size: $WORLD_SIZE"

# Set environment variables for this node
export RANK=$NODE_RANK
export LOCAL_RANK=0
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$WORLD_SIZE
export CUDA_VISIBLE_DEVICES=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN

# Create output directory structure
OUTPUT_DIR="/scratch/$USER/math_reasoning/results/node_${NODE_RANK}_${PE_METHOD}_${SOTA_MODEL}"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
LOG_DIR="$OUTPUT_DIR/logs"

log "Creating output directories..."
mkdir -p $OUTPUT_DIR
mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR

# Create node-specific config file
CONFIG_FILE="/scratch/$USER/math_reasoning/configs/node_configs/node_${NODE_RANK}_config.json"
log "Generating config file: $CONFIG_FILE"

python3 -c "
import json
import os

config = {
    'model_config': {
        'd_model': 4096,
        'n_heads': 32,
        'n_encoder_layers': 24,
        'n_decoder_layers': 24,
        'd_ff': 11008,
        'vocab_size': 50000,
        'max_seq_len': 4096,
        'dropout': 0.1,
        'positional_encoding': '${PE_METHOD}'
    },
    'sota_integration': {
        'base_model': '${MODEL_NAME}',
        'initialization_strategy': 'continued_pretraining',
        'use_grpo': True,
        'math_data_ratio': 0.7,
        'use_lora': True,
        'lora_rank': 64,
        'lora_alpha': 16
    },
    'training_config': {
        'batch_size': 2,
        'gradient_accumulation_steps': 8,
        'learning_rate': 1e-5,
        'warmup_steps': 500,
        'max_steps': 10000,
        'weight_decay': 0.01,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'fp16': True,
        'gradient_checkpointing': True,
        'dataloader_num_workers': 4,
        'save_strategy': 'steps',
        'save_steps': 1000,
        'eval_strategy': 'steps',
        'eval_steps': 500,
        'logging_steps': 50,
        'report_to': 'wandb'
    },
    'data_config': {
        'train_datasets': ['math', 'gsm8k'],
        'eval_datasets': ['math_test', 'gsm8k_test'],
        'chain_of_thought': True,
        'max_reasoning_steps': 10,
        'train_split_size': 0.9,
        'max_train_samples': 50000,
        'max_eval_samples': 1000
    },
    'evaluation_config': {
        'metrics': ['exact_match', 'reasoning_accuracy', 'perplexity', 'attention_entropy'],
        'generate_max_length': 512,
        'generate_temperature': 0.7,
        'generate_do_sample': True
    },
    'distributed_config': {
        'node_rank': ${NODE_RANK},
        'world_size': ${WORLD_SIZE},
        'master_addr': '${MASTER_ADDR}',
        'master_port': ${MASTER_PORT}
    },
    'output_config': {
        'output_dir': '${OUTPUT_DIR}',
        'checkpoint_dir': '${CHECKPOINT_DIR}',
        'log_dir': '${LOG_DIR}',
        'run_name': 'node_${NODE_RANK}_${PE_METHOD}_${SOTA_MODEL}'
    }
}

os.makedirs(os.path.dirname('${CONFIG_FILE}'), exist_ok=True)
with open('${CONFIG_FILE}', 'w') as f:
    json.dump(config, f, indent=2)

print(f'Config file created: ${CONFIG_FILE}')
"

# Verify config file was created
if [ ! -f "$CONFIG_FILE" ]; then
    log "ERROR: Failed to create config file"
    exit 1
fi

log "Config file created successfully"

# Setup WandB environment
export WANDB_PROJECT="math_reasoning_pe_comparison"
export WANDB_RUN_NAME="node_${NODE_RANK}_${PE_METHOD}_${SOTA_MODEL}"
export WANDB_DIR="$LOG_DIR"

# Check if container exists
CONTAINER_FILE="/scratch/$USER/math_reasoning/math-reasoning-hpc.sqsh"
if [ ! -f "$CONTAINER_FILE" ]; then
    log "ERROR: Container file not found: $CONTAINER_FILE"
    exit 1
fi

log "Starting Enroot container..."

# Launch training with Enroot container
enroot start \
    --rw \
    --mount /scratch/$USER/math_reasoning:/workspace \
    --mount /home/$USER/.cache/huggingface:/root/.cache/huggingface \
    --mount /tmp:/tmp \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env RANK=$RANK \
    --env LOCAL_RANK=$LOCAL_RANK \
    --env MASTER_ADDR=$MASTER_ADDR \
    --env MASTER_PORT=$MASTER_PORT \
    --env WORLD_SIZE=$WORLD_SIZE \
    --env NCCL_SOCKET_IFNAME=ib0 \
    --env WANDB_PROJECT=$WANDB_PROJECT \
    --env WANDB_RUN_NAME=$WANDB_RUN_NAME \
    --env WANDB_DIR=$WANDB_DIR \
    --env HF_TOKEN=$HF_TOKEN \
    $CONTAINER_FILE \
    python -m training.sota_mathematical_reasoning_trainer \
    --config $CONFIG_FILE \
    --node_rank $NODE_RANK \
    --pe_method $PE_METHOD \
    --sota_model $SOTA_MODEL

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    log "Training completed successfully on node $NODE_RANK"
    
    # Run evaluation if training succeeded
    log "Starting evaluation..."
    enroot start \
        --rw \
        --mount /scratch/$USER/math_reasoning:/workspace \
        --mount /home/$USER/.cache/huggingface:/root/.cache/huggingface \
        --env CUDA_VISIBLE_DEVICES=0 \
        $CONTAINER_FILE \
        python -m evaluation.comprehensive_evaluator \
        --model_path $CHECKPOINT_DIR \
        --config $CONFIG_FILE \
        --output_dir $OUTPUT_DIR/evaluation
    
    EVAL_EXIT_CODE=$?
    if [ $EVAL_EXIT_CODE -eq 0 ]; then
        log "Evaluation completed successfully on node $NODE_RANK"
    else
        log "WARNING: Evaluation failed on node $NODE_RANK with exit code $EVAL_EXIT_CODE"
    fi
else
    log "ERROR: Training failed on node $NODE_RANK with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

# Generate node-specific report
log "Generating node report..."
python3 -c "
import json
from datetime import datetime

report = {
    'node_rank': ${NODE_RANK},
    'pe_method': '${PE_METHOD}',
    'sota_model': '${SOTA_MODEL}',
    'model_name': '${MODEL_NAME}',
    'completion_time': datetime.now().isoformat(),
    'training_exit_code': ${TRAIN_EXIT_CODE},
    'evaluation_exit_code': ${EVAL_EXIT_CODE:-999},
    'output_dir': '${OUTPUT_DIR}',
    'status': 'completed' if ${TRAIN_EXIT_CODE} == 0 else 'failed'
}

with open('${OUTPUT_DIR}/node_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f'Node report saved to: ${OUTPUT_DIR}/node_report.json')
"

log "Node $NODE_RANK training launcher completed"
log "Final output directory: $OUTPUT_DIR"

exit $TRAIN_EXIT_CODE