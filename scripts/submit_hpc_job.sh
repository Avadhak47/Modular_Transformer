#!/bin/bash
#PBS -N math_reasoning_pe_comparison
#PBS -l select=5:ncpus=32:ngpus=4:mem=128GB
#PBS -l walltime=48:00:00
#PBS -q gpu
#PBS -P iitd_math_reasoning
#PBS -o /scratch/$USER/math_reasoning/logs/job_output.log
#PBS -e /scratch/$USER/math_reasoning/logs/job_error.log
#PBS -m abe
#PBS -M $USER@iitd.ac.in

# Mathematical Reasoning with Positional Encoding Comparison
# Multi-node training on IITD HPC cluster
# Author: Research Team, IIT Delhi

set -e

echo "=========================================="
echo "Mathematical Reasoning Multi-Node Training"
echo "Start Time: $(date)"
echo "Job ID: $PBS_JOBID"
echo "Nodes allocated: $(cat $PBS_NODEFILE | wc -l)"
echo "=========================================="

# Setup environment
export PROJECT_ROOT="/scratch/$USER/math_reasoning"
export DATA_DIR="$PROJECT_ROOT/data"
export OUTPUT_DIR="$PROJECT_ROOT/results"
export LOG_DIR="$PROJECT_ROOT/logs"

# Create directories
mkdir -p $DATA_DIR $OUTPUT_DIR $LOG_DIR

# Load required modules
module purge
module load python/3.9.0
module load cuda/11.8
module load nccl/2.15.5
module load pytorch/1.13.0

# Setup distributed training environment
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
export MASTER_PORT=29500
export WORLD_SIZE=5

# Activate virtual environment (created separately)
source $PROJECT_ROOT/venv/bin/activate

# Copy project files to compute nodes
cd $PROJECT_ROOT
echo "Copying project files..."

# Function to run training on each node
run_node_training() {
    local node_id=$1
    local node_rank=$2
    local node_name=$3
    
    echo "Starting training on Node $node_id (Rank $node_rank) at $node_name"
    
    # Set environment for this node
    export RANK=$node_rank
    export LOCAL_RANK=0
    
    # Create node-specific config
    config_file="$OUTPUT_DIR/node_${node_id}_config.json"
    
    # Define PE methods for each node
    case $node_id in
        0) pe_method="sinusoidal" ;;
        1) pe_method="rope" ;;
        2) pe_method="alibi" ;;
        3) pe_method="diet" ;;
        4) pe_method="t5_relative" ;;
    esac
    
    # Generate configuration
    cat > $config_file << EOF
{
  "model_config": {
    "positional_encoding": "$pe_method",
    "d_model": 4096,
    "n_heads": 32,
    "n_layers": 24,
    "max_seq_len": 4096,
    "dropout": 0.1
  },
  "training_config": {
    "batch_size": 2,
    "learning_rate": 1e-4,
    "max_steps": 10000,
    "warmup_steps": 500,
    "eval_steps": 1000,
    "save_steps": 2000,
    "logging_steps": 100,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "report_to": "wandb"
  },
  "data_config": {
    "train_datasets": ["math", "gsm8k"],
    "eval_datasets": ["math_test", "gsm8k_test"],
    "max_train_samples": 50000,
    "max_eval_samples": 2000
  },
  "sota_integration": {
    "base_model": "deepseek-ai/deepseek-math-7b-instruct",
    "use_lora": true,
    "lora_rank": 64,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "use_quantization": true,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  },
  "experiment_config": {
    "name": "node_${node_id}_${pe_method}",
    "output_dir": "$OUTPUT_DIR/node_${node_id}",
    "logging_dir": "$LOG_DIR/node_${node_id}",
    "wandb_project": "math_reasoning_pe_comparison",
    "wandb_run_name": "node_${node_id}_${pe_method}"
  },
  "hardware_config": {
    "node_id": $node_id,
    "gpus_per_node": 4,
    "cpu_cores": 32,
    "memory_gb": 128
  }
}
EOF
    
    # Run training
    python3 $PROJECT_ROOT/scripts/train_mathematical_model.py \
        --config $config_file \
        --node_id $node_id \
        --local_rank 0 \
        > $LOG_DIR/node_${node_id}_training.log 2>&1 &
    
    echo "Node $node_id training started (PID: $!)"
    echo $! > $LOG_DIR/node_${node_id}.pid
}

# Launch training on all nodes
echo "Launching multi-node training..."

node_list=($(cat $PBS_NODEFILE | sort | uniq))
for i in ${!node_list[@]}; do
    node_name=${node_list[$i]}
    echo "Launching on node $i: $node_name"
    
    ssh $node_name "cd $PROJECT_ROOT && $(declare -f run_node_training); run_node_training $i $i $node_name" &
done

# Wait for all training jobs to start
sleep 30

# Monitor training progress
echo "Monitoring training progress..."
monitor_training() {
    while true; do
        echo "=== Training Status at $(date) ==="
        
        active_nodes=0
        for i in {0..4}; do
            pid_file="$LOG_DIR/node_${i}.pid"
            if [[ -f $pid_file ]]; then
                pid=$(cat $pid_file)
                if ps -p $pid > /dev/null 2>&1; then
                    echo "Node $i: RUNNING (PID: $pid)"
                    active_nodes=$((active_nodes + 1))
                else
                    echo "Node $i: FINISHED/FAILED"
                fi
            else
                echo "Node $i: NOT STARTED"
            fi
        done
        
        echo "Active nodes: $active_nodes/5"
        
        if [[ $active_nodes -eq 0 ]]; then
            echo "All training processes have finished."
            break
        fi
        
        sleep 300  # Check every 5 minutes
    done
}

# Start monitoring in background
monitor_training &
monitor_pid=$!

# Wait for all background jobs
wait

# Kill monitor if still running
kill $monitor_pid 2>/dev/null || true

echo "=========================================="
echo "Training completed at $(date)"
echo "Job ID: $PBS_JOBID"
echo "=========================================="

# Collect results
echo "Collecting results..."
python3 $PROJECT_ROOT/scripts/collect_results.py \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --report_file $OUTPUT_DIR/final_report.json

echo "Results collected. Check $OUTPUT_DIR/final_report.json for summary."

# Optional: Clean up temporary files
# rm -f $LOG_DIR/*.pid

echo "Job completed successfully!"