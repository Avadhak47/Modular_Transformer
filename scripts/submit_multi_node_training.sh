#!/bin/bash
# scripts/submit_multi_node_training.sh
# Master orchestration script for multi-node mathematical reasoning training

#PBS -N math_reasoning_multi_node
#PBS -q gpuq
#PBS -l select=5:ncpus=8:ngpus=1:mem=128GB
#PBS -l walltime=48:00:00
#PBS -P cse
#PBS -o /scratch/$USER/math_reasoning/logs/multi_node_${PBS_JOBID}.out
#PBS -e /scratch/$USER/math_reasoning/logs/multi_node_${PBS_JOBID}.err
#PBS -j oe
#PBS -m abe
#PBS -M ${USER}@iitd.ac.in

# Load required modules for IITD HPC
module load enroot/3.4.1
module load cuda/11.8
module load nccl/2.15.5
module load openmpi/4.1.4

echo "=========================================="
echo "IITD HPC Multi-Node Mathematical Reasoning Training"
echo "Job ID: $PBS_JOBID"
echo "Start Time: $(date)"
echo "Nodes allocated: $(cat $PBS_NODEFILE | sort | uniq | wc -l)"
echo "Total cores: $(cat $PBS_NODEFILE | wc -l)"
echo "=========================================="

# Setup environment variables
export MASTER_ADDR=$(head -1 $PBS_NODEFILE)
export MASTER_PORT=29500
export WORLD_SIZE=5
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=INFO

# Base directory setup
BASE_DIR="/scratch/$USER/math_reasoning"
cd $BASE_DIR

# Create log directories
mkdir -p $BASE_DIR/logs/node_logs
mkdir -p $BASE_DIR/results
mkdir -p $BASE_DIR/checkpoints

# Import Enroot container if not exists
CONTAINER_FILE="math-reasoning-hpc.sqsh"
if [ ! -f $CONTAINER_FILE ]; then
    echo "Importing Enroot container..."
    if [ -f "math-reasoning-hpc.tar" ]; then
        enroot import -o $CONTAINER_FILE dockerd://math-reasoning-hpc.tar
    else
        echo "ERROR: Container tar file not found!"
        exit 1
    fi
fi

# Verify container
if [ ! -f $CONTAINER_FILE ]; then
    echo "ERROR: Failed to create container!"
    exit 1
fi

echo "Container ready: $CONTAINER_FILE"

# Define positional encoding methods and SOTA models
declare -a PE_METHODS=("sinusoidal" "rope" "alibi" "diet" "t5_relative")
declare -a SOTA_MODELS=("deepseek-math" "internlm-math" "orca-math" "dotamath" "mindstar")
declare -a MODEL_NAMES=(
    "deepseek-ai/deepseek-math-7b-base"
    "internlm/internlm-math-7b"
    "microsoft/orca-math-7b" 
    "deepseek-ai/deepseek-coder-7b-base"
    "mistralai/Mistral-7B-v0.1"
)

# Display configuration
echo "Training Configuration:"
echo "======================"
for i in {0..4}; do
    echo "Node $((i+1)): ${PE_METHODS[$i]} + ${SOTA_MODELS[$i]}"
done
echo ""

# Get unique nodes
NODES=($(cat $PBS_NODEFILE | sort | uniq))
echo "Allocated nodes: ${NODES[@]}"

# Launch training on each node
echo "Launching training on all nodes..."
pids=()

for i in {0..4}; do
    NODE_RANK=$i
    PE_METHOD=${PE_METHODS[$i]}
    SOTA_MODEL=${SOTA_MODELS[$i]}
    MODEL_NAME=${MODEL_NAMES[$i]}
    NODE_HOST=${NODES[$i]}
    
    echo "Starting training on node $NODE_HOST (rank $NODE_RANK) with $PE_METHOD + $SOTA_MODEL"
    
    # Launch training on specific node
    ssh $NODE_HOST "cd $BASE_DIR && \
        export CUDA_VISIBLE_DEVICES=0 && \
        export RANK=$NODE_RANK && \
        export LOCAL_RANK=0 && \
        export MASTER_ADDR=$MASTER_ADDR && \
        export MASTER_PORT=$MASTER_PORT && \
        export WORLD_SIZE=$WORLD_SIZE && \
        export NCCL_SOCKET_IFNAME=ib0 && \
        module load enroot/3.4.1 && \
        module load cuda/11.8 && \
        ./scripts/node_training_launcher.sh \
        --node_rank=$NODE_RANK \
        --pe_method=$PE_METHOD \
        --sota_model=$SOTA_MODEL \
        --model_name='$MODEL_NAME' \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --world_size=$WORLD_SIZE \
        > $BASE_DIR/logs/node_logs/node_${NODE_RANK}_${PE_METHOD}.log 2>&1" &
    
    pids+=($!)
    sleep 10  # Stagger starts to avoid conflicts
done

echo "All training jobs launched. PIDs: ${pids[@]}"

# Monitor training progress
echo "Monitoring training progress..."
./scripts/monitor_multi_node_training.sh &
MONITOR_PID=$!

# Wait for all training jobs to complete
echo "Waiting for all training jobs to complete..."
for pid in ${pids[@]}; do
    wait $pid
    exit_code=$?
    echo "Training job $pid completed with exit code: $exit_code"
done

# Stop monitoring
kill $MONITOR_PID 2>/dev/null

echo "All training jobs completed!"

# Aggregate results
echo "Aggregating results from all nodes..."
python3 $BASE_DIR/scripts/aggregate_multi_node_results.py \
    --base_dir $BASE_DIR \
    --output_dir $BASE_DIR/results/final_comparison

# Generate final report
echo "Generating final comparison report..."
python3 $BASE_DIR/analysis/generate_research_report.py \
    --results_dir $BASE_DIR/results/final_comparison \
    --output_file $BASE_DIR/results/final_research_report.pdf

echo "=========================================="
echo "Multi-node training completed successfully!"
echo "End Time: $(date)"
echo "Results available at: $BASE_DIR/results/final_comparison"
echo "Final report: $BASE_DIR/results/final_research_report.pdf"
echo "=========================================="

# Send completion notification
echo "Multi-node mathematical reasoning training completed successfully on $(date)" | \
    mail -s "HPC Training Complete - Job $PBS_JOBID" ${USER}@iitd.ac.in