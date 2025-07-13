#!/bin/bash
# setup_padum_automation.sh
# Usage: ./setup_padum_automation.sh <padum_username>
# Automates the complete setup process on PADUM HPC

PADUM_USER="$1"
PADUM_HOST="padum-login.iitd.ac.in"
PADUM_SCRATCH="/scratch/$PADUM_USER/math_reasoning"
SIF_NAME="math-reasoning-transformer.sif"

if [ -z "$PADUM_USER" ]; then
    echo "Usage: $0 <padum_username>"
    exit 1
fi

echo "=========================================="
echo "PADUM HPC Automation Setup"
echo "=========================================="

# Step 1: Transfer files to PADUM
echo "Step 1: Transferring files to PADUM..."
./export_and_transfer.sh "$PADUM_USER"

# Step 2: Build Singularity image on PADUM
echo "Step 2: Building Singularity image on PADUM..."
ssh "$PADUM_USER@$PADUM_HOST" << 'EOF'
    cd /scratch/$USER/math_reasoning
    
    # Load required modules
    module load singularity/3.8.0
    module load cuda/11.8
    
    echo "Building Singularity image from Docker tarball..."
    singularity build math-reasoning-transformer.sif math-reasoning-transformer.tar
    
    if [ $? -eq 0 ]; then
        echo "Singularity image built successfully!"
        ls -lh math-reasoning-transformer.sif
    else
        echo "Error: Failed to build Singularity image"
        exit 1
    fi
    
    # Test the Singularity image
    echo "Testing Singularity image..."
    singularity exec math-reasoning-transformer.sif python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
    
    # Make monitor script executable
    chmod +x monitor_padum_job.sh
EOF

# Step 3: Create automated job submission script
echo "Step 3: Creating automated job submission script..."
cat > submit_all_experiments.sh << 'EOF'
#!/bin/bash
# submit_all_experiments.sh
# Automatically submits jobs for all positional encoding experiments

SCRATCH_DIR="/scratch/$USER/math_reasoning"
LOG_DIR="$SCRATCH_DIR/logs"
RESULTS_DIR="$SCRATCH_DIR/results"

# Create directories
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# Positional encoding methods to test
PE_METHODS=("sinusoidal" "rope" "alibi" "diet" "nope" "t5_relative")

echo "Submitting jobs for all positional encoding methods..."

for pe_method in "${PE_METHODS[@]}"; do
    echo "Submitting job for $pe_method..."
    
    # Create job-specific PBS script
    cat > "submit_${pe_method}.pbs" << PBS_EOF
#!/bin/bash
#PBS -N math_reasoning_${pe_method}
#PBS -q gpuq
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l mem=32GB
#PBS -l walltime=24:00:00
#PBS -o $LOG_DIR/training_${pe_method}_\${PBS_JOBID}.out
#PBS -e $LOG_DIR/training_${pe_method}_\${PBS_JOBID}.err
#PBS -j oe

# Load required modules
module load cuda/11.8
module load singularity/3.8.0
module load python/3.10

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONPATH=$SCRATCH_DIR:$PYTHONPATH

# Run training with specific positional encoding
cd $SCRATCH_DIR
singularity exec math-reasoning-transformer.sif python train.py \\
    --config config.py \\
    --positional_encoding $pe_method \\
    --output_dir $RESULTS_DIR/${pe_method} \\
    --checkpoint_dir $SCRATCH_DIR/checkpoints/${pe_method} \\
    --log_file $LOG_DIR/training_${pe_method}.log
PBS_EOF

    # Submit the job
    JOB_ID=$(qsub "submit_${pe_method}.pbs")
    echo "Submitted job for $pe_method with ID: $JOB_ID"
    
    # Store job ID for monitoring
    echo "$JOB_ID" > "$LOG_DIR/${pe_method}_job_id.txt"
done

echo "All jobs submitted! Use monitor_padum_job.sh <job_id> to monitor individual jobs."
echo "Job IDs saved in $LOG_DIR/*_job_id.txt"
EOF

# Step 4: Create results aggregation script
echo "Step 4: Creating results aggregation script..."
cat > aggregate_results.sh << 'EOF'
#!/bin/bash
# aggregate_results.sh
# Aggregates and compares results from all experiments

SCRATCH_DIR="/scratch/$USER/math_reasoning"
RESULTS_DIR="$SCRATCH_DIR/results"

echo "Aggregating results from all experiments..."

# Create results summary
cat > "$RESULTS_DIR/experiment_summary.txt" << SUMMARY_EOF
Mathematical Reasoning Transformer - Experiment Summary
====================================================
Date: $(date)
User: $USER

Positional Encoding Comparison Results:
EOF

for pe_dir in "$RESULTS_DIR"/*/; do
    if [ -d "$pe_dir" ]; then
        pe_method=$(basename "$pe_dir")
        echo "Processing $pe_method..."
        
        # Extract metrics if available
        if [ -f "$pe_dir/metrics.json" ]; then
            echo "=== $pe_method ===" >> "$RESULTS_DIR/experiment_summary.txt"
            cat "$pe_dir/metrics.json" >> "$RESULTS_DIR/experiment_summary.txt"
            echo "" >> "$RESULTS_DIR/experiment_summary.txt"
        fi
    fi
done

echo "Results aggregated in $RESULTS_DIR/experiment_summary.txt"
EOF

# Step 5: Transfer automation scripts to PADUM
echo "Step 5: Transferring automation scripts to PADUM..."
scp submit_all_experiments.sh aggregate_results.sh "$PADUM_USER@$PADUM_HOST:$PADUM_SCRATCH/"

# Step 6: Set up monitoring dashboard
echo "Step 6: Setting up monitoring dashboard..."
cat > monitor_dashboard.sh << 'EOF'
#!/bin/bash
# monitor_dashboard.sh
# Provides a dashboard view of all running jobs

SCRATCH_DIR="/scratch/$USER/math_reasoning"
LOG_DIR="$SCRATCH_DIR/logs"

echo "=========================================="
echo "Mathematical Reasoning - Job Dashboard"
echo "=========================================="
echo "Date: $(date)"
echo "User: $USER"
echo ""

# Show all user jobs
echo "=== All User Jobs ==="
qstat -u $USER

echo ""
echo "=== Recent Log Activity ==="
for logfile in "$LOG_DIR"/*.out; do
    if [ -f "$logfile" ]; then
        echo "--- $(basename "$logfile") ---"
        tail -n 5 "$logfile"
        echo ""
    fi
done

echo "=== Error Logs ==="
for errfile in "$LOG_DIR"/*.err; do
    if [ -f "$errfile" ] && [ -s "$errfile" ]; then
        echo "--- $(basename "$errfile") ---"
        tail -n 3 "$errfile"
        echo ""
    fi
done

echo "=== Resource Usage ==="
qstat -f | grep -E "(Job_Name|Job_Owner|job_state|resources_used)"
EOF

scp monitor_dashboard.sh "$PADUM_USER@$PADUM_HOST:$PADUM_SCRATCH/"

# Step 7: Create cleanup script
echo "Step 7: Creating cleanup script..."
cat > cleanup_padum.sh << 'EOF'
#!/bin/bash
# cleanup_padum.sh
# Cleans up old files and jobs

SCRATCH_DIR="/scratch/$USER/math_reasoning"

echo "Cleaning up old files..."

# Remove old log files (keep last 7 days)
find "$SCRATCH_DIR/logs" -name "*.out" -mtime +7 -delete 2>/dev/null
find "$SCRATCH_DIR/logs" -name "*.err" -mtime +7 -delete 2>/dev/null

# Remove old checkpoints (keep last 3 days)
find "$SCRATCH_DIR/checkpoints" -name "*.pt" -mtime +3 -delete 2>/dev/null

# Remove old job scripts
rm -f "$SCRATCH_DIR"/submit_*.pbs 2>/dev/null

echo "Cleanup completed!"
EOF

scp cleanup_padum.sh "$PADUM_USER@$PADUM_HOST:$PADUM_SCRATCH/"

# Step 8: Make all scripts executable on PADUM
echo "Step 8: Making scripts executable on PADUM..."
ssh "$PADUM_USER@$PADUM_HOST" "cd $PADUM_SCRATCH && chmod +x *.sh"

echo "=========================================="
echo "PADUM HPC Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. SSH to PADUM: ssh $PADUM_USER@$PADUM_HOST"
echo "2. Navigate to: cd $PADUM_SCRATCH"
echo "3. Run all experiments: ./submit_all_experiments.sh"
echo "4. Monitor jobs: ./monitor_dashboard.sh"
echo "5. Check individual jobs: ./monitor_padum_job.sh <job_id>"
echo "6. Aggregate results: ./aggregate_results.sh"
echo "7. Clean up: ./cleanup_padum.sh"
echo ""
echo "Files transferred:"
echo "- math-reasoning-transformer.sif (Singularity image)"
echo "- submit_training.pbs (PBS job script)"
echo "- submit_all_experiments.sh (Automated submission)"
echo "- monitor_padum_job.sh (Individual job monitoring)"
echo "- monitor_dashboard.sh (Dashboard view)"
echo "- aggregate_results.sh (Results aggregation)"
echo "- cleanup_padum.sh (Cleanup utility)" 