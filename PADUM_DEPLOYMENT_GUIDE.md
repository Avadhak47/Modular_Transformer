# PADUM HPC Deployment Guide

## Overview
This guide provides complete automation for deploying the Mathematical Reasoning Transformer project to IIT Delhi PADUM HPC cluster.

## Prerequisites
- PADUM account with GPU access
- SSH access to `padum-login.iitd.ac.in`
- Docker installed locally (for building image)

## Quick Start

### 1. Automated Setup (Recommended)
```bash
# Run the complete automation script
./setup_padum_automation.sh <your_padum_username>
```

This single command will:
- Export Docker image as tarball
- Transfer all files to PADUM
- Build Singularity image on PADUM
- Create all monitoring and automation scripts
- Set up the complete deployment pipeline

### 2. Manual Setup (Alternative)
If you prefer manual control:

```bash
# Step 1: Export and transfer
./export_and_transfer.sh <your_padum_username>

# Step 2: SSH to PADUM and build Singularity image
ssh <username>@padum-login.iitd.ac.in
cd /scratch/$USER/math_reasoning

# Load modules and build
module load singularity/3.8.0
module load cuda/11.8
singularity build math-reasoning-transformer.sif math-reasoning-transformer.tar
```

## Available Scripts

### On Local Machine
- `setup_padum_automation.sh` - Complete automation
- `export_and_transfer.sh` - Export Docker and transfer files
- `monitor_padum_job.sh` - Monitor individual jobs

### On PADUM (after setup)
- `submit_all_experiments.sh` - Submit jobs for all PE methods
- `monitor_dashboard.sh` - Dashboard view of all jobs
- `monitor_padum_job.sh <job_id>` - Monitor specific job
- `aggregate_results.sh` - Aggregate and compare results
- `cleanup_padum.sh` - Clean up old files

## Usage Examples

### Submit All Experiments
```bash
# On PADUM
cd /scratch/$USER/math_reasoning
./submit_all_experiments.sh
```

This submits jobs for all positional encoding methods:
- sinusoidal
- rope (RoPE)
- alibi (ALiBi)
- diet
- nope
- t5_relative

### Monitor Jobs
```bash
# Dashboard view
./monitor_dashboard.sh

# Individual job monitoring
./monitor_padum_job.sh <job_id>
```

### Check Results
```bash
# Aggregate all results
./aggregate_results.sh

# View results
cat /scratch/$USER/math_reasoning/results/experiment_summary.txt
```

## File Structure on PADUM
```
/scratch/$USER/math_reasoning/
├── math-reasoning-transformer.sif    # Singularity image
├── submit_training.pbs              # PBS job script
├── submit_all_experiments.sh        # Automated submission
├── monitor_dashboard.sh             # Dashboard monitoring
├── monitor_padum_job.sh            # Individual job monitoring
├── aggregate_results.sh             # Results aggregation
├── cleanup_padum.sh                # Cleanup utility
├── logs/                           # Job logs
├── results/                        # Experiment results
├── checkpoints/                    # Model checkpoints
└── data/                          # Dataset cache
```

## Troubleshooting

### Common Issues

#### 1. Singularity Build Fails
```bash
# Check available modules
module avail singularity
module avail cuda

# Try different CUDA version
module load cuda/11.7
```

#### 2. Job Stuck in Queue
```bash
# Check queue status
qstat -q

# Check your job priority
qstat -f <job_id>
```

#### 3. GPU Not Available
```bash
# Check GPU availability
nvidia-smi

# Check job requirements
qstat -f <job_id> | grep resources
```

#### 4. Memory Issues
```bash
# Monitor memory usage
qstat -f <job_id> | grep resources_used

# Adjust memory in PBS script
#PBS -l mem=64GB  # Increase if needed
```

#### 5. Storage Space Issues
```bash
# Check available space
df -h /scratch/$USER

# Clean up old files
./cleanup_padum.sh
```

### Error Logs
- Check `.out` files for standard output
- Check `.err` files for error messages
- Use `tail -f` for real-time monitoring

### Performance Optimization
- Use `OMP_NUM_THREADS=8` for optimal CPU utilization
- Monitor GPU utilization with `nvidia-smi`
- Adjust batch size based on available memory

## Resource Requirements

### Per Job
- **GPU**: 1x V100 (32GB VRAM)
- **CPU**: 8 cores
- **Memory**: 32GB RAM
- **Storage**: ~10GB for checkpoints and results
- **Walltime**: 24 hours (adjustable)

### Total Resources (All Experiments)
- **Jobs**: 6 (one per PE method)
- **Total Walltime**: 144 hours (6 × 24h)
- **Storage**: ~60GB total

## Security Notes
- All scripts use your user account permissions
- No root access required
- Files stored in `/scratch/$USER/` (user-specific)
- Singularity provides container isolation

## Support
For PADUM-specific issues:
- Contact IIT Delhi HPC support
- Check PADUM documentation
- Use `qstat -f` for detailed job information

## Best Practices
1. **Monitor regularly** - Use dashboard script
2. **Clean up periodically** - Run cleanup script weekly
3. **Backup important results** - Copy to home directory
4. **Check logs early** - Identify issues quickly
5. **Use appropriate walltime** - Don't overestimate

## Next Steps
After successful deployment:
1. Monitor job progress with dashboard
2. Analyze results with aggregation script
3. Compare positional encoding performance
4. Generate final report with metrics
5. Clean up resources when complete 