#!/bin/bash
# setup_padum_automation.sh
# Usage: ./setup_padum_automation.sh <padum_username>
# Automates the setup process on PADUM HPC using only module-based environments (no containerization, no internet)

PADUM_USER="$1"
PADUM_HOST="padum-login.iitd.ac.in"
PADUM_SCRATCH="/scratch/$PADUM_USER/math_reasoning"

if [ -z "$PADUM_USER" ]; then
    echo "Usage: $0 <padum_username>"
    exit 1
fi

echo "=========================================="
echo "PADUM HPC Automation Setup (Module-Only)"
echo "=========================================="

# Step 1: Transfer files to PADUM
echo "Step 1: Transferring files to PADUM..."
scp -r . "$PADUM_USER@$PADUM_HOST:$PADUM_SCRATCH/"

# Step 2: Set up environment and modules on PADUM
echo "Step 2: Setting up environment and modules on PADUM..."
ssh "$PADUM_USER@$PADUM_HOST" << 'EOF'
    cd /scratch/$USER/math_reasoning
    
    # Load recommended Anaconda/Miniconda module
    module purge
    module load apps/anaconda/3EnvCreation
    
    # Clone the base environment (no internet required)
    conda create --prefix=~/transformer_env --clone base -y
    source activate ~/transformer_env
    
    # (Optional) Load additional modules for ML frameworks if needed
    # module avail | grep -i pytorch
    # module load apps/pytorch/1.10.0/gpu/intelpython3.7
    
    # Check available packages
    conda list
    module avail
    
    echo "Environment setup complete. Only HPC-provided packages are available."
EOF

echo "=========================================="
echo "PADUM HPC Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. SSH to PADUM: ssh $PADUM_USER@$PADUM_HOST"
echo "2. Navigate to: cd $PADUM_SCRATCH"
echo "3. Edit and submit your PBS job script (see PADUM_DEPLOYMENT_GUIDE.md)"
echo "4. Monitor jobs and check logs as usual."
echo "" 