#!/bin/bash
# export_and_transfer.sh
# Usage: ./export_and_transfer.sh <padum_username>
# Exports Docker image and transfers all files to PADUM HPC.

PADUM_USER="$1"
PADUM_HOST="padum-login.iitd.ac.in"
PADUM_SCRATCH="/scratch/$PADUM_USER/math_reasoning"

if [ -z "$PADUM_USER" ]; then
    echo "Usage: $0 <padum_username>"
    exit 1
fi

# Save Docker image as tarball
IMAGE_NAME="math-reasoning-transformer:latest"
TAR_NAME="math-reasoning-transformer.tar"
echo "Saving Docker image as $TAR_NAME ..."
docker save -o $TAR_NAME $IMAGE_NAME

# Create scratch directory on PADUM
ssh $PADUM_USER@$PADUM_HOST "mkdir -p $PADUM_SCRATCH"

# Transfer files
echo "Transferring Docker image, Singularity.def, PBS script, and monitor script to PADUM ..."
scp $TAR_NAME Singularity.def submit_training.pbs monitor_padum_job.sh $PADUM_USER@$PADUM_HOST:$PADUM_SCRATCH/

echo "All files transferred to $PADUM_HOST:$PADUM_SCRATCH"
echo "On PADUM, you can now build the Singularity image and submit your job." 