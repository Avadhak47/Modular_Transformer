#!/bin/bash
# monitor_padum_job.sh
# Usage: ./monitor_padum_job.sh <job_id>
# Monitors PBS job status, logs, and provides troubleshooting tips.

JOBID="$1"
SCRATCH_DIR="/scratch/$USER/math_reasoning"
LOG_DIR="$SCRATCH_DIR/logs"
RESULTS_DIR="$SCRATCH_DIR/results"

if [ -z "$JOBID" ]; then
    echo "Usage: $0 <job_id>"
    exit 1
fi

echo "==== PBS Job Status ===="
qstat -f "$JOBID" 2>/dev/null || echo "Job $JOBID not found in queue."

echo
echo "==== Recent Log Output ===="
LOGFILE=$(ls -t $LOG_DIR/*${JOBID}*.out 2>/dev/null | head -n1)
ERRFILE=$(ls -t $LOG_DIR/*${JOBID}*.err 2>/dev/null | head -n1)
if [ -f "$LOGFILE" ]; then
    echo "--- $LOGFILE (last 40 lines) ---"
    tail -n 40 "$LOGFILE"
else
    echo "No output log found for job $JOBID."
fi
if [ -f "$ERRFILE" ]; then
    echo "--- $ERRFILE (last 40 lines) ---"
    tail -n 40 "$ERRFILE"
else
    echo "No error log found for job $JOBID."
fi

echo
echo "==== Checkpoints and Results ===="
ls -lh $RESULTS_DIR 2>/dev/null || echo "No results directory found."

echo
echo "==== Common Troubleshooting ===="
echo "1. If your job is stuck in Q state:"
echo "   - Check resource availability: pbsnodes -a | grep -i state"
echo "   - Check for quota issues: lfs quota -u $USER $SCRATCH_DIR"
echo "2. If your job fails immediately:"
echo "   - Check error log above for Python or Singularity errors."
echo "   - Ensure all modules (cuda, singularity, python) are loaded in your PBS script."
echo "3. If you see CUDA or GPU errors:"
echo "   - Confirm you requested a GPU node and loaded the correct CUDA module."
echo "   - Check nvidia-smi inside your Singularity container: singularity exec --nv math_reasoning.sif nvidia-smi"
echo "4. For out-of-memory errors:"
echo "   - Reduce batch size in your training script."
echo "   - Request more memory in your PBS script."
echo "5. For missing dataset errors:"
echo "   - Ensure datasets are present in /scratch/$USER/math_reasoning/data or update your loader paths."
echo
echo "==== For further help ===="
echo "Contact PADUM support or check the IITD PADUM documentation." 