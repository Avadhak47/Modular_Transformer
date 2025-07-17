# HPC Deployment Guide for IITD PADUM Cluster

**Branch**: `HPC`

This document provides a complete, reproducible workflow for deploying the Transformer-based mathematical-reasoning project on the IIT Delhi PADUM GPU cluster using NVIDIA Enroot containers. It covers:

1. Preparing the HPC branch & codebase
2. Building an Enroot container from the project Dockerfile
3. Transferring the container image to PADUM scratch space
4. Submitting a *multi-node* PBS job where each node trains a different positional-encoding variant
5. Tracking experiments with Weights & Biases (wandb)
6. Post-training evaluation on GSM8K & MATH datasets
7. Result visualisation and comparison across encodings
8. Inference/how-to-present guide

---

## 1. Branch preparation

```bash
# On your workstation/local clone
$ git checkout -b HPC
$ mkdir -p HPC
$ cp submit_multi_node_training.pbs HPC/
$ git add HPC/HPC_DEPLOYMENT_GUIDE.md HPC/submit_multi_node_training.pbs
$ git commit -m "HPC: deployment guide and multi-node PBS script"
$ git push origin HPC
```

> NOTE: The `HPC` directory keeps cluster-specific artifacts isolated from core library code.

---

## 2. Build the container locally (or on any machine with Docker)

```bash
# 2.1 Build the Docker image using the existing Dockerfile in project root
$ docker build -t math-reasoning:latest .

# 2.2 Convert the image to an Enroot squashfs (requires Enroot ≥ 3.4)
$ enroot import --output math_reasoning.sqsh docker://math-reasoning:latest
```

The resulting `math_reasoning.sqsh` (~3-5 GB) is a self-contained image containing CUDA 11.8, Python 3.10, the project source, and all pinned dependencies from `requirements.txt`.

Optional overlays (e.g. pre-downloaded HF models) can be placed in a read-write overlay file created via:
```bash
$ fallocate -l 5G math_overlay.img
$ mkfs.ext4 -F math_overlay.img
```

---

## 3. Transfer image & data to PADUM scratch

```bash
# Replace <username> with IITD login
$ rsync -avP math_reasoning.sqsh math_overlay.img <username>@padum.iitd.ac.in:/scratch/<username>/containers/
# Push training/validation datasets if not already present
$ rsync -avP local_data/ <username>@padum.iitd.ac.in:/scratch/<username>/data/
```

Directory layout on PADUM scratch (per user):
```
/scratch/$USER/
├── containers/
│   ├── math_reasoning.sqsh
│   └── math_overlay.img  # optional RW overlay
├── data/                 # GSM8K & MATH JSONL files
└── jobs/                 # job logs + outputs
```

---

## 4. Multi-node job submission

The file `HPC/submit_multi_node_training.pbs` allocates **5 GPU nodes** with identical hardware. Each node trains the same model architecture but with a *different positional encoding* (`sinusoidal`, `rope`, `alibi`, `t5_relative`, `diet`).

Submit from the PADUM login node:

```bash
$ cd ~/Transformer   # project checkout on cluster
$ qsub HPC/submit_multi_node_training.pbs
```

Key points in the script:
* `#PBS -l select=5:ncpus=8:ngpus=1:mem=32gb` – requests five identical GPU nodes.
* Node-specific positional encoding chosen via `PE_TYPE=${PE_LIST[$PBS_NODENUM]}` where `PE_LIST` is an indexed bash array.
* The container is **created** once per node (`enroot create`) and started with `--root --rw --nv` to enable CUDA & write overlay.
* Training entrypoint inside the container is `train.py` with CLI flags:
  * `--base_model WizardMath/WizardMath-7B-V1.0` (open-source SOTA math-tuned model on HF)
  * `--positional_encoding $PE_TYPE`
  * Runs for `--epochs 6` using mixed precision and gradient accumulation tuned for 1 GPU.
* All metrics & checkpoints are logged to wandb project `math-reasoning‐HPC` with run names reflecting the encoding.

---

## 5. Weights & Biases setup

```bash
# export once in ~/.bashrc (or inside PBS script)
export WANDB_API_KEY=<your-wandb-token>
export WANDB_PROJECT=math-reasoning-HPC
export WANDB_DIR=/scratch/$USER/wandb
```

Logs, artefacts, and best checkpoints live under `/scratch/$USER/wandb` and are synchronised to the cloud dashboard automatically when outbound connectivity is available.

---

## 6. Post-training evaluation

After the job finishes, aggregate checkpoints live at:
```
/scratch/$USER/math_reasoning/checkpoints/<encoding>/best.pt
```

Run evaluation inside the container (interactive session or another PBS job):

```bash
$ enroot start --root --nv math_env \
    python evaluate.py \
      --checkpoint_dir /scratch/$USER/math_reasoning/checkpoints \
      --datasets gsm8k math \
      --metrics accuracy bleu perplexity \
      --output_file /scratch/$USER/math_reasoning/results/eval_summary.json
```

The script combines GSM8K & MATH test splits, producing JSON with per-encoding metrics. It also logs to wandb table `evaluation`.

---

## 7. Visualisation

Example command (already wrapped in PBS script’s *post* section):
```bash
$ python visualize_eval_results.py \
    --input /scratch/$USER/math_reasoning/results/eval_summary.json \
    --save_dir /scratch/$USER/math_reasoning/figures
```

Generates:
* bar-plots of accuracy vs. encoding
* line plots of validation loss across epochs
* scatter comparing GSM8K vs MATH performance

All figures are uploaded as wandb artefacts and saved locally.

---

## 8. Inference & demo notebook

### Quick CLI inference
```bash
$ enroot start --root --nv math_env \
    python - <<'PY'
from src.model import TransformerModel
from src.utils.training_utils import load_checkpoint
from src.positional_encoding import rope  # choose encoding

model = TransformerModel.load_pretrained('WizardMath/WizardMath-7B-V1.0', positional_encoding=rope)
load_checkpoint(model, '/scratch/$USER/math_reasoning/checkpoints/rope/best.pt')
question = "What is the derivative of x^3?"
print(model.generate_text(question))
PY
```

### Jupyter for presentation

1. Request a PADUM *interactive* GPU node with port forwarding:
   ```bash
   $ qsub -I -q gpuq -l select=1:ncpus=4:ngpus=1:mem=16gb -l walltime=04:00:00
   $ enroot start --root --nv math_env jupyter lab --no-browser --port 8888
   ```
2. Open browser on local machine `http://localhost:8888/?token=...` (SSH-tunnel required).
3. Use provided notebook `examples/mathematical_inference_demo.ipynb` to showcase results.

---

## 9. Clean-up

```bash
# Remove container instances but keep sqsh
$ enroot remove math_env

# Optional: purge scratch artefacts older than 14 days
$ find /scratch/$USER/math_reasoning -mtime +14 -exec rm -rf {} +
```

---

### References
* IITD PADUM documentation: <https://cc.iitd.ac.in/content/gpu-computing-padum>
* Enroot user guide: <https://github.com/NVIDIA/enroot/blob/master/doc/README.md>
* WizardMath model card: <https://huggingface.co/WizardMath/WizardMath-7B-V1.0>

Feel free to raise issues or open pull-requests against the `HPC` branch for improvements.