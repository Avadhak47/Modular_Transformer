# IIT Delhi PADUM HPC Deployment Guide (Module-Only, No Internet)

This guide explains how to run the Modular Transformer project on IIT Delhi PADUM HPC using only the modules and pre-installed packages provided by the cluster. **Do not attempt to install new packages from the internet.**

---

## 1. Transfer Your Project

- Zip your project for faster transfer:
  ```bash
  zip -r Transformer.zip Transformer/
  scp Transformer.zip <username>@hpc.iitd.ac.in:~/
  ```
- On HPC, unzip:
  ```bash
  ssh <username>@hpc.iitd.ac.in
  unzip Transformer.zip
  cd Transformer
  ```

---

## 2. Set Up Python Environment (Module-Only)

- **Load the recommended Anaconda/Miniconda module:**
  ```bash
  module purge
  module load apps/anaconda/3EnvCreation  # or apps/miniconda/24.7.1
  ```
- **Clone the base environment (no internet required):**
  ```bash
  conda create --prefix=~/transformer_env --clone base -y
  conda activate ~/transformer_env
  ```
- **(Optional) Load additional modules for ML frameworks if available:**
  ```bash
  module avail | grep -i pytorch
  module load apps/pytorch/1.10.0/gpu/intelpython3.7  # Example, if needed
  ```
- **Check available packages:**
  ```bash
  conda list
  module avail
  ```
- **Do NOT use `conda install` or `pip install` unless you know the package is available locally.**

---

## 3. Prepare PBS Job Script

- Edit `submit_training.pbs` as follows:

```bash
#!/bin/bash
#PBS -N transformer_train
#PBS -l select=1:ncpus=8:ngpus=1:mem=32G
#PBS -l walltime=12:00:00
#PBS -P <your_project_code>
#PBS -o output.log
#PBS -e error.log

cd $PBS_O_WORKDIR

module purge
module load apps/anaconda/3
# (Optional) module load apps/pytorch/1.10.0/gpu/intelpython3.7
conda activate ~/transformer_env

python train.py --pe_type rope --epochs 10 --batch_size 4 --use_wandb --experiment_name "rope_mathematical_reasoning"
```

---

## 4. Submit and Monitor Your Job

- Submit:
  ```bash
  qsub submit_training.pbs
  ```
- Monitor:
  ```bash
  qstat -u $USER
  tail -f output.log
  ```

---

## 5. Tips

- Replace `<username>` and `<your_project_code>` as appropriate.
- Use `$SCRATCH` for large data, copy results to `$HOME` if needed.
- Do NOT run heavy jobs on the login node.
- Use `module avail` to see available modules.
- Check logs (`output.log`, `error.log`) for debugging.
- If a required package is missing, check for a module or request it from HPC support.

---

## 6. (Optional) Using Only Available Packages

- All dependencies must be satisfied by the base environment or loaded modules.
- If you need a package not available, contact HPC support to request installation.

---

**This workflow ensures your project runs reliably on IIT Delhi HPC without internet access or containerization.** 