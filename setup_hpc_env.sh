#!/bin/bash

# 1. Purge all loaded modules to start clean
module purge

# 2. Load the recommended Anaconda/Miniconda module
module load apps/anaconda/3

# 3. (Optional) Load additional modules for specific packages (if available)
# Example: PyTorch, TensorFlow, etc.
# module load apps/pytorch/1.10.0/gpu/intelpython3.7
# module load pythonpackages/3.10.4/cartopy/0.18.0/gnu

# 4. Show all available modules for Python and related packages
echo "Available Python/Conda/ML modules:"
module avail 2>&1 | grep -i -E 'python|anaconda|miniconda|pytorch|tensorflow|scikit|torch|cuda|ml|jupyter'

# 5. Show all packages available in the current conda base environment
echo "Packages available in the current conda environment:"
conda list

# 6. (Optional) Create a new conda environment using only the base packages
# This will NOT fetch anything from the internet, just clone the base environment
conda create --prefix=~/transformer_env --clone base -y

# 7. Activate your new environment
source activate ~/transformer_env

# 8. Show the packages in your new environment
echo "Packages in your new environment:"
conda list

# 9. (Optional) Test Python and key packages
python -c "import sys; print('Python version:', sys.version)"
python -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null || echo "PyTorch not available"
python -c "import numpy; print('NumPy version:', numpy.__version__)" 2>/dev/null || echo "NumPy not available"

echo "Environment setup complete. Only HPC-provided packages are available."