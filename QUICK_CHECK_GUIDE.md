# HPC Pre-flight Check Guide

## Quick Usage

### Run the Complete Check
```bash
python hpc_preflight_check.py
```

### What It Checks

#### ✅ **Environment & Setup**
- Python version (3.8+)
- Project directory structure
- Required directories exist

#### ✅ **Dependencies**
- All required packages installed
- PyTorch, Transformers, Datasets, etc.

#### ✅ **Code Quality**
- Syntax errors in all Python files
- Import errors in all modules
- Type checking issues

#### ✅ **Model Functionality**
- Model initialization
- Forward pass with dummy data
- Output shape validation

#### ✅ **Data Pipeline**
- GSM8K dataset loading
- MATH dataset loading
- Data preprocessing

#### ✅ **Positional Encodings**
- All 6 PE methods (sinusoidal, RoPE, ALiBi, diet, nope, t5_relative)
- Shape consistency
- Forward pass validation

#### ✅ **Evaluation Metrics**
- Mathematical reasoning evaluator
- Exact match accuracy
- Perplexity calculation

#### ✅ **Training Pipeline**
- Trainer initialization
- Configuration loading

#### ✅ **HPC Infrastructure**
- PBS job scripts
- Singularity definition
- Automation scripts
- Docker setup

## Sample Output

```
============================================================
HPC PRE-FLIGHT CHECK
============================================================

🔍 Environment Check...
✅ Environment Check - PASSED

🔍 Dependencies Check...
✅ Dependencies Check - PASSED

🔍 Code Syntax Check...
✅ Code Syntax Check - PASSED

🔍 Import Check...
✅ Import Check - PASSED

🔍 Model Initialization Check...
✅ Model Initialization Check - PASSED

🔍 Data Loading Check...
✅ Data Loading Check - PASSED

🔍 Positional Encoding Check...
✅ Positional Encoding Check - PASSED

🔍 Evaluation Metrics Check...
✅ Evaluation Metrics Check - PASSED

🔍 Training Pipeline Check...
✅ Training Pipeline Check - PASSED

🔍 HPC Scripts Check...
✅ HPC Scripts Check - PASSED

🔍 Docker/Singularity Check...
✅ Docker/Singularity Check - PASSED

============================================================
PRE-FLIGHT CHECK SUMMARY
============================================================

✅ PASSED CHECKS (11):
  • Environment Check
  • Dependencies Check
  • Code Syntax Check
  • Import Check
  • Model Initialization Check
  • Data Loading Check
  • Positional Encoding Check
  • Evaluation Metrics Check
  • Training Pipeline Check
  • HPC Scripts Check
  • Docker/Singularity Check

📊 OVERALL STATUS:
🎉 ALL CHECKS PASSED - Ready for HPC deployment!

============================================================

🚀 Ready for HPC deployment!
```

## Error Examples & Fixes

### ❌ **Missing Dependencies**
```
❌ Dependencies Check - FAILED
• Missing packages: torch, transformers
```
**Fix:** `pip install torch transformers datasets`

### ❌ **Import Errors**
```
❌ Import Check - FAILED
• src.model: No module named 'torch'
```
**Fix:** Check virtual environment and package installation

### ❌ **Syntax Errors**
```
❌ Code Syntax Check - FAILED
• src/model.py:45: invalid syntax
```
**Fix:** Check for missing colons, brackets, or indentation

### ❌ **Model Issues**
```
❌ Model Initialization Check - FAILED
• Model initialization failed: 'config' object has no attribute 'vocab_size'
```
**Fix:** Check config structure and model parameters

### ❌ **Data Loading Issues**
```
❌ Data Loading Check - FAILED
• Data loading failed: Dataset not found
```
**Fix:** Check internet connection and dataset availability

### ❌ **Missing HPC Scripts**
```
❌ HPC Scripts Check - FAILED
• Missing HPC scripts: submit_training.pbs, Singularity.def
```
**Fix:** Run the automation scripts to generate missing files

## When to Run

### **Before HPC Deployment**
```bash
# Run complete check
python hpc_preflight_check.py

# If successful, proceed with deployment
./setup_padum_automation.sh <username>
```

### **After Code Changes**
```bash
# Quick check after modifications
python hpc_preflight_check.py
```

### **Before Committing**
```bash
# Ensure everything works
python hpc_preflight_check.py
```

## Troubleshooting

### **Common Issues**

1. **Import Errors**
   - Check virtual environment activation
   - Verify package installation: `pip list`
   - Check Python path: `python -c "import sys; print(sys.path)"`

2. **Model Errors**
   - Check config structure in `src/config.py`
   - Verify model parameters match config
   - Test with smaller model first

3. **Data Loading Errors**
   - Check internet connection
   - Verify dataset names and splits
   - Test with smaller samples first

4. **HPC Script Errors**
   - Run automation scripts to generate missing files
   - Check file permissions: `chmod +x *.sh`
   - Verify script syntax: `bash -n script.sh`

### **Quick Fixes**

```bash
# Install missing packages
pip install -r requirements.txt

# Fix import issues
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Make scripts executable
chmod +x *.sh

# Test individual components
python -c "import src.model; print('Model OK')"
python -c "from data.math_dataset_loader import MathematicalDatasetLoader; print('Data OK')"
```

## Integration with CI/CD

Add to your deployment pipeline:
```yaml
# .github/workflows/preflight.yml
- name: HPC Pre-flight Check
  run: python hpc_preflight_check.py
```

## Exit Codes

- **0**: All checks passed - Ready for deployment
- **1**: Errors found - Fix before deployment

Use in scripts:
```bash
python hpc_preflight_check.py
if [ $? -eq 0 ]; then
    echo "Ready for HPC deployment"
    ./setup_padum_automation.sh <username>
else
    echo "Fix errors before deployment"
    exit 1
fi
```

## Performance Notes

- **Fast checks**: Environment, dependencies, syntax (~5 seconds)
- **Medium checks**: Imports, model init (~10 seconds)
- **Slow checks**: Data loading, positional encodings (~30 seconds)
- **Total time**: ~1-2 minutes

The script is designed to be comprehensive but fast enough for regular use. 