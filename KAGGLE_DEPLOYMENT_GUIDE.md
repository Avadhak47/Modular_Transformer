# 🚀 Kaggle Deployment Guide for Mathematical Reasoning Model

This guide walks you through deploying and training your mathematical reasoning model on Kaggle, addressing all compatibility issues.

## 🚨 Critical Fixes Applied

### ✅ **NumPy 2.x Compatibility Crisis - SOLVED**
- **Problem**: Kaggle upgraded to NumPy 2.3.1, breaking all ML libraries
- **Solution**: Pin to `numpy<2.0` in requirements and setup scripts
- **Status**: Fixed in requirements.txt and kaggle_setup.py

### ✅ **LoRA/Triton Compilation Error - SOLVED** 
- **Problem**: `torchao` library causing `NameError: name 'int32' is not defined`
- **Solution**: LoRA disabled by default on Kaggle, can be enabled with `--use_lora`
- **Status**: Fixed in train_and_eval.py with Kaggle detection

### ✅ **DataCollator Padding Error - SOLVED**
- **Problem**: Variable tensor lengths causing batch creation failures
- **Solution**: Replaced with `DataCollatorForLanguageModeling`
- **Status**: Fixed in train_and_eval.py and dataset loader

---

## 📋 Step-by-Step Deployment

### 1. 🔧 **Kaggle Notebook Setup**

**Create a new Kaggle notebook with GPU enabled:**
1. Go to [Kaggle.com](https://kaggle.com) → Notebooks → New Notebook
2. **CRITICAL**: Settings → Accelerator → **GPU T4 x2** (must enable GPU!)
3. Settings → Internet → **On** (for downloading models)

### 2. 📦 **Environment Setup (MUST RUN FIRST)**

**Copy-paste this into your first cell:**

```python
# CRITICAL: NumPy 2.x Compatibility Fix (Run this FIRST!)
import subprocess
import sys
import os

def install_compatible_packages():
    """Install packages with NumPy 1.x compatibility."""
    packages = [
        "'numpy<2.0'",  # CRITICAL: avoid NumPy 2.x
        "'matplotlib<3.8.0'",  # Pin for numpy compatibility
        "'transformers>=4.35.0,<4.46.0'",
        "'accelerate>=0.25.0'",
        "'peft>=0.7.0'",
        "'datasets>=2.15.0'",
        "'wandb>=0.16.0'",
        "'scikit-learn>=1.3.0'",
        "'pandas>=2.0.0'",
        "'tqdm>=4.66.0'"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package, "--upgrade", "--quiet"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {package} installed successfully")
        else:
            print(f"❌ Failed to install {package}: {result.stderr}")

# Install compatible packages
install_compatible_packages()

# Set environment variables
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_SILENT'] = 'true'
os.environ['HF_HOME'] = '/kaggle/working/data_cache'
os.environ['TRANSFORMERS_CACHE'] = '/kaggle/working/data_cache'

print("\n🎉 Environment setup completed!")
print("⚠️  If you see NumPy 2.x warnings, restart the kernel and run this cell again.")
```

**⚠️ IMPORTANT**: If you get NumPy warnings, **restart the kernel** and run this cell again!

### 3. 📁 **Upload Your Project**

**Option A: Upload as Dataset (Recommended)**
1. Zip your entire project directory
2. Kaggle → Datasets → New Dataset → Upload zip file
3. Add dataset to your notebook
4. Extract with:

```python
import zipfile
from pathlib import Path

# Find and extract your project zip
input_path = Path('/kaggle/input')
zip_files = list(input_path.glob('**/*.zip'))

if zip_files:
    project_zip = zip_files[0]
    print(f"Found project: {project_zip}")
    
    with zipfile.ZipFile(project_zip, 'r') as zip_ref:
        zip_ref.extractall('/kaggle/working/')
    
    print("✅ Project extracted successfully!")
else:
    print("❌ No zip files found. Please upload your project as a dataset.")
```

**Option B: Copy-Paste Files**
- Manually create the directory structure
- Copy-paste file contents into new cells

### 4. 🗂️ **Setup Directories**

```python
from pathlib import Path

# Create required directories
directories = [
    '/kaggle/working/checkpoints',
    '/kaggle/working/evaluation_results',
    '/kaggle/working/data_cache',
    '/kaggle/working/logs'
]

for dir_path in directories:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"✅ Created: {dir_path}")
```

### 5. 🔍 **Verify GPU and Environment**

```python
import torch

# Check GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🚀 GPU Available: {gpu_name} ({gpu_memory:.1f} GB)")
else:
    print("❌ NO GPU! Enable GPU in Settings → Accelerator → GPU T4 x2")

# Find project directory
project_dirs = list(Path('/kaggle/working').glob('**/math_pe_research'))
if project_dirs:
    print(f"✅ Project found: {project_dirs[0]}")
else:
    print("❌ Project not found. Check extraction step.")
```

### 6. 🔑 **W&B Authentication (Optional)**

```python
import wandb

# Option 1: Login with API key
# wandb.login(key="your-wandb-api-key-here")

# Option 2: Interactive login
try:
    wandb.login()
    print("✅ W&B authenticated!")
except:
    print("⚠️  W&B login failed - training will work without logging")
```

### 7. 🚀 **Start Training**

```python
import subprocess
from pathlib import Path

# Find your project directory
project_dir = list(Path('/kaggle/working').glob('**/math_pe_research'))[0]

# Training configuration
config = {
    'pe_method': 'rope',  # rope, alibi, sinusoidal, diet, t5_relative, math_adaptive
    'batch_size': 4,      # Reduce to 2 if OOM
    'max_steps': 500,     # Reduce for testing
    'learning_rate': 2e-5,
    'experiment_name': 'kaggle_math_reasoning',
    'max_length': 2048,   # Reduce if OOM
}

# Build and execute training command
cmd = f"""
cd {project_dir} && python scripts/train_and_eval.py \
    --pe {config['pe_method']} \
    --batch_size {config['batch_size']} \
    --max_steps {config['max_steps']} \
    --learning_rate {config['learning_rate']} \
    --experiment_name {config['experiment_name']} \
    --checkpoint_dir /kaggle/working/checkpoints \
    --result_dir /kaggle/working/evaluation_results \
    --cache_dir /kaggle/working/data_cache \
    --max_length {config['max_length']} \
    --wandb_project math_pe_research
""".strip()

print("🚀 Starting training...")
print(f"Command: {cmd}")
print("="*80)

# Execute training
subprocess.run(cmd, shell=True)
```

---

## 🔧 Troubleshooting

### **NumPy 2.x Errors**
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.3.1...
```
**Solution**: Restart kernel → Run environment setup cell again

### **Out of Memory (OOM)**
```
RuntimeError: CUDA out of memory...
```
**Solutions**:
- Reduce `batch_size` to 2 or 1
- Reduce `max_length` to 1024
- Use smaller model: `--model_size microsoft/DialoGPT-medium`

### **LoRA/Triton Compilation Error**
```
NameError: name 'int32' is not defined
```
**Solution**: LoRA is disabled by default. Add `--use_lora` only if needed.

### **Import/Module Errors**
```
ModuleNotFoundError: No module named 'positional_encoding'
```
**Solutions**:
- Check project extraction
- Verify directory structure
- Ensure all files are uploaded

### **No GPU Available**
**Solution**: 
1. Kaggle Settings → Accelerator → **GPU T4 x2**
2. Save and restart notebook

---

## ⚡ Performance Tips

### **For Quick Testing:**
```bash
--max_steps 100 \
--batch_size 2 \
--max_length 1024 \
--model_size microsoft/DialoGPT-medium
```

### **For Full Training:**
```bash
--max_steps 2000 \
--batch_size 4 \
--max_length 2048 \
--use_lora  # Only if no compilation errors
```

### **Memory Optimization:**
- Use gradient accumulation: automatic in script
- Enable fp16: automatic in script
- Reduce sequence length: `--max_length 1024`

---

## 📊 Expected Output

**Successful run should show:**
```
🔍 Kaggle environment detected - applying compatibility settings...
   ✅ LoRA disabled (use --use_lora to force enable)
wandb: Syncing run kaggle_math_reasoning
Loading checkpoint shards: 100%|██████████████████| 2/2 [00:46<00:00, 23.16s/it]
Starting training...
  0%|                    | 0/500 [00:00<?, ?it/s]
```

**Training progress:**
- Model downloads (~13GB for DeepSeek-7B)
- Dataset loading (GSM8K, MATH)
- Training steps with loss decreasing
- Evaluation every 250 steps
- Checkpoints saved every 500 steps

**Final outputs:**
- `/kaggle/working/checkpoints/` - Model checkpoints
- `/kaggle/working/evaluation_results/` - Evaluation metrics
- W&B dashboard - Training curves and metrics

---

## 🎯 Key Differences from Local Training

| Feature | Local | Kaggle |
|---------|-------|--------|
| **LoRA** | Enabled | Disabled (compilation issues) |
| **4-bit Quantization** | Optional | Disabled (compatibility) |
| **NumPy Version** | Any | Pinned to <2.0 |
| **Flash Attention** | Optional | Removed (compilation) |
| **GPU Memory** | Variable | ~15GB (T4 x2) |
| **Persistent Storage** | Yes | Only /kaggle/working |

---

## ✅ Success Checklist

- [ ] Kaggle notebook created with **GPU enabled**
- [ ] Environment setup completed **without NumPy warnings**
- [ ] Project uploaded and **extracted successfully**
- [ ] GPU detected and **memory sufficient**
- [ ] Training started **without import errors**
- [ ] W&B logging working (optional)
- [ ] Checkpoints being saved to `/kaggle/working/checkpoints/`

---

## 🆘 Need Help?

1. **Check logs**: Look for error messages in notebook output
2. **Verify setup**: Run the verification cells
3. **Restart fresh**: Sometimes a clean restart fixes issues
4. **Reduce complexity**: Start with smaller models/steps
5. **Check versions**: Ensure NumPy < 2.0

**Common Success Pattern:**
1. Fresh notebook → Environment setup → Restart kernel
2. Re-run environment setup → Upload project → Verify GPU
3. Start training with small config → Scale up if successful

This deployment approach has been tested and resolves all known compatibility issues with Kaggle's current environment! 🎉 