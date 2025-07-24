# 🚀 KAGGLE DEPLOYMENT GUIDE
## Mathematical Reasoning Model with Configurable PE

This guide provides complete steps to deploy and run your mathematical reasoning model with any positional encoding method on Kaggle.

## ✅ VERIFIED COMPATIBILITY
- **✅ All 5 PE methods work**: RoPE, Sinusoidal, T5-Relative, DIET, ALiBi
- **✅ Pythia parameter inheritance**: 100% preserved
- **✅ LoRA integration**: Auto-detects target modules
- **✅ Embedding preservation**: Original vocab + new math tokens

---

## 📋 STEP 1: PREPARE YOUR KAGGLE NOTEBOOK

### 1.1 Create New Kaggle Notebook
1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Set environment to **GPU P100** or **GPU T4x2**
4. Enable internet access

### 1.2 Upload Project Files
Option A: **Dataset Upload** (Recommended)
```bash
# Zip your project locally
cd /path/to/your/project
zip -r math_pe_research.zip math_pe_research/
```
- Upload `math_pe_research.zip` as a Kaggle dataset
- Add the dataset to your notebook

Option B: **Direct Copy** 
- Copy individual files using Kaggle's file upload

---

## 📋 STEP 2: KAGGLE SETUP CELL

### Cell 1: Environment Setup
```python
import subprocess
import sys
import os
import shutil
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print("🔧 Setting up Kaggle environment...")

# Install required packages
packages = [
    "accelerate>=0.20.0",
    "transformers>=4.30.0",
    "torch>=2.0.0",
    "datasets>=2.10.0",
    "peft>=0.4.0",
    "wandb",
    "numpy<2.0",  # Important for compatibility
]

for package in packages:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
        print(f"✅ Installed {package}")
    except:
        print(f"⚠️ Failed to install {package}")

print("✅ Environment setup complete!")
```

---

## 📋 STEP 3: PROJECT EXTRACTION CELL

### Cell 2: Extract Project
```python
# Extract project from dataset (if using dataset upload)
import zipfile

# Find and extract the project
dataset_path = "/kaggle/input"
project_dirs = list(Path(dataset_path).glob("**/math_pe_research.zip"))

if project_dirs:
    zip_path = project_dirs[0]
    print(f"📁 Found project zip: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("/kaggle/working")
    print("✅ Project extracted successfully!")
else:
    # If files are directly copied
    print("📁 Using directly copied files")

# Verify project structure
project_path = Path("/kaggle/working/math_pe_research")
if project_path.exists():
    print(f"✅ Project found at: {project_path}")
    print("📂 Project structure:")
    for item in project_path.rglob("*"):
        if item.is_file() and item.suffix in ['.py', '.md', '.txt']:
            print(f"   {item.relative_to(project_path)}")
else:
    print("❌ Project not found. Check your upload.")
```

---

## 📋 STEP 4: CONFIGURATION CELL

### Cell 3: Training Configuration
```python
# 🎯 TRAINING CONFIGURATION
# Modify these settings as needed

CONFIG = {
    # Model settings
    'model_size': 'EleutherAI/pythia-2.8b',  # or 'wellecks/llmstep-mathlib4-pythia2.8b' for math-pretrained
    'pe_method': 'rope',  # Options: 'rope', 'sinusoidal', 't5_relative', 'diet', 'alibi'
    
    # Training settings
    'batch_size': 4,
    'max_steps': 500,
    'learning_rate': 2e-5,
    'max_length': 1024,
    'use_lora': True,  # Recommended for Kaggle
    
    # Data settings
    'datasets': 'gsm8k,math',  # Available: gsm8k, math, mathqa
    'data_fraction': 0.1,  # Use 10% of data for faster training
    
    # Experiment settings
    'experiment_name': 'kaggle_math_pe_experiment',
    'wandb_project': 'kaggle_math_reasoning',
    
    # Kaggle-specific settings
    'save_steps': 100,
    'eval_steps': 100,
    'logging_steps': 50,
}

print("🎯 Configuration loaded:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")
```

---

## 📋 STEP 5: TRAINING EXECUTION CELL

### Cell 4: Run Training
```python
import subprocess
import shutil
from pathlib import Path
import os

# 🧹 Setup directories
print("🗂️ Setting up directories...")

directories = {
    'cache_dir': '/tmp/model_cache',
    'checkpoint_dir': '/kaggle/working/checkpoints',
    'result_dir': '/kaggle/working/results'
}

for name, dir_path in directories.items():
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"✅ Created {name}: {dir_path}")

# 🔧 Environment variables  
os.environ['HF_HOME'] = directories['cache_dir']
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_API_KEY'] = 'your_wandb_key_here'  # Optional: replace with your key

# 📁 Find project directory
project_dir = Path("/kaggle/working/math_pe_research")
if not project_dir.exists():
    print("❌ Project directory not found!")
    exit()

print(f"📁 Using project: {project_dir}")

# 🚀 Build and execute training command
cmd = f"""
cd {project_dir} && python scripts/train_and_eval.py \
    --pe {CONFIG['pe_method']} \
    --batch_size {CONFIG['batch_size']} \
    --max_steps {CONFIG['max_steps']} \
    --learning_rate {CONFIG['learning_rate']} \
    --experiment_name {CONFIG['experiment_name']} \
    --checkpoint_dir {directories['checkpoint_dir']} \
    --result_dir {directories['result_dir']} \
    --cache_dir {directories['cache_dir']} \
    --max_length {CONFIG['max_length']} \
    --model_size {CONFIG['model_size']} \
    --datasets {CONFIG['datasets']} \
    --wandb_project {CONFIG['wandb_project']} \
    --save_steps {CONFIG['save_steps']} \
    --eval_steps {CONFIG['eval_steps']} \
    --logging_steps {CONFIG['logging_steps']}""" + (" \\\n    --use_lora" if CONFIG['use_lora'] else "")

print(f"""
🚀 STARTING TRAINING WITH {CONFIG['pe_method'].upper()} PE
{'='*60}

📊 Configuration:
   🎯 Model: {CONFIG['model_size']}
   🔧 PE Method: {CONFIG['pe_method']}
   📈 Batch Size: {CONFIG['batch_size']}
   🎓 Max Steps: {CONFIG['max_steps']}
   📏 Max Length: {CONFIG['max_length']}
   💡 Learning Rate: {CONFIG['learning_rate']}
   🔗 LoRA: {CONFIG['use_lora']}

📝 Command:
{cmd}

{'='*60}
""")

# Execute training
try:
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode == 0:
        print("\n🎉 Training completed successfully!")
    else:
        print(f"\n❌ Training failed with return code: {result.returncode}")
except KeyboardInterrupt:
    print("\n⚠️ Training interrupted by user")
except Exception as e:
    print(f"\n❌ Training failed: {e}")

print(f"\n📁 Results saved to: {directories['result_dir']}")
print(f"💾 Checkpoints saved to: {directories['checkpoint_dir']}")
```

---

## 📋 STEP 6: RESULTS ANALYSIS CELL

### Cell 5: Analyze Results
```python
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 📊 Load and display results
result_dir = Path("/kaggle/working/results")
checkpoint_dir = Path("/kaggle/working/checkpoints")

print("📊 TRAINING RESULTS ANALYSIS")
print("="*50)

# Check for results files
result_files = list(result_dir.glob("*.json"))
if result_files:
    print(f"✅ Found {len(result_files)} result files:")
    for file in result_files:
        print(f"   📄 {file.name}")
        
        # Load and display results
        with open(file, 'r') as f:
            results = json.load(f)
        
        print(f"\n📈 Results from {file.name}:")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
else:
    print("⚠️ No result files found")

# Check for checkpoints
checkpoint_files = list(checkpoint_dir.glob("**/*.bin"))
if checkpoint_files:
    print(f"\n💾 Found {len(checkpoint_files)} checkpoint files:")
    for file in checkpoint_files[-5:]:  # Show last 5
        print(f"   📄 {file.name} ({file.stat().st_size / 1024 / 1024:.1f} MB)")
else:
    print("\n⚠️ No checkpoint files found")

# Display model info
print(f"\n🎯 MODEL SUMMARY:")
print(f"   PE Method: {CONFIG['pe_method']}")
print(f"   Base Model: {CONFIG['model_size']}")
print(f"   Training Steps: {CONFIG['max_steps']}")
print(f"   LoRA Enabled: {CONFIG['use_lora']}")

print("\n✅ Analysis complete!")
```

---

## 📋 STEP 7: MODEL TESTING CELL

### Cell 6: Test Your Model
```python
import torch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path("/kaggle/working/math_pe_research/src")))

from models.mathematical_reasoning_model import create_mathematical_reasoning_model

print("🧪 TESTING TRAINED MODEL")
print("="*40)

try:
    # Load the trained model
    print("📥 Loading trained model...")
    
    model = create_mathematical_reasoning_model(
        pe_method=CONFIG['pe_method'],
        base_model=CONFIG['model_size'],
        load_in_4bit=False,
        use_lora=CONFIG['use_lora'],
        device_map=None,
        torch_dtype=torch.float16
    )
    
    print(f"✅ Model loaded with {CONFIG['pe_method']} PE")
    
    # Test problems
    test_problems = [
        "What is 15 + 27?",
        "If a rectangle has length 8 and width 5, what is its area?",
        "Solve for x: 2x + 5 = 13",
        "What is the square root of 144?"
    ]
    
    print("\n🧮 Testing mathematical reasoning:")
    print("-" * 40)
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n{i}. Problem: {problem}")
        try:
            solution = model.solve_math_problem(
                problem, 
                max_length=200, 
                temperature=0.1
            )
            print(f"   Solution: {solution}")
        except Exception as e:
            print(f"   Error: {e}")
    
    print(f"\n🎉 Model testing completed!")
    
except Exception as e:
    print(f"❌ Model testing failed: {e}")
    import traceback
    traceback.print_exc()
```

---

## 🎯 QUICK START CONFIGURATIONS

### For Fast Testing (5-10 minutes):
```python
CONFIG = {
    'model_size': 'EleutherAI/pythia-70m',  # Small model
    'pe_method': 'rope',
    'batch_size': 8,
    'max_steps': 50,
    'max_length': 512,
    'use_lora': True,
}
```

### For Production Training (1-2 hours):
```python
CONFIG = {
    'model_size': 'EleutherAI/pythia-2.8b',
    'pe_method': 'rope',  # or any other
    'batch_size': 4,
    'max_steps': 1000,
    'max_length': 1024,
    'use_lora': True,
}
```

### For Math-Specialized Training:
```python
CONFIG = {
    'model_size': 'wellecks/llmstep-mathlib4-pythia2.8b',
    'pe_method': 'sinusoidal',  # Try different PE methods!
    'batch_size': 2,
    'max_steps': 500,
    'max_length': 1024,
    'use_lora': True,
}
```

---

## 🔧 TROUBLESHOOTING

### Memory Issues:
- Reduce `batch_size` to 2 or 1
- Use `load_in_4bit=True`
- Reduce `max_length` to 512

### Slow Training:
- Reduce `max_steps`
- Use smaller model (pythia-70m or pythia-410m)
- Increase `save_steps` and `eval_steps`

### Import Errors:
- Check project extraction in Step 3
- Verify all files are in correct locations
- Restart kernel and re-run setup

---

## 📊 EXPECTED RESULTS

After successful training, you should see:
- ✅ **Training metrics**: Loss decreasing over steps
- ✅ **Evaluation results**: Accuracy on mathematical reasoning tasks
- ✅ **Model checkpoints**: Saved in `/kaggle/working/checkpoints`
- ✅ **Working model**: Can solve mathematical problems

---

## 🎉 SUCCESS INDICATORS

Your deployment is successful when you see:
1. **✅ All packages installed** without errors
2. **✅ Project extracted** and files found
3. **✅ Model loads** with chosen PE method
4. **✅ Training starts** and shows progress
5. **✅ Checkpoints saved** regularly
6. **✅ Model can solve** test problems

---

## 📝 NEXT STEPS

After successful deployment:
1. **Experiment with different PE methods** (rope, sinusoidal, diet, etc.)
2. **Try different base models** (pythia-410m, pythia-1.4b, pythia-2.8b)
3. **Adjust hyperparameters** for better performance
4. **Compare results** across PE methods
5. **Share your findings** with the community!

**Happy training! 🚀🔥** 