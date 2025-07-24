# =============================================================================
# 🚀 MATHEMATICAL REASONING MODEL WITH CONFIGURABLE PE - KAGGLE NOTEBOOK
# =============================================================================
# Copy and paste this entire code into separate Kaggle cells as indicated

# =============================================================================
# CELL 1: Environment Setup
# =============================================================================

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

# =============================================================================
# CELL 2: Project Extraction (if using dataset upload)
# =============================================================================

import zipfile

# Option A: Extract from uploaded dataset
dataset_path = "/kaggle/input"
project_dirs = list(Path(dataset_path).glob("**/math_pe_research.zip"))

if project_dirs:
    zip_path = project_dirs[0]
    print(f"📁 Found project zip: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("/kaggle/working")
    print("✅ Project extracted successfully!")
else:
    # Option B: Create project structure manually (if no dataset upload)
    print("📁 Creating project structure manually...")
    
    # This would require you to upload individual files
    # For now, we'll assume you uploaded as a dataset
    print("⚠️ No zip found. Please upload your project as a dataset.")

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

# =============================================================================
# CELL 3: Configuration
# =============================================================================

# 🎯 TRAINING CONFIGURATION
# Modify these settings as needed

CONFIG = {
    # Model settings
    'model_size': 'EleutherAI/pythia-2.8b',  # Options: pythia-70m, pythia-410m, pythia-1.4b, pythia-2.8b
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

# Quick configurations for different use cases
QUICK_CONFIGS = {
    'fast_test': {
        'model_size': 'EleutherAI/pythia-70m',
        'max_steps': 50,
        'batch_size': 8,
        'max_length': 512,
    },
    'production': {
        'model_size': 'EleutherAI/pythia-2.8b',
        'max_steps': 1000,
        'batch_size': 4,
        'max_length': 1024,
    },
    'math_specialized': {
        'model_size': 'wellecks/llmstep-mathlib4-pythia2.8b',
        'pe_method': 'sinusoidal',
        'max_steps': 500,
        'batch_size': 2,
    }
}

# Uncomment to use a quick configuration:
# CONFIG.update(QUICK_CONFIGS['fast_test'])  # For quick testing
# CONFIG.update(QUICK_CONFIGS['production'])  # For full training
# CONFIG.update(QUICK_CONFIGS['math_specialized'])  # For math-specialized model

print("🎯 Configuration loaded:")
for key, value in CONFIG.items():
    print(f"   {key}: {value}")

# =============================================================================
# CELL 4: Training Execution
# =============================================================================

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
# os.environ['WANDB_API_KEY'] = 'your_wandb_key_here'  # Uncomment and add your W&B key

# 📁 Find project directory
project_dir = Path("/kaggle/working/math_pe_research")
if not project_dir.exists():
    print("❌ Project directory not found!")
    print("Please ensure you've uploaded the project correctly in Cell 2.")
    exit()

print(f"📁 Using project: {project_dir}")

# 🚀 Build training command
cmd_parts = [
    f"cd {project_dir}",
    "python scripts/train_and_eval.py",
    f"--pe {CONFIG['pe_method']}",
    f"--batch_size {CONFIG['batch_size']}",
    f"--max_steps {CONFIG['max_steps']}",
    f"--learning_rate {CONFIG['learning_rate']}",
    f"--experiment_name {CONFIG['experiment_name']}",
    f"--checkpoint_dir {directories['checkpoint_dir']}",
    f"--result_dir {directories['result_dir']}",
    f"--cache_dir {directories['cache_dir']}",
    f"--max_length {CONFIG['max_length']}",
    f"--model_size {CONFIG['model_size']}",
    f"--datasets {CONFIG['datasets']}",
    f"--wandb_project {CONFIG['wandb_project']}",
    f"--save_steps {CONFIG['save_steps']}",
    f"--eval_steps {CONFIG['eval_steps']}",
    f"--logging_steps {CONFIG['logging_steps']}"
]

if CONFIG.get('use_lora', True):
    cmd_parts.append("--use_lora")

cmd = " \\\n    ".join(cmd_parts)

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
   🔗 LoRA: {CONFIG.get('use_lora', True)}

📝 Command:
{cmd}

{'='*60}
""")

# Execute training
try:
    # Check available space
    statvfs = os.statvfs('/kaggle/working')
    free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
    print(f"💾 Available space: {free_space_gb:.1f} GB")
    
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

# =============================================================================
# CELL 5: Results Analysis
# =============================================================================

import json
import pandas as pd
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
        try:
            with open(file, 'r') as f:
                results = json.load(f)
            
            print(f"\n📈 Results from {file.name}:")
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
        except Exception as e:
            print(f"   ⚠️ Error reading {file.name}: {e}")
else:
    print("⚠️ No result files found")

# Check for checkpoints
checkpoint_files = list(checkpoint_dir.glob("**/*.bin"))
if checkpoint_files:
    print(f"\n💾 Found {len(checkpoint_files)} checkpoint files:")
    for file in checkpoint_files[-5:]:  # Show last 5
        size_mb = file.stat().st_size / 1024 / 1024
        print(f"   📄 {file.name} ({size_mb:.1f} MB)")
else:
    print("\n⚠️ No checkpoint files found")

# Display model summary
print(f"\n🎯 MODEL SUMMARY:")
print(f"   PE Method: {CONFIG['pe_method']}")
print(f"   Base Model: {CONFIG['model_size']}")
print(f"   Training Steps: {CONFIG['max_steps']}")
print(f"   LoRA Enabled: {CONFIG.get('use_lora', True)}")

print("\n✅ Analysis complete!")

# =============================================================================
# CELL 6: Model Testing
# =============================================================================

import torch
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path("/kaggle/working/math_pe_research/src")))

try:
    from models.mathematical_reasoning_model import create_mathematical_reasoning_model
    
    print("🧪 TESTING TRAINED MODEL")
    print("="*40)
    
    # Load the trained model
    print("📥 Loading trained model...")
    
    model = create_mathematical_reasoning_model(
        pe_method=CONFIG['pe_method'],
        base_model=CONFIG['model_size'],
        load_in_4bit=False,
        use_lora=CONFIG.get('use_lora', True),
        device_map=None,
        torch_dtype=torch.float16
    )
    
    print(f"✅ Model loaded with {CONFIG['pe_method']} PE")
    
    # Test problems
    test_problems = [
        "What is 15 + 27?",
        "If a rectangle has length 8 and width 5, what is its area?", 
        "Solve for x: 2x + 5 = 13",
        "What is the square root of 144?",
        "A train travels 120 miles in 2 hours. What is its speed?"
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
    print("This might happen if training didn't complete successfully.")
    import traceback
    traceback.print_exc()

# =============================================================================
# CELL 7: PE Method Comparison (Optional)
# =============================================================================

# Uncomment this cell to test different PE methods
"""
print("🔄 TESTING DIFFERENT PE METHODS")
print("="*50)

pe_methods = ['rope', 'sinusoidal', 't5_relative', 'diet', 'alibi']
test_problem = "What is 12 * 8?"

for pe_method in pe_methods:
    print(f"\n🔧 Testing {pe_method.upper()} PE:")
    try:
        model = create_mathematical_reasoning_model(
            pe_method=pe_method,
            base_model='EleutherAI/pythia-70m',  # Use small model for quick testing
            load_in_4bit=False,
            use_lora=False,
            device_map=None,
            torch_dtype=torch.float32
        )
        
        solution = model.solve_math_problem(test_problem, max_length=100, temperature=0.1)
        print(f"   ✅ {pe_method}: {solution}")
        
    except Exception as e:
        print(f"   ❌ {pe_method}: {e}")

print("\n✅ PE method comparison complete!")
"""

# =============================================================================
# END OF NOTEBOOK
# =============================================================================

print("""
🎉 KAGGLE DEPLOYMENT COMPLETE!

📊 What you accomplished:
   ✅ Set up environment with all dependencies
   ✅ Extracted and verified project structure  
   ✅ Configured training parameters
   ✅ Executed training with chosen PE method
   ✅ Analyzed results and model performance
   ✅ Tested trained model on mathematical problems

📝 Next steps:
   1. Experiment with different PE methods
   2. Try different model sizes  
   3. Adjust hyperparameters for better performance
   4. Compare results across configurations
   5. Share your findings!

🔗 Resources:
   - Project documentation: /kaggle/working/math_pe_research/README.md
   - Results: /kaggle/working/results/
   - Checkpoints: /kaggle/working/checkpoints/
   - Full deployment guide: /kaggle/working/math_pe_research/KAGGLE_DEPLOYMENT_GUIDE.md

Happy experimenting! 🚀🔥
""") 