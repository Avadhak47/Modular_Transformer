#!/usr/bin/env python3
"""
Robust fix script for transformers import issues on Kaggle.
Handles all dependency conflicts and ensures compatibility.
"""

import subprocess
import sys
import os

def robust_transformers_fix():
    """Robust fix for transformers import issues with dependency resolution."""
    
    print("🔧 Starting robust transformers fix...")
    
    # Set environment variables
    os.environ['BITSANDBYTES_DISABLE'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    try:
        # Step 1: Uninstall conflicting packages
        print("📦 Uninstalling conflicting packages...")
        packages_to_remove = [
            "transformers",
            "peft", 
            "accelerate",
            "safetensors"
        ]
        
        for package in packages_to_remove:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package])
                print(f"✅ Uninstalled: {package}")
            except:
                print(f"⚠️  Could not uninstall: {package}")
        
        # Step 2: Install compatible versions in correct order
        print("📦 Installing compatible versions...")
        
        # First, install core dependencies
        core_packages = [
            "safetensors>=0.4.3",
            "tokenizers>=0.15.0",
            "huggingface-hub>=0.19.0"
        ]
        
        for package in core_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", package])
            print(f"✅ Installed core: {package}")
        
        # Then install transformers with specific version
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==4.35.0"])
        print("✅ Installed transformers==4.35.0")
        
        # Finally install other packages
        other_packages = [
            "peft==0.7.0",
            "accelerate==0.25.0",
            "datasets==2.15.0",
            "wandb==0.16.0"
        ]
        
        for package in other_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed: {package}")
        
        # Step 3: Test imports
        print("🧪 Testing imports...")
        
        from transformers import AutoTokenizer, TrainingArguments, Trainer
        from transformers.utils.import_utils import is_torch_available, is_peft_available
        from peft import LoraConfig, get_peft_model
        import accelerate
        
        print("✅ All imports successful!")
        
        # Step 4: Verify versions
        import transformers
        import safetensors
        print(f"✅ Transformers version: {transformers.__version__}")
        print(f"✅ Safetensors version: {safetensors.__version__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Robust fix failed: {e}")
        print("Trying fallback approach...")
        
        # Fallback: Use latest compatible versions
        try:
            fallback_packages = [
                "safetensors>=0.4.3",
                "transformers>=4.35.0,<4.40.0",
                "peft>=0.7.0",
                "accelerate>=0.25.0",
                "datasets>=2.15.0",
                "wandb>=0.16.0"
            ]
            
            for package in fallback_packages:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", package])
                print(f"✅ Installed fallback: {package}")
            
            # Test imports
            from transformers import AutoTokenizer, TrainingArguments, Trainer
            print("✅ Fallback imports successful!")
            
            return True
            
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            return False

def test_training_imports():
    """Test all imports needed for training."""
    
    print("🧪 Testing training imports...")
    
    try:
        # Test all required imports
        from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
        from peft import LoraConfig, get_peft_model
        import torch
        import wandb
        
        print("✅ All training imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Training imports failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting robust transformers fix...")
    
    # Run the fix
    success = robust_transformers_fix()
    
    if success:
        # Test training imports
        training_success = test_training_imports()
        
        if training_success:
            print("\n🎉 Robust transformers fix completed successfully!")
            print("✅ All imports working correctly")
            print("✅ Ready for training")
        else:
            print("\n⚠️  Fix completed but training imports failed")
    else:
        print("\n❌ Robust fix failed")
        print("Please try running the fix script again or check the error messages above.") 