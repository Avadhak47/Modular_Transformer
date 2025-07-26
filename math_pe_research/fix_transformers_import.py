#!/usr/bin/env python3
"""
Fix script for transformers import issues on Kaggle.
"""

import subprocess
import sys
import os

def fix_transformers_import():
    """Fix transformers import issues by installing compatible versions."""
    
    print("Fixing transformers import issues...")
    
    # Set environment variables
    os.environ['BITSANDBYTES_DISABLE'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    try:
        # Uninstall current transformers to avoid conflicts
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "transformers"])
        print("✅ Uninstalled current transformers")
        
        # Install compatible versions with correct dependencies
        packages = [
            "safetensors>=0.4.3",  # Update safetensors first
            "transformers==4.35.0",
            "peft==0.7.0",
            "accelerate==0.25.0",
            "datasets==2.15.0",
            "wandb==0.16.0"
        ]
        
        for package in packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", package])
            print(f"✅ Installed: {package}")
        
        # Test imports
        from transformers import AutoTokenizer, TrainingArguments, Trainer
        print("✅ Transformers imports successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Fix failed: {e}")
        print("Trying alternative approach...")
        
        # Alternative: Install latest compatible versions
        try:
            alt_packages = [
                "safetensors>=0.4.3",
                "transformers>=4.35.0,<4.40.0",
                "peft>=0.7.0",
                "accelerate>=0.25.0"
            ]
            
            for package in alt_packages:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
                print(f"✅ Installed alternative: {package}")
            
            # Test imports again
            from transformers import AutoTokenizer, TrainingArguments, Trainer
            print("✅ Transformers imports successful with alternative approach!")
            
            return True
            
        except Exception as e2:
            print(f"❌ Alternative approach also failed: {e2}")
            return False

if __name__ == "__main__":
    success = fix_transformers_import()
    if success:
        print("\n🎉 Transformers import issue fixed!")
        print("You can now run your training script.")
    else:
        print("\n❌ Failed to fix transformers import issue.")
        print("Please try running the fix script again.") 