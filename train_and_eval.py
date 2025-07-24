#!/usr/bin/env python3
"""
Wrapper script for mathematical reasoning model training and evaluation.
This script ensures proper path setup regardless of where it's called from.
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    # Get the path to the actual training script
    script_dir = Path(__file__).parent
    actual_script = script_dir / "math_pe_research" / "scripts" / "train_and_eval.py"
    
    if not actual_script.exists():
        print(f"Error: Training script not found at {actual_script}")
        sys.exit(1)
    
    # Change to the math_pe_research directory for proper imports
    math_pe_dir = script_dir / "math_pe_research"
    os.chdir(math_pe_dir)
    
    # Execute the actual training script with all passed arguments
    cmd = [sys.executable, "scripts/train_and_eval.py"] + sys.argv[1:]
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {math_pe_dir}")
    
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main() 