#!/usr/bin/env python3
"""
Adaptive Training Example for OpenMathInstruct-1M

This script demonstrates how to use the adaptive checkpointing system
for large-scale training with model comparison and random initialization.

Features:
- Saves models every 10K samples
- Compares performance and keeps best 5 models
- Randomly initializes from best models for next phase
- Handles large datasets efficiently
"""

import subprocess
import sys
from pathlib import Path

def run_adaptive_training():
    """Run adaptive training with OpenMathInstruct-1M."""
    
    # Training configuration
    config = {
        'pe_method': 'rope',  # or 't5_relative', 'alibi', etc.
        'datasets': 'openmath_instruct',
        'large_scale_training': True,
        'adaptive_checkpointing': True,
        'save_every_samples': 10000,
        'keep_best_models': 5,
        'max_train_samples': 800000,  # 80% of 1M
        'max_eval_samples': 200000,   # 20% of 1M
        'batch_size': 1,
        'gradient_accumulation_steps': 32,
        'max_steps': 50000,
        'learning_rate': 2e-5,
        'experiment_name': 'openmath_1m_adaptive',
        'checkpoint_dir': './checkpoints/openmath_1m_adaptive',
        'result_dir': './results/openmath_1m_adaptive',
        'memory_efficient': True,
        'random_seed': 42
    }
    
    # Build command
    cmd = [
        sys.executable, 'math_pe_research/scripts/train_and_eval.py',
        '--pe', config['pe_method'],
        '--datasets', config['datasets'],
        '--large_scale_training',
        '--adaptive_checkpointing',
        '--save_every_samples', str(config['save_every_samples']),
        '--keep_best_models', str(config['keep_best_models']),
        '--max_train_samples', str(config['max_train_samples']),
        '--max_eval_samples', str(config['max_eval_samples']),
        '--batch_size', str(config['batch_size']),
        '--gradient_accumulation_steps', str(config['gradient_accumulation_steps']),
        '--max_steps', str(config['max_steps']),
        '--learning_rate', str(config['learning_rate']),
        '--experiment_name', config['experiment_name'],
        '--checkpoint_dir', config['checkpoint_dir'],
        '--result_dir', config['result_dir'],
        '--memory_efficient',
        '--random_seed', str(config['random_seed'])
    ]
    
    print("🚀 Starting adaptive training with OpenMathInstruct-1M")
    print(f"📊 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print(f"\n🔧 Command: {' '.join(cmd)}")
    
    # Create directories
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['result_dir']).mkdir(parents=True, exist_ok=True)
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        return False
    
    return True

def run_phased_training():
    """Run training in phases with model selection between phases."""
    
    phases = [
        {
            'name': 'phase_1',
            'max_steps': 10000,
            'max_train_samples': 200000,
            'description': 'Initial training phase'
        },
        {
            'name': 'phase_2', 
            'max_steps': 15000,
            'max_train_samples': 400000,
            'description': 'Expanded training phase'
        },
        {
            'name': 'phase_3',
            'max_steps': 25000,
            'max_train_samples': 800000,
            'description': 'Full dataset training phase'
        }
    ]
    
    for i, phase in enumerate(phases, 1):
        print(f"\n🔄 Starting Phase {i}: {phase['description']}")
        print(f"   📊 Steps: {phase['max_steps']:,}")
        print(f"   📈 Samples: {phase['max_train_samples']:,}")
        
        # Build command for this phase
        cmd = [
            sys.executable, 'math_pe_research/scripts/train_and_eval.py',
            '--pe', 'rope',
            '--datasets', 'openmath_instruct',
            '--large_scale_training',
            '--adaptive_checkpointing',
            '--save_every_samples', '10000',
            '--keep_best_models', '5',
            '--max_train_samples', str(phase['max_train_samples']),
            '--max_eval_samples', str(phase['max_train_samples'] // 4),
            '--batch_size', '1',
            '--gradient_accumulation_steps', '32',
            '--max_steps', str(phase['max_steps']),
            '--learning_rate', '2e-5',
            '--experiment_name', f'openmath_1m_{phase["name"]}',
            '--checkpoint_dir', f'./checkpoints/openmath_1m_{phase["name"]}',
            '--result_dir', f'./results/openmath_1m_{phase["name"]}',
            '--memory_efficient',
            '--random_seed', '42'
        ]
        
        # Create directories
        Path(f'./checkpoints/openmath_1m_{phase["name"]}').mkdir(parents=True, exist_ok=True)
        Path(f'./results/openmath_1m_{phase["name"]}').mkdir(parents=True, exist_ok=True)
        
        print(f"🔧 Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(f"✅ Phase {i} completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Phase {i} failed with error: {e}")
            return False
    
    print("\n🎉 All training phases completed!")
    return True

def main():
    """Main function to run adaptive training examples."""
    
    print("🎯 Adaptive Training Examples for OpenMathInstruct-1M")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path('math_pe_research/scripts/train_and_eval.py').exists():
        print("❌ Error: Please run this script from the project root directory")
        return
    
    print("\nChoose training mode:")
    print("1. Single-phase adaptive training (800K samples)")
    print("2. Multi-phase training with model selection")
    print("3. Quick test (10K samples)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        print("\n🚀 Running single-phase adaptive training...")
        success = run_adaptive_training()
    elif choice == '2':
        print("\n🔄 Running multi-phase training...")
        success = run_phased_training()
    elif choice == '3':
        print("\n🧪 Running quick test...")
        # Modify config for quick test
        import subprocess
        cmd = [
            sys.executable, 'math_pe_research/scripts/train_and_eval.py',
            '--pe', 'rope',
            '--datasets', 'openmath_instruct',
            '--adaptive_checkpointing',
            '--save_every_samples', '5000',
            '--keep_best_models', '3',
            '--max_train_samples', '10000',
            '--max_eval_samples', '2000',
            '--batch_size', '1',
            '--gradient_accumulation_steps', '16',
            '--max_steps', '1000',
            '--experiment_name', 'quick_test',
            '--checkpoint_dir', './checkpoints/quick_test',
            '--result_dir', './results/quick_test',
            '--memory_efficient'
        ]
        success = subprocess.run(cmd, check=True, capture_output=False)
    else:
        print("❌ Invalid choice")
        return
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("📁 Check the checkpoint and result directories for outputs")
    else:
        print("\n❌ Training failed")

if __name__ == "__main__":
    main() 