#!/usr/bin/env python3
"""
Test script to verify the trainer's dataset loading process.
"""
import sys
import os
sys.path.append(os.path.abspath('.'))

def test_trainer_dataset_loading():
    """Test the trainer's dataset loading process."""
    print("=== Testing Trainer Dataset Loading ===")
    
    try:
        from training.mathematical_reasoning_trainer import MathematicalReasoningTrainer, get_mathematical_reasoning_config
        print("✓ MathematicalReasoningTrainer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MathematicalReasoningTrainer: {e}")
        return False
    
    try:
        # Get configuration
        config = get_mathematical_reasoning_config("sinusoidal")
        config['num_epochs'] = 1  # Minimal epochs for testing
        config['batch_size'] = 2
        config['model_size'] = 'small'
        
        # Update model size for small
        config['model'].update({
            'd_model': 256, 
            'n_heads': 8, 
            'd_ff': 1024, 
            'n_encoder_layers': 4, 
            'n_decoder_layers': 4
        })
        config['batch_size'] = 2
        config['eval_batch_size'] = 4
        
        print("✓ Configuration created successfully")
        print(f"Model config: {config['model']}")
    except Exception as e:
        print(f"✗ Failed to create configuration: {e}")
        return False
    
    try:
        # Initialize trainer
        print("Initializing trainer...")
        trainer = MathematicalReasoningTrainer(config)
        print("✓ Trainer initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize trainer: {e}")
        return False
    
    try:
        # Test dataset loading
        print("\n--- Testing Trainer Dataset Loading ---")
        trainer.load_datasets()
        
        # Check results
        if hasattr(trainer, 'train_problems'):
            print(f"Train problems loaded: {len(trainer.train_problems)}")
        else:
            print("No train_problems attribute found")
            
        if hasattr(trainer, 'test_problems'):
            print(f"Test problems loaded: {len(trainer.test_problems)}")
        else:
            print("No test_problems attribute found")
            
        if hasattr(trainer, 'train_loader') and trainer.train_loader is not None:
            print(f"Train loader created: {len(trainer.train_loader)} batches")
        else:
            print("No train_loader created")
            
        if hasattr(trainer, 'test_loader') and trainer.test_loader is not None:
            print(f"Test loader created: {len(trainer.test_loader)} batches")
        else:
            print("No test_loader created")
            
        # Check if datasets were loaded successfully
        if (hasattr(trainer, 'train_problems') and len(trainer.train_problems) > 0):
            print("✓ Trainer dataset loading successful")
            return True
        else:
            print("✗ Trainer dataset loading failed - no problems loaded")
            return False
            
    except Exception as e:
        print(f"✗ Trainer dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_trainer_dataset_loading()
    if success:
        print("✅ Trainer dataset loading is working correctly!")
    else:
        print("❌ Trainer dataset loading has issues that need to be fixed.")
        sys.exit(1) 