# Kaggle Mathematical Reasoning Training Guide

## 🚀 Quick Start

### Option 1: Use the Complete Notebook
Run the complete notebook that handles everything automatically:

```python
# Copy and paste this into a Kaggle notebook cell
exec(open('/kaggle/working/Transformer/math_pe_research/kaggle_fixed_notebook.py').read())
```

### Option 2: Run Individual Cells

#### Cell 1: Environment Setup
```python
# Run the environment setup
setup_environment()
```

#### Cell 2: Training Parameters
```python
# View training parameters and explanations
config = TrainingConfig()
config.print_parameter_guide()
config.print_parameter_explanations()
```

#### Cell 3: Start Training
```python
# Start the training process
training_success = start_training()
```

#### Cell 4: Model Evaluation
```python
# Evaluate the trained model
if training_success:
    evaluation_success = evaluate_model()
```

#### Cell 5: Visualize Results
```python
# Visualize training results
if training_success:
    visualization_success = visualize_results()
```

## 🔧 Configuration Options

### Model Parameters
- `base_model`: Base model to fine-tune (default: 'deepseek-ai/deepseek-math-7b-instruct')
- `pe_method`: Positional encoding method ('rope', 'sinusoidal', 'alibi', 't5_relative', 'diet')
- `use_lora`: Enable LoRA for parameter-efficient fine-tuning (default: True)
- `load_in_4bit`: 4-bit quantization (DISABLED due to bitsandbytes issues)

### Dataset Parameters
- `datasets`: Dataset name ('openmath_instruct', 'math_dataset', 'gsm8k')
- `large_scale_training`: Enable optimizations for large datasets (default: True)
- `max_train_samples`: Limit training samples (default: 100000)
- `max_eval_samples`: Limit evaluation samples (default: 20000)
- `streaming`: Enable streaming for large datasets (default: True)

### Training Parameters
- `num_train_epochs`: Number of training epochs (default: 3)
- `per_device_train_batch_size`: Batch size per device (default: 2)
- `gradient_accumulation_steps`: Steps to accumulate gradients (default: 8)
- `learning_rate`: Learning rate for optimization (default: 2e-4)
- `warmup_steps`: Steps for learning rate warmup (default: 100)
- `fp16`: Enable mixed precision training (default: True)

### Adaptive Checkpointing
- `adaptive_checkpointing`: Enable intelligent model saving (default: True)
- `save_every_samples`: Save model every N samples (default: 10000)
- `keep_best_models`: Number of best models to keep (default: 5)

## 📊 Monitoring

### WandB Integration
The training automatically logs to Weights & Biases:
- Project: `kaggle_math_reasoning`
- Run name: `math_pe_experiment`

### Output Files
- Models saved to: `/kaggle/working/Transformer/math_pe_research/outputs/`
- Checkpoints: `/kaggle/working/Transformer/math_pe_research/checkpoints/`
- Logs: `/kaggle/working/Transformer/math_pe_research/logs/`
- Training curves: `/kaggle/working/Transformer/math_pe_research/outputs/training_curves.png`

## 🛠️ Troubleshooting

### Common Issues

1. **Bitsandbytes CUDA Error**: Already fixed by disabling 4-bit quantization
2. **Memory Issues**: Reduce batch size or enable gradient accumulation
3. **Import Errors**: All dependencies are automatically installed
4. **GPU Issues**: The script automatically detects and uses available GPU

### Performance Tips

1. **For Large Datasets**: Enable `large_scale_training` and `streaming`
2. **For Memory Constraints**: Reduce `per_device_train_batch_size` and increase `gradient_accumulation_steps`
3. **For Faster Training**: Use `fp16` mixed precision
4. **For Better Results**: Use RoPE positional encoding for mathematical tasks

## 📈 Expected Results

With the default configuration:
- **Training Time**: ~2-4 hours on Kaggle P100 GPU
- **Memory Usage**: ~12-14 GB GPU memory
- **Dataset Size**: 100K training samples, 20K evaluation samples
- **Model Size**: ~7B parameters with LoRA adaptation
- **Expected Loss**: Training loss should decrease from ~2.5 to ~0.9

## 🔄 Customization

To modify training parameters, edit the `TrainingConfig` class:

```python
config = TrainingConfig()
config.model_config['pe_method'] = 'alibi'  # Change PE method
config.training_config['num_train_epochs'] = 5  # Change epochs
config.dataset_config['max_train_samples'] = 50000  # Change dataset size
```

## 📝 Notes

- The notebook automatically handles all CUDA compatibility issues
- 4-bit quantization is disabled to avoid bitsandbytes issues
- Adaptive checkpointing saves the best 5 models and randomly selects from them
- All outputs are saved to Kaggle's working directory
- The script is optimized for Kaggle's Tesla P100 GPU environment 