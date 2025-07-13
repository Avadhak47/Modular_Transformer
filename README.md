# Modular Transformer Architecture

A comprehensive, modular implementation of the Transformer architecture from "Attention is All You Need" with interchangeable positional encoding methods.

## Features

- **Modular Design**: Easy to swap positional encoding methods
- **Multiple Positional Encodings**: Sinusoidal, RoPE, ALiBi, DIET, T5-relative, NoPE
- **Research-Ready**: Comprehensive training and evaluation framework
- **Extensible**: Clean architecture for adding new components
- **Production-Quality**: Proper error handling, logging, and testing

## Project Structure

```
modular_transformer/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── main.py                      # Main entry point
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── config.py                    # Configuration management
├── src/
│   └── transformer/
│       ├── __init__.py         # Package initialization
│       ├── model.py            # Main transformer model
│       ├── config.py           # Model configuration
│       ├── layers/
│       │   ├── __init__.py     # Layer package init
│       │   ├── attention.py    # Multi-head attention
│       │   ├── feed_forward.py # Feed-forward network
│       │   ├── layer_norm.py   # Layer normalization
│       │   ├── encoder.py      # Encoder layers
│       │   ├── decoder.py      # Decoder layers
│       │   └── embedding.py    # Embedding layers
│       ├── positional_encoding/
│       │   ├── __init__.py     # PE package init
│       │   ├── base.py         # Base PE class
│       │   ├── sinusoidal.py   # Sinusoidal PE
│       │   ├── rope.py         # Rotary PE
│       │   ├── alibi.py        # ALiBi PE
│       │   ├── diet.py         # DIET PE
│       │   ├── t5_relative.py  # T5 relative PE
│       │   └── nope.py         # No PE
│       └── utils/
│           ├── __init__.py     # Utils package init
│           ├── mask_utils.py   # Masking utilities
│           ├── training_utils.py # Training helpers
│           └── metrics.py      # Evaluation metrics
├── tests/
│   ├── __init__.py            # Test package init
│   ├── test_model.py          # Model tests
│   ├── test_positional_encoding.py # PE tests
│   └── test_layers.py         # Layer tests
├── examples/
│   ├── __init__.py            # Examples package init
│   ├── basic_training.py      # Basic training example
│   └── positional_encoding_comparison.py # PE comparison
└── docs/
    ├── architecture.md         # Architecture documentation
    ├── positional_encodings.md # PE documentation
    └── training_guide.md       # Training guide
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/modular-transformer.git
cd modular-transformer

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
from src.transformer.model import TransformerModel

# Create model with default configuration
config = {
    "d_model": 512,
    "n_heads": 8,
    "d_ff": 2048,
    "n_encoder_layers": 6,
    "n_decoder_layers": 6,
    "vocab_size": 32000,
    "max_seq_len": 512,
    "dropout": 0.1,
    "positional_encoding": "sinusoidal"
}

model = TransformerModel(config)

# Switch positional encoding
model.switch_positional_encoding("rope")
```

### Training

```bash
# Train with sinusoidal positional encoding
python train.py --config sinusoidal --epochs 10

# Train with RoPE
python train.py --config rope --epochs 10

# Compare different positional encodings
python examples/positional_encoding_comparison.py
```

### Evaluation

```bash
# Evaluate trained model
python evaluate.py --model_path checkpoints/model.pt --data_path data/test.txt
```

## File Descriptions

### Core Files

- **`main.py`**: Entry point for the application, demonstrates basic usage
- **`train.py`**: Comprehensive training script with support for different PE methods
- **`evaluate.py`**: Evaluation script for trained models
- **`config.py`**: Configuration management and hyperparameter definitions

### Source Code (`src/transformer/`)

- **`model.py`**: Main transformer model implementation
- **`config.py`**: Model-specific configuration classes

### Layers (`src/transformer/layers/`)

- **`attention.py`**: Multi-head attention mechanism
- **`feed_forward.py`**: Position-wise feed-forward networks
- **`layer_norm.py`**: Layer normalization implementation
- **`encoder.py`**: Transformer encoder layers and stack
- **`decoder.py`**: Transformer decoder layers and stack
- **`embedding.py`**: Token and positional embedding layers

### Positional Encodings (`src/transformer/positional_encoding/`)

- **`base.py`**: Abstract base class for all positional encodings
- **`sinusoidal.py`**: Original sinusoidal positional encoding
- **`rope.py`**: Rotary Positional Encoding (RoPE)
- **`alibi.py`**: Attention with Linear Biases (ALiBi)
- **`diet.py`**: Decoupled Positional Attention (DIET)
- **`t5_relative.py`**: T5-style relative positional encoding
- **`nope.py`**: No positional encoding baseline

### Utilities (`src/transformer/utils/`)

- **`mask_utils.py`**: Masking functions for attention
- **`training_utils.py`**: Training helper functions
- **`metrics.py`**: Evaluation metrics and logging

### Tests (`tests/`)

- **`test_model.py`**: Unit tests for the main model
- **`test_positional_encoding.py`**: Tests for all PE methods
- **`test_layers.py`**: Tests for individual layers

### Examples (`examples/`)

- **`basic_training.py`**: Simple training example
- **`positional_encoding_comparison.py`**: Compare different PE methods

## Positional Encoding Methods

### 1. Sinusoidal (Default)
Original positional encoding from "Attention is All You Need"
```python
model = TransformerModel(config)  # Default is sinusoidal
```

### 2. RoPE (Rotary Positional Encoding)
Rotary positional encoding for better length extrapolation
```python
config["positional_encoding"] = "rope"
model = TransformerModel(config)
```

### 3. ALiBi (Attention with Linear Biases)
Linear biases in attention for improved length generalization
```python
config["positional_encoding"] = "alibi"
model = TransformerModel(config)
```

### 4. DIET (Decoupled Positional Attention)
Separate positional encodings per attention head
```python
config["positional_encoding"] = "diet"
model = TransformerModel(config)
```

### 5. T5-Relative
T5-style bucketed relative positional encoding
```python
config["positional_encoding"] = "t5_relative"
model = TransformerModel(config)
```

### 6. NoPE (No Positional Encoding)
Baseline without explicit positional encoding
```python
config["positional_encoding"] = "nope"
model = TransformerModel(config)
```

## Training Different Positional Encodings

### Single Method Training
```bash
python train.py --pe_type sinusoidal --epochs 20 --lr 1e-4
python train.py --pe_type rope --epochs 20 --lr 1e-4
python train.py --pe_type alibi --epochs 20 --lr 1e-4
```

### Comparison Training
```bash
python examples/positional_encoding_comparison.py --epochs 10
```

## Configuration

### Model Configuration
```python
config = {
    "d_model": 512,          # Model dimension
    "n_heads": 8,            # Number of attention heads
    "d_ff": 2048,           # Feed-forward dimension
    "n_encoder_layers": 6,   # Number of encoder layers
    "n_decoder_layers": 6,   # Number of decoder layers
    "vocab_size": 32000,     # Vocabulary size
    "max_seq_len": 512,      # Maximum sequence length
    "dropout": 0.1,          # Dropout rate
    "positional_encoding": "sinusoidal"  # PE method
}
```

### Training Configuration
```python
training_config = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "warmup_steps": 4000,
    "max_steps": 100000,
    "gradient_clip": 1.0,
    "optimizer": "adam",
    "weight_decay": 0.01
}
```

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_model.py

# Run with coverage
python -m pytest tests/ --cov=src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your implementation
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{modular_transformer,
  title={Modular Transformer: A Comprehensive Implementation with Interchangeable Positional Encodings},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/modular-transformer}
}
```

## Acknowledgments

- Original Transformer paper: "Attention is All You Need"
- RoPE: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- ALiBi: "Train Short, Test Long: Attention with Linear Biases"
- DIET: "Decoupled Positional Attention for Transformer"