# Mathematical Reasoning with Positional Encoding Research

A comprehensive research framework for comparing positional encoding methods in mathematical reasoning tasks using state-of-the-art models.

## Project Overview

This project investigates the impact of different positional encoding methods on mathematical reasoning performance using:
- **Base Model**: DeepSeekMath-Instruct-7B and DeepSeekMath-RL-7B
- **Tokenizer**: Mathematical reasoning optimized tokenizer
- **Datasets**: Large-scale mathematical problem datasets (MATH, GSM8K, OpenMathInstruct)
- **Positional Encodings**: Sinusoidal, RoPE, ALiBi, DIET, T5-Relative

## Architecture

```
Mathematical Reasoning Model
├── Tokenizer (Math-optimized)
├── Embedding Layer
├── Positional Encoding (Variable)
│   ├── Sinusoidal
│   ├── RoPE (Rotary)
│   ├── ALiBi (Attention with Linear Biases)
│   ├── DIET (Dynamic Encoding)
│   └── T5-Relative
├── Transformer Layers (from DeepSeekMath)
└── Mathematical Reasoning Head
```

## Project Structure

```
math_pe_research/
├── src/
│   ├── models/           # Model architecture
│   ├── positional_encoding/  # PE implementations
│   ├── data/            # Data loading and processing
│   ├── training/        # Training loops and utilities
│   └── utils/           # Helper functions
├── configs/             # Experiment configurations
├── scripts/             # Automation scripts
├── experiments/         # Results and checkpoints
└── docs/               # Documentation
```

## Quick Start

1. **Setup Environment**:
   ```bash
   ./scripts/setup_environment.sh
   ```

2. **Run Experiment**:
   ```bash
   ./scripts/run_experiment.sh --pe_method rope --node_id 0
   ```

3. **Monitor Training**:
   ```bash
   ./scripts/monitor_training.sh
   ```

## Supported Models

- DeepSeekMath-Instruct-7B (Primary)
- DeepSeekMath-RL-7B (RL-trained variant)
- Custom mathematical reasoning architectures

## Key Features

- **SOTA Integration**: Latest DeepSeekMath models
- **Optimized Tokenization**: Mathematical symbol-aware tokenizer
- **Multiple PE Methods**: Comprehensive positional encoding comparison
- **Large-scale Data**: OpenMathInstruct-1M+ dataset support
- **HPC Ready**: Distributed training on IITD cluster
- **Comprehensive Evaluation**: Mathematical reasoning metrics

## Research Goals

1. Compare positional encoding effectiveness in mathematical reasoning
2. Identify optimal PE methods for different problem types
3. Analyze attention patterns across PE variants
4. Evaluate scaling behavior with sequence length
5. Develop improved PE methods for mathematical tasks