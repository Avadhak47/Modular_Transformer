# Mathematical Reasoning with Advanced Positional Encoding

## 🧮 Project Overview

A comprehensive framework for training and evaluating transformer models on mathematical reasoning tasks using state-of-the-art positional encoding techniques and the latest DeepSeekMath models.

## 🎯 Key Features

- **SOTA Base Models**: DeepSeekMath-Instruct and DeepSeekMath-RL integration
- **Advanced Positional Encoding**: 5 cutting-edge PE methods comparison
- **Mathematical Datasets**: Large-scale mathematical reasoning datasets
- **HPC Optimized**: Multi-node training on IITD HPC cluster
- **Automated Pipeline**: End-to-end experiment automation

## 🏗️ Architecture

### Multi-Node Positional Encoding Comparison
- **Node 0**: Sinusoidal PE + DeepSeekMath-Instruct
- **Node 1**: RoPE + DeepSeekMath-RL  
- **Node 2**: ALiBi + DeepSeekMath-Instruct
- **Node 3**: DIET + DeepSeekMath-RL
- **Node 4**: T5-Relative + DeepSeekMath-Instruct

### Model Components
- **Base Model**: DeepSeekMath-7B-Instruct/RL
- **Tokenizer**: Mathematical reasoning optimized
- **PE Layers**: Advanced positional encoding implementations
- **Training**: LoRA fine-tuning with mathematical reasoning datasets

## 📊 Datasets

- **Primary**: MATH (Mathematics Aptitude Test of Heuristics)
- **Secondary**: GSM8K, MathQA, MMLU-Math
- **Augmented**: Synthetic mathematical problems
- **Size**: 500K+ mathematical reasoning problems

## 🚀 Quick Start

```bash
# 1. Setup environment
./setup.sh

# 2. Verify configuration
./verify_setup.sh

# 3. Run full experiment
./run_experiment.sh

# 4. Monitor progress
./monitor_training.sh
```

## 📋 Project Structure

```
├── src/                          # Core source code
│   ├── models/                   # Model implementations
│   ├── positional_encoding/      # PE layer implementations
│   ├── tokenizers/              # Mathematical tokenizers
│   └── utils/                   # Utility functions
├── data/                        # Dataset management
├── configs/                     # Node configurations
├── scripts/                     # Automation scripts
├── experiments/                 # Experiment results
└── docs/                       # Documentation
```

## 🔬 Experimental Design

The experiment compares 5 positional encoding methods on mathematical reasoning tasks using identical base models and training procedures, enabling direct performance comparison.

## 📈 Expected Outcomes

- Comprehensive PE method comparison on mathematical reasoning
- SOTA performance on MATH and GSM8K benchmarks
- Production-ready mathematical reasoning models
- Research insights on positional encoding effectiveness

---

**Author**: Research Team  
**Institution**: IIT Delhi  
**Date**: 2024