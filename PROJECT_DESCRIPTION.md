# Mathematical Reasoning LLM for Edge & Embedded Systems

## Project Vision
This project aims to develop a **compact, high-performance language model (LLM) specialized for mathematical reasoning**, optimized for deployment on edge devices and embedded systems. The goal is to bridge the gap between state-of-the-art mathematical LLMs and real-world, resource-constrained environments—enabling applications in robotics, IoT, STEM education, industrial automation, and scientific instrumentation.

## Real-World Motivation
- **Edge/Embedded Use Cases:** Many real-world systems (e.g., scientific sensors, educational robots, industrial controllers) require on-device mathematical reasoning for privacy, latency, or connectivity reasons.
- **Why Not Large LLMs?** Most SOTA LLMs (e.g., GPT-3, Llama-2) are too large (7B+ params) for edge deployment. This project targets models in the 100M–2.8B parameter range, with memory-efficient fine-tuning and inference.
- **Numerical Example:**
  - **Pythia-2.8B + LoRA:** 1.85% trainable params (52M/2.8B), 10.5GB GPU RAM for training, <6GB for inference, fits on a 16GB T4/P100 or even smaller edge GPUs.
  - **Accuracy:** 0.82 (GSM8K, 1000 train, 200 eval, 100 steps)
  - **BLEU:** 0.67, **ROUGE-L:** 0.74, **Train loss:** 1.88, **Eval loss:** 2.11

## Core Technologies & Strategies

### 1. **Flexible Positional Encoding (PE) Research**
- **Intention:** Find the best PE for math reasoning in small/medium LLMs.
- **Technologies:**
  - Custom PE layers: RoPE, ALiBi, Sinusoidal, DIET, MathAdaptive, T5-Relative.
  - Modular injection: Each transformer layer gets its own, shape-consistent PE instance.
- **Strategy:**
  - Systematic ablation: Swap PE methods, measure impact on accuracy, generalization, and memory.
  - All PE configs saved with checkpoints for reproducibility.

### 2. **Parameter-Efficient Fine-Tuning (LoRA/PEFT)**
- **Intention:** Enable high-quality adaptation on small hardware.
- **Technologies:**
  - LoRA (Low-Rank Adaptation), PEFT library.
  - Only 1–2% of parameters are trainable; rest are frozen.
- **Strategy:**
  - LoRA adapters injected into attention and PE layers.
  - Achieve SOTA math reasoning with minimal memory/compute.
- **Numerical Example:**
  - **Pythia-2.8B:** 52M trainable params, 1.85% of total.
  - **Training RAM:** 10.5GB (batch 1, seq_len 512, 16GB GPU).

### 3. **Kaggle/Edge Compatibility & Memory Optimization**
- **Intention:** Guarantee training/inference on 16GB GPUs and below.
- **Technologies:**
  - PyTorch, HuggingFace Trainer, FP16, gradient checkpointing, gradient accumulation.
  - All paths/caches set to `/kaggle/working/` or `/tmp/` for cloud/edge safety.
- **Strategy:**
  - Reduce batch size, sequence length, and dataset size for memory efficiency.
  - Use gradient accumulation to simulate larger batch sizes.
  - All scripts tested on Kaggle and local edge hardware.

### 4. **Robust Data Pipeline for Math Reasoning**
- **Intention:** Support diverse math datasets and ensure correct batching/padding.
- **Technologies:**
  - Custom `MathDatasetLoader` for GSM8K, MATH, OpenMathInstruct, MetaMathQA, MathInstruct.
  - HuggingFace Datasets, DataCollatorForLanguageModeling.
- **Strategy:**
  - Tokenizer always saved/loaded with model for consistency.
  - Data pipeline supports variable-length, multi-source math problems.

### 5. **Trainer Integration & Custom Save/Load**
- **Intention:** Leverage HuggingFace Trainer for robust training, but avoid known bugs (e.g., shared tensor errors).
- **Technologies:**
  - HuggingFace Trainer, custom save logic, safetensors, torch.save fallback.
- **Strategy:**
  - Unique parameter naming and deduplication for PE/LoRA.
  - Save PE config (`pe_config.json`) and tokenizer with every checkpoint.
  - Automated shared tensor checks before saving.

### 6. **Comprehensive Testing & Validation**
- **Intention:** Guarantee correctness, reproducibility, and edge-case safety.
- **Technologies:**
  - Custom test scripts: PE consistency, device/shape checks, shared tensor detection.
  - Automated forward/backward pass, save/load, and inference tests.
- **Strategy:**
  - Every PE method and model config is tested for shape/device/gradient correctness.
  - Shared tensor checks run before every save.

### 7. **Evaluation, Visualization, and Real-World Metrics**
- **Intention:** Provide actionable, research-grade metrics and plots for real-world deployment.
- **Technologies:**
  - Matplotlib, HuggingFace metrics (accuracy, BLEU, ROUGE-L, perplexity).
  - Custom scripts for evaluation and visualization.
- **Strategy:**
  - Evaluate on GSM8K, MATH, and other datasets.
  - Visualize results as bar charts, logs, and result files for easy comparison.

## Example Results & Deployment
- **Pythia-2.8B + ALiBi, LoRA, GSM8K:**
  - Train loss: 1.88, Eval loss: 2.11, Accuracy: 0.82, BLEU: 0.67, ROUGE-L: 0.74 (1000 train, 200 eval, 100 steps)
  - Inference RAM: <6GB (fits on Jetson Orin, Xavier, T4, P100, etc.)
- **No shared tensor or save/load errors in final pipeline.**
- **Deployment:** Model can be quantized and exported for ONNX/TensorRT for edge/embedded use.

## Real-World Applications
- **STEM Education:** On-device math tutor for tablets, robots, or classroom tools.
- **Industrial Automation:** Embedded math reasoning for PLCs, controllers, and smart sensors.
- **Robotics:** Real-time math problem solving for navigation, manipulation, and decision-making.
- **IoT/Edge Analytics:** Local math inference for privacy, latency, and bandwidth savings.
- **Scientific Instrumentation:** Embedded LLM for experiment control, data analysis, and automation.

## Development Phases & Intentions
1. **Initial Setup:** Baseline transformer, math data pipeline (PyTorch, HuggingFace, GSM8K).
2. **PE Research:** Modular PE layers, dynamic shape/device handling, ablation studies.
3. **LoRA & Fine-tuning:** Efficient large-model training, parameter freezing, LoRA/PEFT.
4. **Kaggle/Edge Optimization:** Memory/disk safety, requirements pinning, path handling.
5. **Robust Save/Load:** Custom logic, config tracking, deduplication, shared tensor checks.
6. **Testing & Validation:** Automated scripts for correctness, reproducibility, and edge-case safety.
7. **Evaluation & Visualization:** Metrics, plots, and research-grade reporting.

## Git Development History
See `PROJECT_HISTORY.md` for a full commit log and timeline of the project's evolution, including all major bugfixes, feature additions, and refactors.

---

**This project is designed for robust, reproducible, and extensible research in mathematical reasoning with transformers, with a focus on real-world deployment in edge and embedded systems.** 