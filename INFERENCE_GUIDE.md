# Mathematical Reasoning Transformer Inference Guide
## Production Deployment and Presentation Guide

**Author:** Research Team  
**Target:** IITD HPC Cluster Trained Models  
**Purpose:** Model Inference and Demonstration  

---

## 🎯 Overview

This guide provides comprehensive instructions for using the trained Mathematical Reasoning Transformer models for inference, demonstrations, and presentations. After multi-node training on IITD HPC cluster, you'll have 5 different models (one for each positional encoding method) ready for comparison and demonstration.

---

## 📁 Trained Model Structure

After training completion, your models will be organized as:

```
/scratch/$USER/math_reasoning/results/
├── node_0_sinusoidal_deepseek/
│   ├── final_model/           # Ready for inference
│   ├── checkpoints/          # Training checkpoints
│   ├── evaluation/           # Evaluation results
│   └── logs/                 # Training logs
├── node_1_rope_internlm/
├── node_2_alibi_orca/
├── node_3_diet_dotamath/
├── node_4_t5_mindstar/
└── final_comparison/         # Comparative analysis
    ├── comparison_report.json
    ├── visualizations/
    └── final_research_report.pdf
```

---

## 🚀 Quick Start Inference

### 1. Single Model Inference

```python
#!/usr/bin/env python3
# quick_inference.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your trained model
model_path = "/scratch/$USER/math_reasoning/results/node_0_sinusoidal_deepseek/final_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

def solve_math_problem(problem: str) -> str:
    """Solve a mathematical problem using the trained model."""
    prompt = f"Solve this step by step:\n\nProblem: {problem}\n\nSolution:"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return solution.replace(prompt, "").strip()

# Example usage
problem = "A train travels 120 miles in 2 hours. What is its speed in miles per hour?"
solution = solve_math_problem(problem)
print(f"Problem: {problem}")
print(f"Solution: {solution}")
```

### 2. Multi-Model Comparison

```python
#!/usr/bin/env python3
# compare_models.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List
import json

class MathReasoningComparator:
    """Compare all 5 positional encoding models."""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.models = {}
        self.tokenizers = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all trained models."""
        model_configs = {
            "sinusoidal": "node_0_sinusoidal_deepseek",
            "rope": "node_1_rope_internlm", 
            "alibi": "node_2_alibi_orca",
            "diet": "node_3_diet_dotamath",
            "t5_relative": "node_4_t5_mindstar"
        }
        
        for pe_method, model_dir in model_configs.items():
            model_path = f"{self.base_path}/{model_dir}/final_model"
            
            print(f"Loading {pe_method} model...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            self.tokenizers[pe_method] = tokenizer
            self.models[pe_method] = model
    
    def solve_with_all_models(self, problem: str) -> Dict[str, str]:
        """Solve problem with all models for comparison."""
        results = {}
        
        for pe_method in self.models.keys():
            try:
                solution = self._solve_with_model(problem, pe_method)
                results[pe_method] = solution
            except Exception as e:
                results[pe_method] = f"Error: {str(e)}"
        
        return results
    
    def _solve_with_model(self, problem: str, pe_method: str) -> str:
        """Solve problem with specific model."""
        model = self.models[pe_method]
        tokenizer = self.tokenizers[pe_method]
        
        prompt = f"Solve this step by step:\n\nProblem: {problem}\n\nSolution:"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return solution.replace(prompt, "").strip()

# Example usage
comparator = MathReasoningComparator("/scratch/$USER/math_reasoning/results")
problem = "If 3x + 7 = 22, what is the value of x?"
results = comparator.solve_with_all_models(problem)

for method, solution in results.items():
    print(f"\n{method.upper()} Model:")
    print(f"Solution: {solution[:200]}...")
```

---

## 🖥️ Interactive Demo Setup

### 1. Streamlit Web Application

```python
#!/usr/bin/env python3
# demo_app.py

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import pandas as pd
import json
import time

@st.cache_resource
def load_models():
    """Load all models (cached for performance)."""
    base_path = "/scratch/$USER/math_reasoning/results"
    models = {}
    tokenizers = {}
    
    model_configs = {
        "Sinusoidal": "node_0_sinusoidal_deepseek",
        "RoPE": "node_1_rope_internlm",
        "ALiBi": "node_2_alibi_orca", 
        "DIET": "node_3_diet_dotamath",
        "T5-Relative": "node_4_t5_mindstar"
    }
    
    for display_name, model_dir in model_configs.items():
        model_path = f"{base_path}/{model_dir}/final_model"
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        tokenizers[display_name] = tokenizer
        models[display_name] = model
    
    return models, tokenizers

def solve_problem(problem: str, model_name: str, models, tokenizers):
    """Solve problem with selected model."""
    model = models[model_name]
    tokenizer = tokenizers[model_name]
    
    prompt = f"Solve this step by step:\n\nProblem: {problem}\n\nSolution:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=st.session_state.get('temperature', 0.7),
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    end_time = time.time()
    
    solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
    solution = solution.replace(prompt, "").strip()
    
    return solution, end_time - start_time

# Streamlit App
st.set_page_config(
    page_title="Mathematical Reasoning Transformer Demo",
    page_icon="🧮",
    layout="wide"
)

st.title("🧮 Mathematical Reasoning Transformer Demo")
st.subtitle("Comparison of Positional Encoding Methods")

# Load models
with st.spinner("Loading models..."):
    models, tokenizers = load_models()

st.success("Models loaded successfully!")

# Sidebar configuration
st.sidebar.header("Configuration")
st.sidebar.slider("Temperature", 0.1, 1.0, 0.7, 0.1, key="temperature")

selected_model = st.sidebar.selectbox(
    "Select Model",
    ["All Models"] + list(models.keys())
)

# Main interface
st.header("Problem Input")
problem = st.text_area(
    "Enter a mathematical problem:",
    placeholder="Example: A train travels 120 miles in 2 hours. What is its speed in miles per hour?",
    height=100
)

if st.button("Solve Problem") and problem:
    if selected_model == "All Models":
        # Compare all models
        st.header("Model Comparison")
        
        results = {}
        times = {}
        
        for model_name in models.keys():
            with st.spinner(f"Solving with {model_name}..."):
                solution, solve_time = solve_problem(problem, model_name, models, tokenizers)
                results[model_name] = solution
                times[model_name] = solve_time
        
        # Display results
        for model_name, solution in results.items():
            with st.expander(f"{model_name} Solution (⏱️ {times[model_name]:.2f}s)"):
                st.write(solution)
        
        # Performance comparison
        st.header("Performance Comparison")
        time_df = pd.DataFrame({
            "Model": list(times.keys()),
            "Time (seconds)": list(times.values())
        })
        
        fig, ax = plt.subplots()
        ax.bar(time_df["Model"], time_df["Time (seconds)"])
        ax.set_ylabel("Inference Time (seconds)")
        ax.set_title("Model Inference Time Comparison")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    else:
        # Single model inference
        st.header(f"{selected_model} Solution")
        
        with st.spinner("Generating solution..."):
            solution, solve_time = solve_problem(problem, selected_model, models, tokenizers)
        
        st.write(solution)
        st.info(f"Inference time: {solve_time:.2f} seconds")

# Performance metrics display
if st.sidebar.button("Show Training Results"):
    st.header("Training Results Summary")
    
    # Load training results
    try:
        with open("/scratch/$USER/math_reasoning/results/final_comparison/comparison_report.json", "r") as f:
            results = json.load(f)
        
        # Display metrics
        metrics_df = pd.DataFrame(results.get("model_comparison", {})).T
        st.dataframe(metrics_df)
        
        # Visualization
        if "accuracy" in metrics_df.columns:
            fig, ax = plt.subplots()
            ax.bar(metrics_df.index, metrics_df["accuracy"])
            ax.set_ylabel("Accuracy")
            ax.set_title("Model Accuracy Comparison")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
    except FileNotFoundError:
        st.warning("Training results not found. Complete training first.")

# Usage instructions
with st.expander("How to use this demo"):
    st.markdown("""
    1. **Single Model**: Select a specific positional encoding model to see its solution
    2. **All Models**: Compare solutions from all 5 models simultaneously
    3. **Configuration**: Adjust temperature in the sidebar to control randomness
    4. **Performance**: View inference times and training metrics
    
    **Model Types:**
    - **Sinusoidal**: Classic transformer positional encoding
    - **RoPE**: Rotary Position Embedding (used in LLaMA)
    - **ALiBi**: Attention with Linear Biases
    - **DIET**: Decoupled positional attention  
    - **T5-Relative**: T5-style relative positional encoding
    """)

# Run with: streamlit run demo_app.py
```

### 2. Running the Demo

```bash
# On HPC cluster
cd /scratch/$USER/math_reasoning

# Install streamlit if not available
pip install streamlit

# Run the demo
streamlit run demo_app.py --server.port 8501

# Access via SSH tunnel (from local machine)
ssh -L 8501:localhost:8501 username@hpc.iitd.ac.in
# Then open http://localhost:8501 in your browser
```

---

## 📊 Presentation Materials

### 1. Performance Analysis Script

```python
#!/usr/bin/env python3
# generate_presentation_materials.py

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

class PresentationGenerator:
    """Generate presentation materials from training results."""
    
    def __init__(self, results_path: str):
        self.results_path = Path(results_path)
        self.output_dir = self.results_path / "presentation_materials"
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_all_materials(self):
        """Generate all presentation materials."""
        print("Generating presentation materials...")
        
        # Load results
        self.load_results()
        
        # Generate plots
        self.create_accuracy_comparison()
        self.create_efficiency_analysis()
        self.create_training_curves()
        self.create_error_analysis()
        self.create_attention_visualization()
        
        # Generate summary report
        self.create_summary_report()
        
        print(f"Materials saved to: {self.output_dir}")
    
    def load_results(self):
        """Load all training and evaluation results."""
        self.results = {}
        
        # Load individual model results
        for i in range(5):
            pe_methods = ["sinusoidal", "rope", "alibi", "diet", "t5_relative"]
            sota_models = ["deepseek", "internlm", "orca", "dotamath", "mindstar"]
            
            node_dir = self.results_path / f"node_{i}_{pe_methods[i]}_{sota_models[i]}"
            result_file = node_dir / "comprehensive_evaluation.json"
            
            if result_file.exists():
                with open(result_file, 'r') as f:
                    self.results[pe_methods[i]] = json.load(f)
        
        # Load comparison results
        comparison_file = self.results_path / "final_comparison" / "comparison_report.json"
        if comparison_file.exists():
            with open(comparison_file, 'r') as f:
                self.comparison_results = json.load(f)
    
    def create_accuracy_comparison(self):
        """Create accuracy comparison plot."""
        plt.figure(figsize=(12, 8))
        
        # Extract accuracy metrics
        methods = []
        exact_match = []
        math_correctness = []
        reasoning_accuracy = []
        
        for method, results in self.results.items():
            methods.append(method.upper())
            exact_match.append(results.get('exact_match_accuracy', 0) * 100)
            math_correctness.append(results.get('math_correctness', 0) * 100)
            reasoning_accuracy.append(results.get('reasoning_accuracy', 0) * 100)
        
        # Create grouped bar chart
        x = np.arange(len(methods))
        width = 0.25
        
        plt.bar(x - width, exact_match, width, label='Exact Match', alpha=0.8)
        plt.bar(x, math_correctness, width, label='Mathematical Correctness', alpha=0.8)
        plt.bar(x + width, reasoning_accuracy, width, label='Reasoning Accuracy', alpha=0.8)
        
        plt.xlabel('Positional Encoding Method')
        plt.ylabel('Accuracy (%)')
        plt.title('Mathematical Reasoning Performance Comparison')
        plt.xticks(x, methods, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / "accuracy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_efficiency_analysis(self):
        """Create efficiency analysis plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Inference time comparison
        methods = []
        inference_times = []
        parameter_counts = []
        
        for method, results in self.results.items():
            methods.append(method.upper())
            inference_times.append(results.get('mean_inference_time', 0))
            parameter_counts.append(results.get('trainable_parameters', 0) / 1e6)  # Convert to millions
        
        # Plot 1: Inference time
        ax1.bar(methods, inference_times, alpha=0.7, color='skyblue')
        ax1.set_ylabel('Inference Time (seconds)')
        ax1.set_title('Model Inference Time')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Parameter efficiency
        ax2.bar(methods, parameter_counts, alpha=0.7, color='lightcoral')
        ax2.set_ylabel('Trainable Parameters (Millions)')
        ax2.set_title('Model Parameter Count')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "efficiency_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_training_curves(self):
        """Create training curves visualization."""
        # This would require loading training logs
        # For now, create a placeholder
        plt.figure(figsize=(12, 8))
        
        # Simulated training curves (replace with actual data)
        epochs = np.arange(1, 11)
        
        for method in ["Sinusoidal", "RoPE", "ALiBi", "DIET", "T5-Relative"]:
            # Simulated loss curve
            loss = 2.0 * np.exp(-epochs/3) + 0.1 * np.random.random(10)
            plt.plot(epochs, loss, label=method, linewidth=2)
        
        plt.xlabel('Training Steps (×1000)')
        plt.ylabel('Training Loss')
        plt.title('Training Loss Curves by Positional Encoding Method')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_error_analysis(self):
        """Create error pattern analysis."""
        plt.figure(figsize=(10, 8))
        
        # Error categories
        error_types = ['Calculation\nErrors', 'Reasoning\nErrors', 'Format\nErrors', 'Incomplete\nSolutions']
        
        # Simulated error data (replace with actual analysis)
        methods = list(self.results.keys())
        error_data = np.random.rand(len(methods), len(error_types)) * 20
        
        # Create heatmap
        sns.heatmap(error_data, 
                   xticklabels=error_types,
                   yticklabels=[m.upper() for m in methods],
                   annot=True, 
                   fmt='.1f',
                   cmap='Reds',
                   cbar_kws={'label': 'Error Rate (%)'})
        
        plt.title('Error Pattern Analysis by Positional Encoding Method')
        plt.tight_layout()
        
        plt.savefig(self.output_dir / "error_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_attention_visualization(self):
        """Create attention pattern visualization."""
        # Placeholder for attention analysis
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        methods = ["Sinusoidal", "RoPE", "ALiBi", "DIET", "T5-Relative"]
        
        for i, method in enumerate(methods):
            # Simulated attention pattern
            attention_matrix = np.random.rand(10, 10)
            attention_matrix = (attention_matrix + attention_matrix.T) / 2  # Make symmetric
            
            im = axes[i].imshow(attention_matrix, cmap='Blues')
            axes[i].set_title(f'{method} Attention Pattern')
            axes[i].set_xlabel('Token Position')
            axes[i].set_ylabel('Token Position')
        
        # Remove empty subplot
        axes[5].remove()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "attention_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_report(self):
        """Create text summary report."""
        report = f"""
# Mathematical Reasoning Transformer Comparison Report

## Executive Summary

This report presents the results of comparing 5 different positional encoding methods for mathematical reasoning tasks using transformer models trained on the IITD HPC cluster.

## Models Evaluated

1. **Sinusoidal PE + DeepSeekMath-7B**: Classic transformer positional encoding
2. **RoPE + InternLM-Math**: Rotary Position Embedding 
3. **ALiBi + Orca-Math-7B**: Attention with Linear Biases
4. **DIET + DotaMath**: Decoupled positional attention
5. **T5-Relative + MindStar**: T5-style relative positional encoding

## Key Findings

### Performance Rankings

"""
        
        # Add performance data if available
        if hasattr(self, 'comparison_results'):
            report += "Performance metrics extracted from training results.\n\n"
        
        report += """
### Training Configuration

- **Hardware**: 5 GPU nodes on IITD HPC cluster
- **Container**: NVIDIA Enroot with PyTorch 2.1+
- **Training Steps**: 10,000 per model
- **Datasets**: MATH and GSM8K with chain-of-thought augmentation
- **Evaluation**: Comprehensive mathematical reasoning metrics

### Recommendations

Based on the experimental results:

1. **Best Overall Performance**: [To be filled based on results]
2. **Most Efficient**: [To be filled based on results]  
3. **Best for Long Sequences**: [To be filled based on results]

### Usage Instructions

See INFERENCE_GUIDE.md for detailed instructions on using these models for inference and demonstration.

---

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(self.output_dir / "summary_report.md", 'w') as f:
            f.write(report)

# Generate materials
if __name__ == "__main__":
    generator = PresentationGenerator("/scratch/$USER/math_reasoning/results")
    generator.generate_all_materials()
```

### 2. LaTeX Presentation Template

```latex
% presentation.tex
\documentclass{beamer}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}

\title{Mathematical Reasoning Transformers:\\ Comparative Analysis of Positional Encoding Methods}
\author{Your Name}
\institute{Indian Institute of Technology Delhi}
\date{\today}

\begin{document}

\frame{\titlepage}

\begin{frame}{Overview}
\begin{itemize}
    \item Compared 5 positional encoding methods for mathematical reasoning
    \item Trained on IITD HPC cluster using multi-node deployment
    \item Evaluated on MATH and GSM8K datasets
    \item Used SOTA techniques: LoRA, GRPO, Chain-of-Thought
\end{itemize}
\end{frame}

\begin{frame}{Experimental Setup}
\begin{columns}
\begin{column}{0.5\textwidth}
\textbf{Architecture:}
\begin{itemize}
    \item 5 GPU nodes
    \item NVIDIA Enroot containers
    \item Each node: different PE method
\end{itemize}
\end{column}
\begin{column}{0.5\textwidth}
\textbf{Models:}
\begin{itemize}
    \item Sinusoidal + DeepSeekMath
    \item RoPE + InternLM-Math
    \item ALiBi + Orca-Math
    \item DIET + DotaMath
    \item T5-Relative + MindStar
\end{itemize}
\end{column}
\end{columns}
\end{frame}

\begin{frame}{Results: Accuracy Comparison}
\centering
\includegraphics[width=0.9\textwidth]{presentation_materials/accuracy_comparison.png}
\end{frame}

\begin{frame}{Results: Efficiency Analysis}
\centering
\includegraphics[width=0.9\textwidth]{presentation_materials/efficiency_analysis.png}
\end{frame}

\begin{frame}{Interactive Demo}
\begin{itemize}
    \item Streamlit web application
    \item Real-time model comparison
    \item Performance metrics visualization
    \item Available at: [Your deployment URL]
\end{itemize}
\end{frame}

\begin{frame}{Conclusions}
\begin{itemize}
    \item [Key finding 1]
    \item [Key finding 2]
    \item [Key finding 3]
    \item Future work: [Next steps]
\end{itemize}
\end{frame}

\end{document}
```

---

## 🔧 Production Deployment

### 1. Model Serving with FastAPI

```python
#!/usr/bin/env python3
# model_server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
from typing import Dict, List, Optional

app = FastAPI(title="Mathematical Reasoning API", version="1.0.0")

class ProblemRequest(BaseModel):
    problem: str
    model_type: Optional[str] = "sinusoidal"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512

class ProblemResponse(BaseModel):
    solution: str
    model_used: str
    inference_time: float
    confidence_score: Optional[float] = None

class ModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.load_models()
    
    def load_models(self):
        """Load all trained models."""
        base_path = "/scratch/$USER/math_reasoning/results"
        
        model_configs = {
            "sinusoidal": "node_0_sinusoidal_deepseek",
            "rope": "node_1_rope_internlm",
            "alibi": "node_2_alibi_orca",
            "diet": "node_3_diet_dotamath", 
            "t5_relative": "node_4_t5_mindstar"
        }
        
        for model_type, model_dir in model_configs.items():
            model_path = f"{base_path}/{model_dir}/final_model"
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
                
                self.tokenizers[model_type] = tokenizer
                self.models[model_type] = model
                print(f"Loaded {model_type} model")
                
            except Exception as e:
                print(f"Failed to load {model_type}: {e}")

# Initialize model manager
model_manager = ModelManager()

@app.post("/solve", response_model=ProblemResponse)
async def solve_problem(request: ProblemRequest):
    """Solve a mathematical problem."""
    import time
    
    if request.model_type not in model_manager.models:
        raise HTTPException(status_code=400, detail=f"Model {request.model_type} not available")
    
    model = model_manager.models[request.model_type]
    tokenizer = model_manager.tokenizers[request.model_type]
    
    # Prepare prompt
    prompt = f"Solve this step by step:\n\nProblem: {request.problem}\n\nSolution:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate solution
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
    solution = solution.replace(prompt, "").strip()
    inference_time = time.time() - start_time
    
    return ProblemResponse(
        solution=solution,
        model_used=request.model_type,
        inference_time=inference_time
    )

@app.get("/models")
async def list_models():
    """List available models."""
    return {"available_models": list(model_manager.models.keys())}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "loaded_models": len(model_manager.models)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. Docker Deployment

```dockerfile
# Dockerfile.inference
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Install dependencies
RUN pip install fastapi uvicorn streamlit

# Copy inference code
COPY inference/ /app/
COPY models/ /app/models/

WORKDIR /app

EXPOSE 8000 8501

# Start both API and demo
CMD ["python", "model_server.py"]
```

---

## 📈 Performance Monitoring

### Real-time Metrics Dashboard

```python
#!/usr/bin/env python3
# metrics_dashboard.py

import streamlit as st
import psutil
import nvidia_ml_py3 as nvml
import time
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def get_gpu_metrics():
    """Get GPU utilization metrics."""
    nvml.nvmlInit()
    device_count = nvml.nvmlDeviceGetCount()
    
    metrics = []
    for i in range(device_count):
        handle = nvml.nvmlDeviceGetHandleByIndex(i)
        util = nvml.nvmlDeviceGetUtilizationRates(handle)
        memory = nvml.nvmlDeviceGetMemoryInfo(handle)
        
        metrics.append({
            "gpu_id": i,
            "utilization": util.gpu,
            "memory_used": memory.used / 1024**3,  # GB
            "memory_total": memory.total / 1024**3  # GB
        })
    
    return metrics

def create_metrics_dashboard():
    """Create real-time metrics dashboard."""
    st.title("🔍 Model Performance Monitor")
    
    # Create placeholders for real-time updates
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cpu_placeholder = st.empty()
    with col2:
        memory_placeholder = st.empty()
    with col3:
        gpu_placeholder = st.empty()
    
    # Real-time updates
    if st.button("Start Monitoring"):
        for _ in range(100):  # Monitor for 100 iterations
            # CPU metrics
            cpu_percent = psutil.cpu_percent()
            cpu_placeholder.metric("CPU Usage", f"{cpu_percent}%")
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_placeholder.metric("Memory Usage", f"{memory.percent}%")
            
            # GPU metrics
            try:
                gpu_metrics = get_gpu_metrics()
                if gpu_metrics:
                    gpu_util = gpu_metrics[0]["utilization"]
                    gpu_placeholder.metric("GPU Usage", f"{gpu_util}%")
            except:
                gpu_placeholder.metric("GPU Usage", "N/A")
            
            time.sleep(1)

if __name__ == "__main__":
    create_metrics_dashboard()
```

---

## 🎯 Best Practices for Demonstrations

### 1. Problem Selection for Demos

```python
# demo_problems.py

DEMO_PROBLEMS = {
    "basic_arithmetic": [
        "What is 15% of 80?",
        "If a rectangle has length 12 cm and width 8 cm, what is its perimeter?"
    ],
    "algebra": [
        "Solve for x: 3x + 7 = 22",
        "If 2x - 5 = 11, what is the value of x?"
    ],
    "geometry": [
        "A circle has radius 5 cm. What is its area?",
        "Find the area of a triangle with base 10 cm and height 6 cm."
    ],
    "word_problems": [
        "A train travels 120 miles in 2 hours. What is its speed in miles per hour?",
        "Sarah has 24 apples. She gives away 1/3 of them. How many apples does she have left?"
    ],
    "complex_reasoning": [
        "A company's revenue increased by 25% in the first quarter and decreased by 10% in the second quarter. If the initial revenue was $100,000, what was the revenue at the end of the second quarter?",
        "A water tank is filled at a rate of 5 gallons per minute and drained at a rate of 2 gallons per minute. If the tank starts empty, how long will it take to fill a 90-gallon tank?"
    ]
}
```

### 2. Demo Script Template

```python
#!/usr/bin/env python3
# demo_script.py

class MathReasoningDemo:
    """Interactive demonstration script."""
    
    def __init__(self):
        self.load_models()
        self.current_problem = None
    
    def run_demo(self):
        """Run interactive demo."""
        print("🧮 Mathematical Reasoning Transformer Demo")
        print("=" * 50)
        
        while True:
            print("\nOptions:")
            print("1. Quick comparison")
            print("2. Custom problem")
            print("3. Benchmark problems")
            print("4. Performance analysis")
            print("5. Exit")
            
            choice = input("\nSelect option (1-5): ")
            
            if choice == "1":
                self.quick_comparison()
            elif choice == "2":
                self.custom_problem()
            elif choice == "3":
                self.benchmark_problems()
            elif choice == "4":
                self.performance_analysis()
            elif choice == "5":
                break
    
    def quick_comparison(self):
        """Quick comparison of all models."""
        problem = "A train travels 120 miles in 2 hours. What is its speed?"
        print(f"\nProblem: {problem}")
        print("\nSolutions from all models:")
        print("-" * 40)
        
        # Show solutions from all models
        # Implementation here...
    
    def custom_problem(self):
        """Handle custom problem input."""
        problem = input("\nEnter your mathematical problem: ")
        # Process custom problem...
    
    def benchmark_problems(self):
        """Run benchmark problems."""
        # Run predefined benchmark problems...
        pass
    
    def performance_analysis(self):
        """Show performance analysis."""
        # Display performance metrics...
        pass

if __name__ == "__main__":
    demo = MathReasoningDemo()
    demo.run_demo()
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

1. **Model Loading Errors**
   ```bash
   # Check model files
   ls -la /scratch/$USER/math_reasoning/results/*/final_model/
   
   # Verify permissions
   chmod -R 755 /scratch/$USER/math_reasoning/results/
   ```

2. **CUDA Memory Issues**
   ```python
   # Use smaller batch sizes or model sharding
   model = AutoModelForCausalLM.from_pretrained(
       model_path,
       torch_dtype=torch.bfloat16,
       device_map="auto",
       max_memory={0: "20GB", 1: "20GB"}
   )
   ```

3. **Performance Optimization**
   ```python
   # Enable optimizations
   model = torch.compile(model)  # PyTorch 2.0+
   
   # Use Flash Attention
   with torch.backends.cuda.sdp_kernel(enable_flash=True):
       outputs = model.generate(...)
   ```

---

## 📚 Additional Resources

### Documentation Links
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Model Cards and Citations
Create model cards for each trained model following Hugging Face standards.

### Performance Baselines
Document baseline performance metrics for comparison with future experiments.

---

**End of Inference Guide**

This comprehensive guide provides everything needed to deploy, demonstrate, and present your Mathematical Reasoning Transformer models effectively. Follow the examples and adapt them to your specific requirements.