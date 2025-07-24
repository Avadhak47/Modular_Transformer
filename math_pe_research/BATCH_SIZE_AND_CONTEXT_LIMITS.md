# Batch Size and Context Limits in Mathematical Reasoning Models

## Overview

This document explains how batch size and context limits work in our mathematical reasoning model, including how they're imposed, their memory implications, and how different positional encoding methods handle them.

## 1. Batch Size Handling

### What is Batch Size?
- **Definition**: Number of sequences processed in parallel during training/inference
- **Purpose**: Improves computational efficiency and gradient stability
- **Memory Scaling**: Linear with batch size

### Batch Size in Practice
```python
# Example: batch_size = 4
batch_size = 4
seq_len = 2048
hidden_size = 2560

# Each sequence can have different lengths, padded to max length
sequences = [
    "Solve: 2x + 3 = 7",           # Length: 8 tokens
    "What is the derivative of x²?", # Length: 12 tokens  
    "Find the area of a circle...",  # Length: 15 tokens
    "Prove that sqrt(2) is irrational" # Length: 10 tokens
]

# All padded to max_length = 2048
padded_sequences = pad_to_max_length(sequences, max_length=2048)
```

### Memory Usage with Batch Size
```python
# Memory scales linearly with batch size
attention_memory = batch_size × seq_len² × num_heads × head_dim

# For typical values:
# batch_size=4, seq_len=2048, num_heads=40, head_dim=64
attention_memory = 4 × 2048² × 40 × 64 = ~671M elements
# With float16: ~1.34 GB just for attention matrices
```

## 2. Context Limit Imposition

### What is Context Limit?
- **Definition**: Maximum sequence length the model can process
- **Purpose**: Control memory usage and computational complexity
- **Typical Values**: 2048-8192 tokens for training, longer for inference

### Multiple Levels of Context Limitation

#### 1. Tokenizer Level
```python
# In train_and_eval.py
tokenizer.padding_side = "left"  # Pad on the left for decoder-only models
tokenizer.truncation_side = "right"  # Truncate from the right

# Tokenization with max_length
encoded = tokenizer(
    text,
    max_length=2048,  # Context limit
    truncation=True,
    padding=True,
    return_tensors="pt"
)
```

#### 2. Positional Encoding Level
```python
# Each PE layer has max_seq_len parameter
rope_pe = RotaryPositionalEmbedding(dim=64, max_seq_len=32768)
sinusoidal_pe = SinusoidalPositionalEncoding(d_model=2560, max_seq_len=8192)
alibi_pe = ALiBiPositionalEncoding(d_model=2560, max_seq_len=8192)
```

#### 3. Attention Level
```python
# Attention has quadratic memory scaling
# Memory = O(batch_size × seq_len² × num_heads)
# This is the primary constraint for context length
```

#### 4. Model Configuration
```python
# Model config typically specifies max_position_embeddings
config = AutoConfig.from_pretrained("pythia-2.8b")
config.max_position_embeddings = 2048  # Default context limit
```

## 3. Memory Scaling Analysis

### Attention Memory Complexity
```python
def calculate_attention_memory(batch_size, seq_len, num_heads, head_dim, dtype="float16"):
    """Calculate memory usage for attention matrices."""
    
    # Q, K, V matrices: batch_size × num_heads × seq_len × head_dim
    qkv_memory = 3 * batch_size * num_heads * seq_len * head_dim
    
    # Attention scores: batch_size × num_heads × seq_len × seq_len
    attention_scores = batch_size * num_heads * seq_len * seq_len
    
    # Total memory
    total_elements = qkv_memory + attention_scores
    
    # Convert to bytes
    if dtype == "float16":
        bytes_per_element = 2
    elif dtype == "float32":
        bytes_per_element = 4
    
    total_bytes = total_elements * bytes_per_element
    total_gb = total_bytes / (1024**3)
    
    return {
        "total_elements": total_elements,
        "total_bytes": total_bytes,
        "total_gb": total_gb,
        "qkv_memory": qkv_memory,
        "attention_scores": attention_scores
    }

# Example calculations
configs = [
    {"batch_size": 1, "seq_len": 2048, "num_heads": 40, "head_dim": 64},
    {"batch_size": 4, "seq_len": 2048, "num_heads": 40, "head_dim": 64},
    {"batch_size": 1, "seq_len": 4096, "num_heads": 40, "head_dim": 64},
    {"batch_size": 1, "seq_len": 8192, "num_heads": 40, "head_dim": 64},
]

for config in configs:
    memory = calculate_attention_memory(**config)
    print(f"Config: {config}")
    print(f"Memory: {memory['total_gb']:.2f} GB")
    print()
```

### Practical Memory Limits
```python
# Typical GPU memory constraints
gpu_memory_limits = {
    "RTX 4090": 24,  # GB
    "RTX 3090": 24,  # GB
    "V100": 32,      # GB
    "A100": 80,      # GB
}

# Safe memory usage (leave 20% for other operations)
safe_memory = gpu_memory * 0.8

# Maximum sequence length for different batch sizes
def max_seq_len_for_memory(gpu_memory_gb, batch_size=1, num_heads=40, head_dim=64):
    """Calculate maximum sequence length for given GPU memory."""
    safe_memory = gpu_memory_gb * 0.8 * (1024**3)  # Convert to bytes
    
    # Solve: batch_size * num_heads * seq_len² * head_dim * 2 bytes = safe_memory
    # seq_len² = safe_memory / (batch_size * num_heads * head_dim * 2)
    seq_len_squared = safe_memory / (batch_size * num_heads * head_dim * 2)
    max_seq_len = int(seq_len_squared ** 0.5)
    
    return max_seq_len
```

## 4. Context Limit Enforcement

### 1. Input Truncation
```python
# In MathDatasetLoader
def tokenize_text(self, text, max_length=2048):
    """Tokenize text with truncation."""
    encoded = self.tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None
    )
    
    # Truncation warning
    if len(encoded['input_ids']) > max_length:
        print(f"Warning: Text truncated from {len(encoded['input_ids'])} to {max_length} tokens")
    
    return encoded
```

### 2. PE Layer Limits
```python
# Each PE layer handles context limits differently
class SinusoidalPositionalEncoding(nn.Module):
    def forward(self, x, position_ids=None):
        batch_size, seq_len, d_model = x.shape
        
        if seq_len > self.max_seq_len:
            # Generate PE on-the-fly for longer sequences
            positions = torch.arange(seq_len, device=x.device)
            # ... generate PE dynamically
        else:
            # Use pre-computed PE table
            pe = self.pe[:, :seq_len, :]
```

### 3. Attention Masking
```python
# Attention masks prevent attention beyond sequence length
def create_attention_mask(seq_len, device):
    """Create causal attention mask."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask
```

### 4. Memory Management
```python
# Gradient checkpointing for long sequences
training_args = TrainingArguments(
    gradient_checkpointing=True,  # Trade compute for memory
    max_grad_norm=1.0,
    # ... other args
)
```

## 5. PE Layer Context Handling

### RoPE (Rotary Positional Embedding)
```python
# ✅ Excellent extrapolation properties
# Can handle sequences longer than training length
rope_pe = RotaryPositionalEmbedding(dim=64, max_seq_len=32768)

# Extrapolates beyond max_seq_len
long_query = torch.randn(1, 40, 8192, 64)  # 8K sequence
long_key = torch.randn(1, 40, 8192, 64)
output_q, output_k = rope_pe(long_query, long_key)  # ✅ Works
```

### Sinusoidal PE
```python
# ⚠️ Fixed max_seq_len, but can generate on-the-fly
sinusoidal_pe = SinusoidalPositionalEncoding(d_model=2560, max_seq_len=2048)

# For sequences > max_seq_len, generates PE dynamically
long_hidden = torch.randn(1, 4096, 2560)  # 4K sequence
output = sinusoidal_pe(long_hidden)  # ✅ Now works with fix
```

### T5 Relative PE
```python
# ✅ Uses relative position buckets, handles any length
t5_pe = T5RelativePositionalBias(d_model=2560, num_heads=40)

# Works with any sequence length
long_hidden = torch.randn(1, 8192, 2560)  # 8K sequence
attn_scores = torch.randn(1, 40, 8192, 8192)
output = t5_pe(long_hidden, attention_scores=attn_scores)  # ✅ Works
```

### ALiBi PE
```python
# ✅ Excellent extrapolation properties
alibi_pe = ALiBiPositionalEncoding(d_model=2560, num_heads=40, max_seq_len=2048)

# Can handle longer sequences
long_hidden = torch.randn(1, 8192, 2560)  # 8K sequence
output = alibi_pe(long_hidden)  # ✅ Works
```

### DIET PE
```python
# ⚠️ Standard truncation, but now generates on-the-fly
diet_pe = DIETPositionalEncoding(d_model=2560, max_seq_len=2048)

# For sequences > max_seq_len, generates PE dynamically
long_hidden = torch.randn(1, 4096, 2560)  # 4K sequence
output = diet_pe(long_hidden)  # ✅ Now works with fix
```

### Math-Adaptive PE
```python
# ✅ Adaptive to mathematical content, handles any length
math_pe = MathAdaptivePositionalEncoding(d_model=2560, max_seq_len=2048)

# Works with any sequence length
long_hidden = torch.randn(1, 8192, 2560)  # 8K sequence
tokens = torch.randint(0, 1000, (1, 8192))
output = math_pe(long_hidden, token_ids=tokens)  # ✅ Works
```

## 6. Practical Recommendations

### Training Configuration
```python
# Recommended settings for different GPU memory
recommendations = {
    "8GB GPU": {
        "batch_size": 1,
        "max_length": 1024,
        "gradient_checkpointing": True,
        "fp16": True
    },
    "16GB GPU": {
        "batch_size": 2,
        "max_length": 2048,
        "gradient_checkpointing": False,
        "fp16": True
    },
    "24GB GPU": {
        "batch_size": 4,
        "max_length": 2048,
        "gradient_checkpointing": False,
        "fp16": True
    },
    "32GB+ GPU": {
        "batch_size": 8,
        "max_length": 4096,
        "gradient_checkpointing": False,
        "fp16": True
    }
}
```

### Inference Configuration
```python
# For inference, can use longer sequences
inference_config = {
    "batch_size": 1,  # Usually 1 for inference
    "max_length": 8192,  # Can be longer than training
    "use_cache": True,  # Enable KV cache for efficiency
    "fp16": True
}
```

### Memory Optimization Techniques
```python
# 1. Gradient Checkpointing
training_args = TrainingArguments(
    gradient_checkpointing=True,  # Trade compute for memory
)

# 2. Mixed Precision Training
training_args = TrainingArguments(
    fp16=True,  # Use float16 for memory efficiency
)

# 3. LoRA for Large Models
model = MathematicalReasoningModel(
    model_size="deepseek-ai/deepseek-math-7b-instruct",
    use_lora=True,  # Reduce memory footprint
)

# 4. Dynamic Batching
def dynamic_batch_size(available_memory_gb, seq_len):
    """Calculate optimal batch size for available memory."""
    memory_per_sample = seq_len * seq_len * 40 * 64 * 2  # bytes
    max_batch_size = int(available_memory_gb * 0.8 * (1024**3) / memory_per_sample)
    return max(1, max_batch_size)
```

## 7. Verification Results

All PE layers have been verified to maintain constant output shapes:

```
✅ RoPE PE: Shape consistency PASSED
✅ Sinusoidal PE: Shape consistency PASSED  
✅ T5 Relative PE: Shape consistency PASSED
✅ DIET PE: Shape consistency PASSED
✅ ALiBi PE: Shape consistency PASSED
✅ Math-Adaptive PE: Shape consistency PASSED
```

All PE layers now handle context limits properly:

```
✅ RoPE PE: Extrapolates beyond training length
✅ Sinusoidal PE: Generates PE on-the-fly for long sequences
✅ T5 Relative PE: Uses relative position buckets
✅ DIET PE: Generates PE on-the-fly for long sequences
✅ ALiBi PE: Excellent extrapolation properties
✅ Math-Adaptive PE: Adaptive to mathematical content
```

## Summary

- **Batch Size**: Scales linearly with memory, typically 1-32 for training
- **Context Limits**: Imposed at tokenizer, PE, attention, and model levels
- **Memory Scaling**: Quadratic with sequence length due to attention
- **PE Handling**: All PE layers now properly handle sequences beyond their max_seq_len
- **Practical Limits**: Determined by GPU memory and attention complexity
- **Optimization**: Use gradient checkpointing, mixed precision, and LoRA for efficiency 