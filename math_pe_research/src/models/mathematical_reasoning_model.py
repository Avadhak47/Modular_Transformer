"""
Mathematical Reasoning Model with Configurable Positional Encoding

This module implements a mathematical reasoning model based on DeepSeekMath
with pluggable positional encoding methods for research comparison.

Key Features:
- Integration with DeepSeekMath-Instruct-7B and DeepSeekMath-RL-7B
- Configurable positional encoding methods
- Mathematical reasoning optimized tokenizer
- Fine-tuning capabilities with LoRA
- Optimized for mathematical problem solving
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math  # Add missing import
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig,
    LlamaConfig,
    LlamaForCausalLM
)
# from peft import LoraConfig, get_peft_model, TaskType  # Optional dependency
import logging
from typing import Optional, Dict, Any, Tuple, List
import warnings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from positional_encoding import get_positional_encoding, PE_REGISTRY

logger = logging.getLogger(__name__)


class MathematicalReasoningModel(nn.Module):
    """
    Mathematical Reasoning Model with configurable positional encoding.
    
    Built on top of DeepSeekMath models with the ability to swap out
    positional encoding methods for research purposes.
    """
    
    def __init__(
        self,
        base_model_name: str = "deepseek-ai/deepseek-math-7b-instruct",
        pe_method: str = "rope",
        pe_config: Optional[Dict[str, Any]] = None,
        use_lora: bool = True,
        lora_config: Optional[Dict[str, Any]] = None,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        trust_remote_code: bool = True,
        torch_dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        cache_dir: Optional[str] = None
    ):
        super().__init__()
        
        self.base_model_name = base_model_name
        self.pe_method = pe_method
        self.pe_config = pe_config or {}
        self.use_lora = use_lora
        self.lora_config = lora_config or {}
        self.cache_dir = cache_dir
        
        # Load tokenizer (mathematical reasoning optimized)
        self.tokenizer = self._load_tokenizer()
        
        # Load base model configuration
        self.config = self._load_model_config()
        
        # Load base model
        self.base_model = self._load_base_model(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        
        # Replace positional encoding
        self._replace_positional_encoding()
        
        # Apply LoRA if specified
        if self.use_lora:
            self.base_model = self._apply_lora()
        
        # Mathematical reasoning specific modifications
        self._apply_math_optimizations()
        
        logger.info(f"Initialized MathematicalReasoningModel with {pe_method} positional encoding")
    
    def _load_tokenizer(self) -> AutoTokenizer:
        """Load and configure the mathematical reasoning optimized tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            use_fast=True,
            cache_dir=self.cache_dir
        )
        
        # Add mathematical symbols if not present
        math_symbols = [
            # Mathematical operators
            "∑", "∏", "∫", "∂", "∇", "√", "∞", "∝", "≈", "≠", "≤", "≥",
            # Greek letters (common in math)
            "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "λ", "μ", "π", "ρ", "σ", "τ", "φ", "ψ", "ω",
            "Γ", "Δ", "Θ", "Λ", "Ξ", "Π", "Σ", "Φ", "Ψ", "Ω",
            # Mathematical notation
            "∈", "∉", "⊂", "⊃", "∩", "∪", "∅", "ℝ", "ℂ", "ℕ", "ℤ", "ℚ",
            # Special tokens for mathematical reasoning
            "<math>", "</math>", "<solution>", "</solution>", 
            "<step>", "</step>", "<reasoning>", "</reasoning>"
        ]
        
        # Add tokens that aren't already in vocabulary
        new_tokens = []
        for symbol in math_symbols:
            if symbol not in tokenizer.get_vocab():
                new_tokens.append(symbol)
        
        if new_tokens:
            tokenizer.add_tokens(new_tokens)
            logger.info(f"Added {len(new_tokens)} mathematical symbols to tokenizer")
        
        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def _load_model_config(self) -> AutoConfig:
        """Load model configuration."""
        config = AutoConfig.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )
        
        # Store original config for PE replacement
        self.original_config = config
        
        return config
    
    def _load_base_model(
        self,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        trust_remote_code: bool = True,
        torch_dtype: torch.dtype = torch.float16,
        device_map: str = "auto"
    ) -> nn.Module:
        """Load the base DeepSeekMath model."""
        
        # Quantization configuration
        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            config=self.config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            quantization_config=quantization_config,
            cache_dir=self.cache_dir
        )
        
        # Set use_cache after loading
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        
        # Resize token embeddings if we added new tokens
        if len(self.tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Resized token embeddings to {len(self.tokenizer)}")
        
        return model
    
    def _replace_positional_encoding(self):
        """Replace the model's positional encoding with the specified method."""
        logger.info(f"Replacing positional encoding with {self.pe_method}")
        
        # Get model architecture details
        d_model = self.config.hidden_size
        max_position_embeddings = getattr(self.config, 'max_position_embeddings', 8192)
        num_heads = getattr(self.config, 'num_attention_heads', 8)
        
        # Create new positional encoding with proper arguments for each type
        pe_config = {
            'd_model': d_model,
            'max_seq_len': max_position_embeddings,
            'num_heads': num_heads,
            **self.pe_config
        }
        
        # The get_positional_encoding function now handles argument filtering
        new_pe = get_positional_encoding(self.pe_method, **pe_config)
        
        # Replace positional encoding in the model
        # This depends on the model architecture - DeepSeekMath is based on Llama
        self._replace_llama_positional_encoding(new_pe)
    
    def _replace_llama_positional_encoding(self, new_pe: nn.Module):
        """Replace positional encoding in Llama-based architecture."""
        
        # Check model architecture and handle accordingly
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'layers'):
            # Llama/DeepSeek style models
            layers = self.base_model.model.layers
            layer_attr = 'self_attn'
        elif hasattr(self.base_model, 'transformer') and hasattr(self.base_model.transformer, 'h'):
            # GPT2 style models
            layers = self.base_model.transformer.h
            layer_attr = 'attn'
        elif hasattr(self.base_model, 'layers'):
            # Direct layer access
            layers = self.base_model.layers
            layer_attr = 'self_attn'
        else:
            logger.warning(f"Unknown model architecture, skipping PE replacement")
            return
        
        # For Llama/DeepSeek models, we need to modify the attention layers
        for layer_idx, layer in enumerate(layers):
            # Store reference to the attention layer
            attention = getattr(layer, layer_attr, None)
            if attention is None:
                logger.warning(f"No attention layer found at layer {layer_idx}")
                continue
            
            # Create a custom attention wrapper that uses our PE
            custom_attention = CustomAttentionWithPE(
                attention, 
                new_pe, 
                self.pe_method
            )
            
            # Replace the attention layer
            setattr(layer, layer_attr, custom_attention)
        
        logger.info(f"Replaced positional encoding in {len(layers)} layers")
    
    def _apply_lora(self) -> nn.Module:
        """Apply LoRA (Low-Rank Adaptation) for efficient fine-tuning."""
        
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            # Default LoRA configuration optimized for mathematical reasoning
            default_lora_config = {
                "task_type": TaskType.CAUSAL_LM,
                "r": 64,  # Rank
                "lora_alpha": 16,  # Scaling parameter
                "lora_dropout": 0.1,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                "bias": "none",
                "inference_mode": False
            }
            
            # Merge with user-provided config
            lora_config = {**default_lora_config, **self.lora_config}
            
            # Create LoRA configuration
            peft_config = LoraConfig(**lora_config)
            
            # Apply LoRA to model
            model = get_peft_model(self.base_model, peft_config)
            
            logger.info(f"Applied LoRA with rank {lora_config['r']} to model")
            logger.info(f"Trainable parameters: {model.print_trainable_parameters()}")
            
            return model
            
        except ImportError:
            logger.warning("PEFT not available, skipping LoRA fine-tuning")
            return self.base_model
    
    def _apply_math_optimizations(self):
        """Apply mathematical reasoning specific optimizations."""
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable()
        
        # Set model to training mode
        self.base_model.train()
        
        # Mathematical reasoning specific settings
        if hasattr(self.base_model.config, 'use_cache'):
            self.base_model.config.use_cache = False
        
        logger.info("Applied mathematical reasoning optimizations")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        
        # Prepare inputs
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Forward through base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 2048,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate mathematical reasoning solutions."""
        
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        
        # Set model to evaluation mode
        self.base_model.eval()
        
        with torch.no_grad():
            outputs = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **kwargs
            )
        
        return outputs
    
    def solve_math_problem(
        self,
        problem: str,
        max_length: int = 2048,
        temperature: float = 0.1,  # Lower temperature for mathematical reasoning
        **kwargs
    ) -> str:
        """Solve a mathematical problem and return the solution."""
        
        # Format the problem with mathematical reasoning prompt
        prompt = self._format_math_prompt(problem)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Generate solution
        with torch.no_grad():
            outputs = self.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                temperature=temperature,
                **kwargs
            )
        
        # Decode solution
        solution = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return solution.strip()
    
    def _format_math_prompt(self, problem: str) -> str:
        """Format mathematical problem with appropriate prompt."""
        
        prompt_template = """<|im_start|>system
You are a mathematical reasoning expert. Solve the following problem step by step, showing your work clearly.
<|im_end|>

<|im_start|>user
{problem}
<|im_end|>

<|im_start|>assistant
I'll solve this step by step.

<reasoning>
"""
        
        return prompt_template.format(problem=problem)
    
    def save_pretrained(self, save_directory: str):
        """Save the model and tokenizer."""
        # Save the base model (with LoRA if applied)
        self.base_model.save_pretrained(save_directory)
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(save_directory)
        
        # Save configuration
        import json
        config_dict = {
            'base_model_name': self.base_model_name,
            'pe_method': self.pe_method,
            'pe_config': self.pe_config,
            'use_lora': self.use_lora,
            'lora_config': self.lora_config
        }
        
        with open(f"{save_directory}/model_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load a saved model."""
        import json
        
        # Load configuration
        with open(f"{model_path}/model_config.json", "r") as f:
            config_dict = json.load(f)
        
        # Merge with any provided kwargs
        config_dict.update(kwargs)
        
        # Create model instance
        model = cls(**config_dict)
        
        # Load the saved weights
        model.base_model = AutoModelForCausalLM.from_pretrained(model_path)
        
        logger.info(f"Model loaded from {model_path}")
        return model


class CustomAttentionWithPE(nn.Module):
    """
    Custom attention layer that integrates with configurable positional encoding.
    Wraps the original attention mechanism while applying our PE method.
    """
    
    def __init__(self, original_attention: nn.Module, pe_layer: nn.Module, pe_method: str):
        super().__init__()
        self.original_attention = original_attention
        self.pe_layer = pe_layer
        self.pe_method = pe_method
        
        # Copy attributes from original attention
        for attr in ['hidden_size', 'num_heads', 'head_dim', 'config']:
            if hasattr(original_attention, attr):
                setattr(self, attr, getattr(original_attention, attr))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        # Apply positional encoding based on method
        if self.pe_method in ['rope', 'math_adaptive']:
            # For RoPE-like methods, we apply PE within attention computation
            return self._forward_with_rope_like_pe(
                hidden_states, attention_mask, position_ids, 
                past_key_value, output_attentions, use_cache, **kwargs
            )
        else:
            # For other methods, apply PE to hidden states before attention
            return self._forward_with_additive_pe(
                hidden_states, attention_mask, position_ids,
                past_key_value, output_attentions, use_cache, **kwargs
            )
    
    def _forward_with_rope_like_pe(self, hidden_states, attention_mask, position_ids, 
                                  past_key_value, output_attentions, use_cache, **kwargs):
        """Forward pass for RoPE-like positional encodings."""
        
        # Get query, key, value projections
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        query_states = self.original_attention.q_proj(hidden_states)
        key_states = self.original_attention.k_proj(hidden_states)
        value_states = self.original_attention.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply positional encoding
        if hasattr(self.pe_layer, 'forward'):
            # Extract token IDs if available (for mathematical adaptive PE)
            token_ids = kwargs.get('input_ids', None)
            
            query_states = self.pe_layer(query_states, token_ids=token_ids)
            key_states = self.pe_layer(key_states, token_ids=token_ids)
        
        # Continue with standard attention computation
        return self._compute_attention(
            query_states, key_states, value_states, attention_mask,
            past_key_value, output_attentions, use_cache
        )
    
    def _forward_with_additive_pe(self, hidden_states, attention_mask, position_ids,
                                 past_key_value, output_attentions, use_cache, **kwargs):
        """Forward pass for additive positional encodings."""
        
        # Apply positional encoding to hidden states
        if hasattr(self.pe_layer, 'forward'):
            token_ids = kwargs.get('input_ids', None)
            hidden_states = self.pe_layer(hidden_states, token_ids=token_ids, position_ids=position_ids)
        
        # Use original attention with modified hidden states
        return self.original_attention(
            hidden_states, attention_mask, position_ids,
            past_key_value, output_attentions, use_cache, **kwargs
        )
    
    def _compute_attention(self, query_states, key_states, value_states, attention_mask,
                          past_key_value, output_attentions, use_cache):
        """Compute multi-head attention."""
        
        batch_size, seq_len, num_heads, head_dim = query_states.shape
        
        # Transpose for attention computation
        query_states = query_states.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = self.original_attention.o_proj(attn_output)
        
        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (past_key_value,)
        
        return outputs


# Factory function for easy model creation
def create_mathematical_reasoning_model(
    pe_method: str = "rope",
    base_model: str = "deepseek-ai/deepseek-math-7b-instruct",
    **kwargs
) -> MathematicalReasoningModel:
    """
    Factory function to create a mathematical reasoning model with specified PE method.
    
    Args:
        pe_method: Positional encoding method ('rope', 'alibi', 'sinusoidal', 'diet', 't5_relative', 'math_adaptive')
        base_model: Base model name
        **kwargs: Additional arguments for model initialization
    
    Returns:
        Configured MathematicalReasoningModel
    """
    
    if pe_method not in PE_REGISTRY:
        raise ValueError(f"Unknown PE method: {pe_method}. Available: {list(PE_REGISTRY.keys())}")
    
    return MathematicalReasoningModel(
        base_model_name=base_model,
        pe_method=pe_method,
        **kwargs  # Remove explicit cache_dir to avoid duplication
    )


if __name__ == "__main__":
    # Test model creation
    print("Testing Mathematical Reasoning Model creation...")
    
    # Test with different PE methods
    pe_methods = ['rope', 'sinusoidal', 'math_adaptive']
    
    for pe_method in pe_methods:
        print(f"\nTesting with {pe_method} positional encoding...")
        
        try:
            model = create_mathematical_reasoning_model(
                pe_method=pe_method,
                base_model="microsoft/DialoGPT-medium",  # Smaller model for testing
                load_in_4bit=False,
                use_lora=True
            )
            
            print(f"✓ Successfully created model with {pe_method}")
            
            # Test solving a simple math problem
            problem = "What is 2 + 3 * 4?"
            solution = model.solve_math_problem(problem, max_length=100)
            print(f"Problem: {problem}")
            print(f"Solution: {solution}")
            
        except Exception as e:
            print(f"✗ Failed to create model with {pe_method}: {e}")
    
    print("\nModel testing completed!")