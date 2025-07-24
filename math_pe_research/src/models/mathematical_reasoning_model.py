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
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            quantization_config=quantization_config,
            cache_dir=self.cache_dir
        )
        
        # Update our config reference to match the loaded model
        self.config = model.config
        
        # Set use_cache after loading (this is the correct way)
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        
        # Resize token embeddings if we added new tokens
        if len(self.tokenizer) > model.config.vocab_size:
            old_num_tokens = model.config.vocab_size
            model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Resized token embeddings to {len(self.tokenizer)}")
            # Only new rows are randomly initialized; old ones are copied from Pythia
        
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
        
        # For RoPE, use head_dim instead of d_model
        if self.pe_method in ['rope', 'math_adaptive']:
            head_dim = d_model // num_heads
            pe_config['dim'] = head_dim  # RoPE expects 'dim' parameter
        
        # The get_positional_encoding function now handles argument filtering
        new_pe = get_positional_encoding(self.pe_method, **pe_config)
        
        # Replace positional encoding in the model
        # This depends on the model architecture - DeepSeekMath is based on Llama
        self._replace_llama_positional_encoding(new_pe)
    
    def _replace_llama_positional_encoding(self, new_pe: nn.Module):
        """Replace positional encoding in various transformer architectures."""
        
        # For RoPE-like methods, replace at attention level
        if self.pe_method in ['rope', 'math_adaptive']:
            self._replace_attention_level_pe(new_pe)
        else:
            # For additive methods, replace at embedding level
            self._replace_embedding_level_pe(new_pe)
    
    def _replace_attention_level_pe(self, new_pe: nn.Module):
        """Replace PE at the attention level (for RoPE-like methods)."""
        
        # Auto-detect model architecture and get layers
        layers, layer_attr = self._detect_attention_layers()
        
        if layers is None:
            logger.warning(f"Unknown model architecture, skipping PE replacement")
            return
        
        # Modify attention layers across all transformer layers
        for layer_idx, layer in enumerate(layers):
            # Store reference to the attention layer
            attention = getattr(layer, layer_attr, None)
            if attention is None:
                logger.warning(f"No attention layer found at layer {layer_idx}")
                continue
            
            # Initialize PE parameters properly for this layer
            if layer_idx == 0:  # Only initialize once, then reuse
                self._initialize_pe_parameters(new_pe, attention)
            
            # Create a custom attention wrapper that uses our PE
            custom_attention = CustomAttentionWithPE(
                attention, 
                new_pe, 
                self.pe_method,
                layer_idx=layer_idx
            )
            
            # Replace the attention layer
            setattr(layer, layer_attr, custom_attention)
        
        logger.info(f"Replaced positional encoding in {len(layers)} attention layers")
    
    def _replace_embedding_level_pe(self, new_pe: nn.Module):
        """Replace PE at the embedding level (for additive methods)."""
        
        # Store the PE module for embedding-level application
        self.embedding_pe = new_pe
        
        # Wrap the model's forward method to apply PE at embedding level
        original_forward = self.base_model.forward
        
        def forward_with_embedding_pe(*args, **kwargs):
            # Get input_ids from args or kwargs
            if len(args) > 0:
                input_ids = args[0]
                new_args = args[1:]
            else:
                input_ids = kwargs.pop('input_ids')
                new_args = ()
            
            # Apply embedding PE
            inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
            
            # Apply our positional encoding
            batch_size, seq_len = input_ids.shape
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            inputs_embeds = self.embedding_pe(inputs_embeds, position_ids=position_ids)
            
            # Call original forward with inputs_embeds instead of input_ids
            return original_forward(inputs_embeds=inputs_embeds, *new_args, **kwargs)
        
        # Replace the forward method
        self.base_model.forward = forward_with_embedding_pe
        
        logger.info(f"Replaced positional encoding at embedding level")
    
    def _apply_lora(self) -> nn.Module:
        """Apply LoRA (Low-Rank Adaptation) for efficient fine-tuning."""
        
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            # Auto-detect target modules based on model architecture
            target_modules = self._detect_lora_target_modules()
            
            if not target_modules:
                logger.warning("No suitable target modules found for LoRA, skipping LoRA adaptation")
                return self.base_model
            
            # Default LoRA configuration optimized for mathematical reasoning
            default_lora_config = {
                "task_type": TaskType.CAUSAL_LM,
                "r": 64,  # Rank
                "lora_alpha": 16,  # Scaling parameter
                "lora_dropout": 0.1,
                "target_modules": target_modules,
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
    
    def _detect_lora_target_modules(self) -> List[str]:
        """Auto-detect LoRA target modules based on model architecture."""
        target_modules = []
        
        # Get a sample layer to inspect architecture
        sample_layer = None
        
        # Try different model architectures
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'layers'):
            # Llama/DeepSeek style
            if len(self.base_model.model.layers) > 0:
                sample_layer = self.base_model.model.layers[0]
                arch_type = "llama"
        elif hasattr(self.base_model, 'transformer') and hasattr(self.base_model.transformer, 'h'):
            # GPT2/Pythia style
            if len(self.base_model.transformer.h) > 0:
                sample_layer = self.base_model.transformer.h[0]
                arch_type = "gpt"
        elif hasattr(self.base_model, 'gpt_neox') and hasattr(self.base_model.gpt_neox, 'layers'):
            # GPT-NeoX/Pythia style
            if len(self.base_model.gpt_neox.layers) > 0:
                sample_layer = self.base_model.gpt_neox.layers[0]
                arch_type = "neox"
        
        if sample_layer is None:
            logger.warning("Could not detect model architecture for LoRA")
            return []
        
        # Inspect the layer to find attention and MLP modules
        layer_modules = dict(sample_layer.named_modules())
        
        # Common patterns for different architectures
        attention_patterns = [
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",  # Llama
            "attn.c_attn", "attn.c_proj",  # GPT2
            "attention.query_key_value", "attention.dense",  # GPT-NeoX
        ]
        
        mlp_patterns = [
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",  # Llama
            "mlp.c_fc", "mlp.c_proj",  # GPT2
            "mlp.dense_h_to_4h", "mlp.dense_4h_to_h",  # GPT-NeoX
        ]
        
        # Find actual module names that exist
        for pattern in attention_patterns + mlp_patterns:
            if pattern in layer_modules:
                # Extract just the module name (last part)
                module_name = pattern.split('.')[-1]
                if module_name not in target_modules:
                    target_modules.append(module_name)
        
        # Alternative: scan all linear layers
        if not target_modules:
            for name, module in layer_modules.items():
                if isinstance(module, nn.Linear) and '.' in name:
                    module_name = name.split('.')[-1]
                    if module_name not in target_modules and len(module_name) > 1:
                        target_modules.append(module_name)
        
        logger.info(f"Detected LoRA target modules: {target_modules}")
        return target_modules
    
    def _detect_attention_layers(self) -> Tuple[Optional[nn.ModuleList], Optional[str]]:
        """Detect attention layers in different model architectures."""
        
        # Try different model architectures
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'layers'):
            # Llama/DeepSeek style models
            return self.base_model.model.layers, 'self_attn'
            
        elif hasattr(self.base_model, 'transformer') and hasattr(self.base_model.transformer, 'h'):
            # GPT2/DialoGPT style models
            return self.base_model.transformer.h, 'attn'
            
        elif hasattr(self.base_model, 'gpt_neox') and hasattr(self.base_model.gpt_neox, 'layers'):
            # GPT-NeoX/Pythia style models
            return self.base_model.gpt_neox.layers, 'attention'
            
        elif hasattr(self.base_model, 'layers'):
            # Direct layer access
            return self.base_model.layers, 'self_attn'
            
        else:
            # Try to find any attention-like modules
            for attr_name in ['layers', 'blocks', 'transformer_layers']:
                if hasattr(self.base_model, attr_name):
                    layers = getattr(self.base_model, attr_name)
                    if hasattr(layers, '__len__') and len(layers) > 0:
                        # Try to find attention in first layer
                        first_layer = layers[0]
                        for attn_name in ['self_attn', 'attn', 'attention', 'mha']:
                            if hasattr(first_layer, attn_name):
                                return layers, attn_name
        
        return None, None
    
    def _initialize_pe_parameters(self, new_pe: nn.Module, original_attention: nn.Module):
        """Initialize PE parameters with proper parameter mapping and initialization."""
        
        # Get model configuration parameters
        d_model = self.config.hidden_size
        num_heads = getattr(self.config, 'num_attention_heads', 8)
        max_seq_len = getattr(self.config, 'max_position_embeddings', 8192)
        
        # Initialize PE based on type
        if hasattr(new_pe, 'initialize_from_config'):
            # Custom initialization method
            new_pe.initialize_from_config(
                d_model=d_model,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                original_params=dict(original_attention.named_parameters())
            )
        elif hasattr(new_pe, 'reset_parameters'):
            # Standard PyTorch reset
            new_pe.reset_parameters()
        else:
            # Manual initialization for common PE types
            for name, param in new_pe.named_parameters():
                if 'weight' in name:
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.normal_(param, std=0.02)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        
        logger.info(f"Initialized PE parameters for {type(new_pe).__name__}")
    
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
    
    def __init__(self, original_attention: nn.Module, pe_layer: nn.Module, pe_method: str, layer_idx: int = 0):
        super().__init__()
        self.original_attention = original_attention
        self.pe_layer = pe_layer
        self.pe_method = pe_method
        self.layer_idx = layer_idx
        
        # Copy attributes from original attention
        for attr in ['hidden_size', 'num_heads', 'head_dim', 'config']:
            if hasattr(original_attention, attr):
                setattr(self, attr, getattr(original_attention, attr))
        
        # Auto-detect attention parameters if not already set
        self._detect_attention_params()
    
    def _detect_attention_params(self):
        """Auto-detect attention parameters from the original attention module."""
        
        # Try to get num_heads from various possible attributes
        if not hasattr(self, 'num_heads'):
            for attr in ['num_heads', 'num_attention_heads', 'n_head', 'n_heads']:
                if hasattr(self.original_attention, attr):
                    self.num_heads = getattr(self.original_attention, attr)
                    break
            else:
                # Fallback: try to infer from weight shapes
                for name, param in self.original_attention.named_parameters():
                    if 'query' in name or 'q_proj' in name or 'query_key_value' in name:
                        # Assume hidden_size = num_heads * head_dim
                        if hasattr(self, 'hidden_size'):
                            self.num_heads = getattr(self, 'hidden_size', 768) // 64  # Assume head_dim=64
                        else:
                            self.num_heads = 12  # Default fallback
                        break
                else:
                    self.num_heads = 12  # Default fallback
        
        # Try to get hidden_size
        if not hasattr(self, 'hidden_size'):
            for attr in ['hidden_size', 'd_model', 'embed_dim', 'embedding_size']:
                if hasattr(self.original_attention, attr):
                    self.hidden_size = getattr(self.original_attention, attr)
                    break
            else:
                self.hidden_size = self.num_heads * 64  # Default: head_dim=64
        
        # Calculate head_dim
        if not hasattr(self, 'head_dim'):
            # Try to get head_dim from attention module
            for attr in ['head_dim', 'head_size']:
                if hasattr(self.original_attention, attr):
                    self.head_dim = getattr(self.original_attention, attr)
                    break
            else:
                # Calculate from hidden_size and num_heads
                self.head_dim = self.hidden_size // self.num_heads
        
        logger.debug(f"Detected attention params: num_heads={self.num_heads}, "
                    f"hidden_size={self.hidden_size}, head_dim={self.head_dim}")

    def _apply_pe(self, hidden_states, query_states=None, key_states=None, position_ids=None, token_ids=None, seq_len=None):
        """Apply the correct PE logic for each PE type."""
        # RoPE and math_adaptive: expects [batch, heads, seq_len, head_dim]
        if self.pe_method in ['rope', 'math_adaptive']:
            # query_states/key_states: [batch, heads, seq_len, head_dim]
            # If not already in [batch, heads, seq_len, head_dim], transpose
            if query_states is not None and key_states is not None:
                return self.pe_layer(query_states, key_states, seq_len=seq_len)
            else:
                raise ValueError("RoPE/m-adaptive PE requires query_states and key_states.")
        # Sinusoidal, t5_relative, diet: expects [batch, seq_len, hidden_dim] (additive)
        elif self.pe_method in ['sinusoidal', 't5_relative', 'diet']:
            # hidden_states: [batch, seq_len, hidden_dim]
            if hasattr(self.pe_layer, 'forward'):
                return self.pe_layer(hidden_states, position_ids=position_ids)
            else:
                raise ValueError(f"PE {self.pe_method} does not support forward.")
        # Alibi: expects [batch, heads, seq_len, head_dim] or [batch, seq_len, hidden_dim]
        elif self.pe_method == 'alibi':
            # Some Alibi impls expect [batch, heads, seq_len, head_dim], some [batch, seq_len, hidden_dim]
            if hasattr(self.pe_layer, 'forward'):
                try:
                    return self.pe_layer(hidden_states, position_ids=position_ids)
                except Exception:
                    # Try with [batch, heads, seq_len, head_dim]
                    return self.pe_layer(hidden_states)
            else:
                raise ValueError("Alibi PE does not support forward.")
        else:
            # Default: pass through
            return hidden_states

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
        # For RoPE-like methods, we apply PE within attention computation
        if self.pe_method in ['rope', 'math_adaptive']:
            batch_size, seq_len, _ = hidden_states.shape
            # Project to Q, K, V
            if hasattr(self.original_attention, 'q_proj'):
                query_states = self.original_attention.q_proj(hidden_states)
                key_states = self.original_attention.k_proj(hidden_states)
                value_states = self.original_attention.v_proj(hidden_states)
            elif hasattr(self.original_attention, 'query_key_value'):
                qkv = self.original_attention.query_key_value(hidden_states)
                num_heads = getattr(self.original_attention, 'num_attention_heads', 8)
                head_dim = getattr(self.original_attention, 'head_size', 64)
                qkv = qkv.view(batch_size, seq_len, 3, num_heads, head_dim)
                query_states, key_states, value_states = qkv.unbind(2)
            elif hasattr(self.original_attention, 'c_attn'):
                qkv = self.original_attention.c_attn(hidden_states)
                num_heads = self.num_heads
                head_dim = self.head_dim
                qkv = qkv.view(batch_size, seq_len, 3, num_heads, head_dim)
                query_states, key_states, value_states = qkv.unbind(2)
            else:
                logger.warning("Unknown attention architecture, using original attention")
                return self.original_attention(
                    hidden_states, attention_mask, position_ids,
                    past_key_value, output_attentions, use_cache, **kwargs
                )
            # Transpose to [batch, heads, seq_len, head_dim]
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            # Apply PE
            query_states, key_states = self._apply_pe(
                None, query_states=query_states, key_states=key_states, seq_len=seq_len
            )
            # Continue with standard attention computation
            outputs = self._compute_attention(
                query_states, key_states, value_states, attention_mask,
                past_key_value, output_attentions, use_cache
            )
            
            # Ensure we return the expected format (at least 2 elements for unpacking)
            if len(outputs) == 1:
                return (outputs[0], None)  # (attn_output, attn_weights)
            else:
                return outputs
        else:
            # For additive PE, apply to hidden_states before attention
            hidden_states = self._apply_pe(hidden_states, position_ids=position_ids)
            
            # Handle different argument names for different architectures
            if hasattr(self.original_attention, 'query_key_value'):
                # GPT-NeoX uses 'layer_past' instead of 'past_key_value' and expects position_embeddings
                # For non-RoPE PE, we don't provide position_embeddings (let it use original RoPE)
                return self.original_attention(
                    hidden_states, attention_mask, layer_past=past_key_value, 
                    head_mask=None, use_cache=use_cache, output_attentions=output_attentions,
                    position_embeddings=None  # Let it compute its own position embeddings
                )
            else:
                # Standard transformers interface
                return self.original_attention(
                    hidden_states, attention_mask, position_ids,
                    past_key_value, output_attentions, use_cache, **kwargs
                )
    
    def _compute_attention(self, query_states, key_states, value_states, attention_mask,
                          past_key_value, output_attentions, use_cache):
        """Compute multi-head attention."""
        
        batch_size, num_heads, seq_len, head_dim = query_states.shape
        
        # Compute attention scores (tensors are already in [batch, heads, seq_len, head_dim])
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            # Reshape attention mask to match attention weights
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
            attn_weights = attn_weights + attention_mask
        
        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, seq_len, heads, head_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, num_heads * head_dim)  # [batch, seq_len, hidden_size]
        
        # Handle different output projection architectures
        if hasattr(self.original_attention, 'o_proj'):
            # Llama-style: o_proj
            attn_output = self.original_attention.o_proj(attn_output)
        elif hasattr(self.original_attention, 'dense'):
            # GPT-NeoX/GPT2 style: dense
            attn_output = self.original_attention.dense(attn_output)
        elif hasattr(self.original_attention, 'c_proj'):
            # GPT2 style: c_proj
            attn_output = self.original_attention.c_proj(attn_output)
        else:
            logger.warning("Unknown output projection, skipping")
        
        # Return format expected by transformers: (attn_output, attn_weights, present_key_value)
        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        else:
            outputs += (None,)  # Always include attn_weights slot
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