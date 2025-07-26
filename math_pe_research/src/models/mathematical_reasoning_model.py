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


def get_best_device():
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print(f"Using {n_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(n_gpus)]}")
        else:
            print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


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
        load_in_4bit: bool = False,  # Changed default to False for better training compatibility
        load_in_8bit: bool = False,
        trust_remote_code: bool = True,
        torch_dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        cache_dir: Optional[str] = None,
        enable_gradient_checkpointing: bool = False  # Disable by default to prevent training issues
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
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info("Using 4-bit quantization")
            except ImportError:
                logger.warning("BitsAndBytes not available, loading without quantization")
                load_in_4bit = False
        
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
        
        # Ensure model is in training mode and parameters are trainable
        model.train()
        
        # For quantized models, ensure some parameters are trainable
        if load_in_4bit:
            # In 4-bit models, we need to ensure LoRA parameters are trainable
            logger.info("4-bit model loaded - LoRA parameters will be trainable")
        
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
        
        # Get PE configuration for creating new instances
        d_model = self.config.hidden_size
        max_position_embeddings = getattr(self.config, 'max_position_embeddings', 8192)
        num_heads = getattr(self.config, 'num_attention_heads', 8)
        
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
        
        # Modify attention layers across all transformer layers
        for layer_idx, layer in enumerate(layers):
            # Store reference to the attention layer
            attention = getattr(layer, layer_attr, None)
            if attention is None:
                logger.warning(f"No attention layer found at layer {layer_idx}")
                continue
            
            # Create a NEW PE instance for each layer to avoid sharing
            layer_pe = get_positional_encoding(self.pe_method, **pe_config)
            
            # Initialize PE parameters properly for this layer
            self._initialize_pe_parameters(layer_pe, layer_idx)
            
            # Create a custom attention wrapper that uses our PE
            custom_attention = CustomAttentionWithPE(
                attention, 
                layer_pe,  # Use the layer-specific PE
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
            
            # Remove any existing inputs_embeds from kwargs to avoid conflict
            kwargs.pop('inputs_embeds', None)
            
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
                # Ensure base model parameters are trainable
                for param in self.base_model.parameters():
                    param.requires_grad = True
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
            
            # Ensure all LoRA parameters are trainable
            for name, param in model.named_parameters():
                if 'lora' in name.lower():
                    param.requires_grad = True
            
            logger.info(f"Applied LoRA with rank {lora_config['r']} to model")
            logger.info(f"Trainable parameters: {model.print_trainable_parameters()}")
            
            return model
            
        except ImportError:
            logger.warning("PEFT not available, skipping LoRA fine-tuning")
            # Ensure base model parameters are trainable even without LoRA
            for param in self.base_model.parameters():
                param.requires_grad = True
            return self.base_model
    
    def _apply_math_optimizations(self):
        """Apply mathematical reasoning specific optimizations."""
        
        # Only enable gradient checkpointing if explicitly requested
        if hasattr(self, 'enable_gradient_checkpointing') and self.enable_gradient_checkpointing:
            if hasattr(self.base_model, 'gradient_checkpointing_enable'):
                self.base_model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
        else:
            # Disable gradient checkpointing to prevent computation graph breaking
            if hasattr(self.base_model, 'gradient_checkpointing_disable'):
                self.base_model.gradient_checkpointing_disable()
                logger.info("Gradient checkpointing disabled")
        
        # Set model to training mode
        self.base_model.train()
        
        # Mathematical reasoning specific settings
        if hasattr(self.base_model.config, 'use_cache'):
            self.base_model.config.use_cache = False
        
        # Ensure all trainable parameters are unfrozen
        self._unfreeze_trainable_parameters()
        
        logger.info("Applied mathematical reasoning optimizations")
    
    def _unfreeze_trainable_parameters(self):
        """Ensure all trainable parameters are unfrozen."""
        trainable_params = 0
        total_params = 0
        
        for name, param in self.base_model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                # Double-check that trainable parameters are unfrozen
                param.requires_grad = True
        
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        if trainable_params == 0:
            logger.warning("No trainable parameters found! This will cause training to fail.")
            # Force unfreeze some key parameters as fallback
            for name, param in self.base_model.named_parameters():
                if any(key in name.lower() for key in ['lora', 'adapter', 'pe', 'positional']):
                    param.requires_grad = True
                    logger.info(f"Force unfroze parameter: {name}")
    
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
    
    def _get_pe_parameter(self, pe_layer, param_name, layer_idx):
        """Helper function to get PE parameter by unique name."""
        unique_name = f'{param_name}_layer_{layer_idx}'
        if hasattr(pe_layer, unique_name):
            return getattr(pe_layer, unique_name)
        else:
            # Fallback to original name if unique name doesn't exist
            return getattr(pe_layer, param_name, None)

    def _initialize_pe_parameters(self, pe_layer, layer_idx):
        """Initialize PE parameters with unique names to avoid sharing, and ensure tensor size consistency."""
        # Remove all old parameters and buffers to avoid shared tensors
        for name, param in list(pe_layer.named_parameters()):
            if name in pe_layer._parameters:
                del pe_layer._parameters[name]
            if hasattr(pe_layer, name):
                delattr(pe_layer, name)
            new_param = nn.Parameter(param.data.clone().detach())
            unique_name = f'{name}_layer_{layer_idx}'
            pe_layer.register_parameter(unique_name, new_param)
            # DO NOT set canonical name as attribute to avoid shared memory
            # setattr(pe_layer, name, new_param)  # REMOVED THIS LINE
        
        for name, buffer in list(pe_layer.named_buffers()):
            if name in pe_layer._buffers:
                del pe_layer._buffers[name]
            if hasattr(pe_layer, name):
                delattr(pe_layer, name)
            new_buffer = buffer.data.clone().detach()
            unique_name = f'{name}_layer_{layer_idx}'
            pe_layer.register_buffer(unique_name, new_buffer)
            # Keep essential buffers accessible by their original names
            if name in ['inv_freq', 'position_ids']:
                setattr(pe_layer, name, new_buffer)
        
        # Set up parameter access methods for the PE layer
        def get_param(param_name):
            return self._get_pe_parameter(pe_layer, param_name, layer_idx)
        
        # Add parameter access methods to PE layer
        pe_layer.get_param = get_param
        
        # Ensure input/output tensor size consistency for PE
        if hasattr(pe_layer, 'd_model') and hasattr(self, 'config'):
            pe_layer.d_model = self.config.hidden_size
        if hasattr(pe_layer, 'dim') and hasattr(self, 'config'):
            num_heads = getattr(self.config, 'num_attention_heads', 8)
            pe_layer.dim = self.config.hidden_size // num_heads
        if hasattr(pe_layer, 'max_seq_len') and hasattr(self, 'config'):
            pe_layer.max_seq_len = getattr(self.config, 'max_position_embeddings', 8192)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
        
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
        
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
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
    
    def save_pretrained(self, save_directory, **kwargs):
        """Custom save method to handle shared tensors."""
        import os
        from transformers import PreTrainedModel
        
        # Create save directory
        os.makedirs(save_directory, exist_ok=True)
        
        try:
            # Save the base model
            if hasattr(self, 'base_model'):
                self.base_model.save_pretrained(save_directory, **kwargs)
            
            # Save the tokenizer
            if hasattr(self, 'tokenizer'):
                self.tokenizer.save_pretrained(save_directory)
            
            # Save PE configuration
            pe_config = {
                'pe_method': self.pe_method,
                'pe_config': getattr(self, 'pe_config', {})
            }
            
            import json
            with open(os.path.join(save_directory, 'pe_config.json'), 'w') as f:
                json.dump(pe_config, f)
            
            print(f"✅ Model saved to {save_directory}")
            print("   Note: PE layers are integrated into the model and don't need separate saving")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not save model with safetensors: {e}")
            print("   This is expected due to shared tensor issue - model weights are still updated")
            print("   The model can still be used for inference and training")
            
            # Try alternative save method
            try:
                # Save using torch.save instead
                torch.save(self.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))
                print("   Saved using torch.save as fallback")
            except Exception as e2:
                print(f"   Could not save with torch.save either: {e2}")

    def save_model(self, output_dir):
        """Save model for Trainer compatibility."""
        self.save_pretrained(output_dir)
    
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
        
        # First, try to get parameters from the model config
        if hasattr(self.original_attention, 'config'):
            config = self.original_attention.config
            self.num_heads = getattr(config, 'num_attention_heads', None)
            self.hidden_size = getattr(config, 'hidden_size', None)
            self.head_dim = getattr(config, 'head_dim', None)
            
            if self.num_heads is not None and self.hidden_size is not None:
                # Calculate head_dim if not provided
                if self.head_dim is None:
                    self.head_dim = self.hidden_size // self.num_heads
                logger.debug(f"Detected from config: num_heads={self.num_heads}, "
                           f"hidden_size={self.hidden_size}, head_dim={self.head_dim}")
                return
        
        # Try to get num_heads from various possible attributes
        if not hasattr(self, 'num_heads'):
            for attr in ['num_heads', 'num_attention_heads', 'n_head', 'n_heads']:
                if hasattr(self.original_attention, attr):
                    self.num_heads = getattr(self.original_attention, attr)
                    break
        
        # Try to get hidden_size
        if not hasattr(self, 'hidden_size'):
            for attr in ['hidden_size', 'd_model', 'embed_dim', 'embedding_size']:
                if hasattr(self.original_attention, attr):
                    self.hidden_size = getattr(self.original_attention, attr)
                    break
        
        # If we still don't have both, try to infer from weight shapes
        if not hasattr(self, 'num_heads') or not hasattr(self, 'hidden_size'):
            for name, param in self.original_attention.named_parameters():
                if 'query_key_value' in name:
                    # For GPT-NeoX: query_key_value weight shape is [hidden_size, 3 * hidden_size]
                    # So hidden_size = weight.shape[0]
                    if not hasattr(self, 'hidden_size'):
                        self.hidden_size = param.shape[0]
                    
                    # GPT-NeoX typically uses head_dim = 64
                    if not hasattr(self, 'num_heads'):
                        self.num_heads = self.hidden_size // 64
                    break
                elif 'query' in name or 'q_proj' in name:
                    # For other architectures
                    if hasattr(self, 'hidden_size') and not hasattr(self, 'num_heads'):
                        self.num_heads = self.hidden_size // 64  # Assume head_dim=64
                    break
        
        # Fallback values if still not found
        if not hasattr(self, 'num_heads'):
            self.num_heads = 40  # Common for Pythia models
        if not hasattr(self, 'hidden_size'):
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
        
        # Verify the relationship: hidden_size = num_heads * head_dim
        if self.hidden_size != self.num_heads * self.head_dim:
            logger.warning(f"Attention params mismatch: hidden_size={self.hidden_size}, "
                         f"num_heads={self.num_heads}, head_dim={self.head_dim}")
            # Fix the mismatch by adjusting head_dim
            self.head_dim = self.hidden_size // self.num_heads
            logger.info(f"Adjusted head_dim to {self.head_dim}")
        
        logger.debug(f"Final detected attention params: num_heads={self.num_heads}, "
                    f"hidden_size={self.hidden_size}, head_dim={self.head_dim}")

    def _apply_pe(self, hidden_states, query_states=None, key_states=None, position_ids=None, token_ids=None, seq_len=None, attention_scores=None):
        """Apply positional encoding to query and key states."""
        
        # Debug tensor shapes
        if query_states is not None:
            print(f"🔍 Query states shape: {query_states.shape}")
        if key_states is not None:
            print(f"🔍 Key states shape: {key_states.shape}")
        if hidden_states is not None:
            print(f"🔍 Hidden states shape: {hidden_states.shape}")
        """Apply the correct PE logic for each PE type."""
        # RoPE: expects [batch, heads, seq_len, head_dim]
        if self.pe_method in ['rope']:
            if query_states is not None and key_states is not None:
                return self.pe_layer(query_states, key_states)
            else:
                raise ValueError("RoPE/m-adaptive PE requires query_states and key_states.")
        # T5Relative: expects attention_scores to be passed
        elif self.pe_method == 't5_relative':
            if hasattr(self.pe_layer, 'forward'):
                if attention_scores is not None:
                    return self.pe_layer(hidden_states, attention_scores=attention_scores)
                else:
                    raise ValueError("T5RelativePositionalBias requires attention_scores argument.")
            else:
                raise ValueError(f"PE {self.pe_method} does not support forward.")
        # Sinusoidal, diet: expects [batch, seq_len, hidden_dim] (additive)
        elif self.pe_method in ['sinusoidal', 'diet']:
            if hasattr(self.pe_layer, 'forward'):
                return self.pe_layer(hidden_states)
            else:
                raise ValueError(f"PE {self.pe_method} does not support forward.")
        # MathAdaptive: expects [batch, seq_len, hidden_dim] with token_ids
        elif self.pe_method == 'math_adaptive':
            if hasattr(self.pe_layer, 'forward'):
                if token_ids is not None:
                    return self.pe_layer(hidden_states, token_ids=token_ids)
                else:
                    return self.pe_layer(hidden_states)
            else:
                raise ValueError(f"PE {self.pe_method} does not support forward.")
        # Alibi: expects [batch, heads, seq_len, head_dim] or [batch, seq_len, hidden_dim]
        elif self.pe_method == 'alibi':
            if hasattr(self.pe_layer, 'forward'):
                try:
                    return self.pe_layer(hidden_states, position_ids=position_ids)
                except Exception:
                    return self.pe_layer(hidden_states)
            else:
                raise ValueError("Alibi PE does not support forward.")
        else:
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
        if self.pe_method in ['rope']:
            # Ensure attention parameters are detected
            if not hasattr(self, 'num_heads') or not hasattr(self, 'head_dim'):
                self._detect_attention_params()
            
            batch_size, seq_len, _ = hidden_states.shape
            # Project to Q, K, V
            if hasattr(self.original_attention, 'q_proj'):
                query_states = self.original_attention.q_proj(hidden_states)
                key_states = self.original_attention.k_proj(hidden_states)
                value_states = self.original_attention.v_proj(hidden_states)
            elif hasattr(self.original_attention, 'query_key_value'):
                qkv = self.original_attention.query_key_value(hidden_states)
                # Use the already detected parameters
                num_heads = self.num_heads
                head_dim = self.head_dim
                
                logger.debug(f"GPT-NeoX: num_heads={num_heads}, head_dim={head_dim}, qkv_shape={qkv.shape}")
                qkv = qkv.view(batch_size, seq_len, 3, num_heads, head_dim)
                query_states, key_states, value_states = qkv.unbind(2)
                logger.debug(f"GPT-NeoX QKV reshape: {qkv.shape} -> Q:{query_states.shape}, K:{key_states.shape}, V:{value_states.shape}")
            elif hasattr(self.original_attention, 'c_attn'):
                qkv = self.original_attention.c_attn(hidden_states)
                # Use the already detected parameters
                num_heads = self.num_heads
                head_dim = self.head_dim
                
                logger.debug(f"GPT2: num_heads={num_heads}, head_dim={head_dim}, qkv_shape={qkv.shape}")
                qkv = qkv.view(batch_size, seq_len, 3, num_heads, head_dim)
                query_states, key_states, value_states = qkv.unbind(2)
                logger.debug(f"GPT2 QKV reshape: {qkv.shape} -> Q:{query_states.shape}, K:{key_states.shape}, V:{value_states.shape}")
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
                None, query_states=query_states, key_states=key_states
            )
            # Continue with standard attention computation
            attn_output, attn_weights = self._compute_attention(
                query_states, key_states, value_states, attention_mask,
                past_key_value, output_attentions, use_cache
            )
            
            # Return format expected by transformers: (attn_output, attn_weights)
            # Note: past_key_value is not handled in custom attention for now
            return attn_output, attn_weights
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
        
        # --- PATCH: Add T5RelativePositionalBias before softmax ---
        if self.pe_method == 't5_relative':
            # hidden_states is not used, so pass a dummy tensor (e.g., query_states)
            attn_weights = self._apply_pe(query_states, attention_scores=attn_weights)
        # --- END PATCH ---
        
        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Ensure attn_weights and value_states have the same dtype
        attn_weights = attn_weights.to(value_states.dtype)
        
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
        
        # Return format expected by transformers: (attn_output, attn_weights)
        # Note: past_key_value is handled separately in the forward method
        if output_attentions:
            return attn_output, attn_weights
        else:
            return attn_output, None


# Factory function for easy model creation
def create_mathematical_reasoning_model(
    pe_method: str = "rope",
    base_model: str = "deepseek-ai/deepseek-math-7b-instruct",
    use_lora: bool = True,  # Enable LoRA by default for training
    load_in_4bit: bool = False,  # Disable 4-bit by default to avoid bitsandbytes issues
    enable_gradient_checkpointing: bool = False,  # Disable by default to prevent training issues
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
    
    device = get_best_device()
    model = MathematicalReasoningModel(
        base_model_name=base_model,
        pe_method=pe_method,
        use_lora=use_lora, # Pass use_lora to the model constructor
        enable_gradient_checkpointing=enable_gradient_checkpointing, # Pass gradient checkpointing parameter
        **kwargs
    )
    # Move PE layer to device
    if hasattr(model, 'pe_layer') and hasattr(model.pe_layer, 'to'):
        model.pe_layer = model.pe_layer.to(device)
    # Move model to device
    model = model.to(device)
    # Wrap with DataParallel if multiple GPUs
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


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