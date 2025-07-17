"""
Mathematical Reasoning Model with Advanced Positional Encoding

Integrates DeepSeekMath base models with state-of-the-art positional encoding
methods for enhanced mathematical reasoning capabilities.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    LlamaForCausalLM, LlamaConfig,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, Dict, Any, Tuple
import logging

from ..positional_encoding import get_positional_encoding

logger = logging.getLogger(__name__)


class MathematicalReasoningModel(nn.Module):
    """
    Mathematical Reasoning Model with Advanced Positional Encoding
    
    Features:
    - DeepSeekMath base model integration
    - Pluggable positional encoding methods
    - LoRA fine-tuning support
    - Mathematical reasoning optimizations
    """
    
    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-math-7b-instruct",
        pe_type: str = "rope",
        pe_config: Optional[Dict[str, Any]] = None,
        use_lora: bool = True,
        lora_config: Optional[Dict[str, Any]] = None,
        quantization_config: Optional[Dict[str, Any]] = None,
        max_seq_len: int = 4096,
        math_enhanced: bool = True,
        device_map: Optional[str] = "auto"
    ):
        super().__init__()
        
        self.model_name = model_name
        self.pe_type = pe_type
        self.max_seq_len = max_seq_len
        self.math_enhanced = math_enhanced
        
        # Setup quantization if specified
        quantization_kwargs = {}
        if quantization_config:
            quantization_kwargs["quantization_config"] = self._setup_quantization(quantization_config)
        
        # Load base model
        self.config = AutoConfig.from_pretrained(model_name)
        self.base_model = self._load_base_model(device_map, **quantization_kwargs)
        
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Replace positional encoding
        self._setup_positional_encoding(pe_config or {})
        
        # Setup LoRA if specified
        if use_lora:
            self.base_model = self._setup_lora(lora_config or {})
        
        # Mathematical reasoning enhancements
        if math_enhanced:
            self._add_mathematical_enhancements()
    
    def _load_base_model(self, device_map: Optional[str], **kwargs):
        """Load the base DeepSeekMath model."""
        try:
            model = LlamaForCausalLM.from_pretrained(
                self.model_name,
                config=self.config,
                device_map=device_map,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                **kwargs
            )
            logger.info(f"Successfully loaded model: {self.model_name}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}, using Llama-2-7B as fallback: {e}")
            # Fallback to a standard Llama model
            fallback_model = "meta-llama/Llama-2-7b-chat-hf"
            return LlamaForCausalLM.from_pretrained(
                fallback_model,
                device_map=device_map,
                torch_dtype=torch.float16,
                **kwargs
            )
    
    def _setup_quantization(self, config: Dict[str, Any]) -> BitsAndBytesConfig:
        """Setup quantization configuration."""
        return BitsAndBytesConfig(
            load_in_4bit=config.get("load_in_4bit", True),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type=config.get("quant_type", "nf4"),
            bnb_4bit_use_double_quant=config.get("use_double_quant", True),
        )
    
    def _setup_positional_encoding(self, pe_config: Dict[str, Any]):
        """Replace the model's positional encoding with the specified type."""
        # Get model dimensions
        hidden_size = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        head_dim = hidden_size // num_heads
        
        # Create positional encoding
        pe_config.update({
            "d_model": hidden_size if self.pe_type in ["sinusoidal", "diet"] else head_dim,
            "dim": head_dim if self.pe_type == "rope" else hidden_size,
            "num_heads": num_heads,
            "max_seq_len": self.max_seq_len
        })
        
        self.positional_encoding = get_positional_encoding(self.pe_type, **pe_config)
        
        # Store original PE for reference
        self.original_pe_type = getattr(self.config, 'position_embedding_type', 'rope')
        
        logger.info(f"Replaced positional encoding with {self.pe_type}")
    
    def _setup_lora(self, lora_config: Dict[str, Any]) -> nn.Module:
        """Setup LoRA fine-tuning."""
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.get("r", 64),
            lora_alpha=lora_config.get("alpha", 16),
            lora_dropout=lora_config.get("dropout", 0.1),
            target_modules=lora_config.get(
                "target_modules", 
                ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            ),
            bias="none",
            use_rslora=lora_config.get("use_rslora", True),
        )
        
        model = get_peft_model(self.base_model, config)
        logger.info(f"Applied LoRA with rank {config.r}")
        return model
    
    def _add_mathematical_enhancements(self):
        """Add mathematical reasoning specific enhancements."""
        hidden_size = self.config.hidden_size
        
        # Mathematical pattern recognition layer
        self.math_pattern_detector = nn.Linear(hidden_size, hidden_size)
        
        # Mathematical operation type classifier
        self.math_op_classifier = nn.Linear(hidden_size, 10)  # 10 math operation types
        
        # Mathematical confidence estimator
        self.confidence_estimator = nn.Linear(hidden_size, 1)
        
        # Mathematical reasoning head
        self.math_reasoning_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        logger.info("Added mathematical reasoning enhancements")
    
    def _apply_positional_encoding(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Apply the selected positional encoding method."""
        if self.pe_type == "sinusoidal":
            return self.positional_encoding(hidden_states)
        
        elif self.pe_type in ["rope", "diet", "t5_relative"]:
            # These are applied in attention layers, return unchanged
            return hidden_states
        
        elif self.pe_type == "alibi":
            # ALiBi is applied to attention scores, return unchanged
            return hidden_states
        
        else:
            logger.warning(f"Unknown PE type: {self.pe_type}, using original")
            return hidden_states
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        output_mathematical_features: bool = False,
        **kwargs
    ):
        """
        Forward pass with mathematical reasoning enhancements.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels for training
            return_dict: Whether to return ModelOutput
            output_mathematical_features: Whether to return mathematical features
            
        Returns:
            Model outputs with optional mathematical features
        """
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            output_hidden_states=self.math_enhanced,
            **kwargs
        )
        
        mathematical_features = {}
        
        if self.math_enhanced and hasattr(outputs, 'hidden_states'):
            # Extract last hidden state
            last_hidden_state = outputs.hidden_states[-1]
            
            # Apply mathematical enhancements
            math_patterns = self.math_pattern_detector(last_hidden_state)
            math_operations = self.math_op_classifier(last_hidden_state.mean(dim=1))
            confidence = self.confidence_estimator(last_hidden_state.mean(dim=1))
            reasoning_features = self.math_reasoning_head(last_hidden_state)
            
            mathematical_features = {
                'math_patterns': math_patterns,
                'math_operations': math_operations,
                'confidence': torch.sigmoid(confidence),
                'reasoning_features': reasoning_features
            }
        
        if output_mathematical_features:
            if return_dict:
                outputs.mathematical_features = mathematical_features
            else:
                outputs = (*outputs, mathematical_features)
        
        return outputs
    
    def generate_mathematical_solution(
        self,
        problem: str,
        max_length: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate solution for a mathematical problem.
        
        Args:
            problem: Mathematical problem text
            max_length: Maximum generation length
            temperature: Generation temperature
            do_sample: Whether to use sampling
            top_p: Top-p sampling parameter
            
        Returns:
            Dictionary with solution and metadata
        """
        # Format the problem
        formatted_prompt = self._format_mathematical_prompt(problem)
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=self.max_seq_len,
            truncation=True,
            padding=True
        )
        
        # Move to device
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate with mathematical features
        with torch.no_grad():
            generation_outputs = self.base_model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
            # Get mathematical features for the input
            model_outputs = self.forward(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_mathematical_features=True
            )
        
        # Decode solution
        solution = self.tokenizer.decode(
            generation_outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        result = {
            'problem': problem,
            'solution': solution,
            'formatted_prompt': formatted_prompt,
        }
        
        # Add mathematical features if available
        if hasattr(model_outputs, 'mathematical_features'):
            features = model_outputs.mathematical_features
            result.update({
                'confidence': features['confidence'].cpu().numpy(),
                'math_operations': torch.softmax(features['math_operations'], dim=-1).cpu().numpy(),
            })
        
        return result
    
    def _format_mathematical_prompt(self, problem: str) -> str:
        """Format mathematical problem for the model."""
        # DeepSeekMath-style formatting
        return f"""Problem: {problem}

Solution: Let me solve this step by step.

"""
    
    def save_model(self, save_path: str):
        """Save the model and configuration."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save base model
        if hasattr(self.base_model, 'save_pretrained'):
            self.base_model.save_pretrained(save_path)
        else:
            # Handle PEFT models
            self.base_model.save_pretrained(save_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save additional configurations
        config_dict = {
            'pe_type': self.pe_type,
            'max_seq_len': self.max_seq_len,
            'math_enhanced': self.math_enhanced,
            'model_name': self.model_name
        }
        
        import json
        with open(os.path.join(save_path, 'math_model_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load a pretrained mathematical reasoning model."""
        import json
        import os
        
        # Load configuration
        config_path = os.path.join(model_path, 'math_model_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            kwargs.update(saved_config)
        
        # Create model
        model = cls(**kwargs)
        
        # Load state dict if available
        state_dict_path = os.path.join(model_path, 'pytorch_model.bin')
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        
        logger.info(f"Model loaded from {model_path}")
        return model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'pe_type': self.pe_type,
            'max_seq_len': self.max_seq_len,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'trainable_percentage': (trainable_params / total_params) * 100,
            'math_enhanced': self.math_enhanced,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config)
        }


class MathematicalModelFactory:
    """Factory class for creating mathematical reasoning models with different configurations."""
    
    @staticmethod
    def create_node_model(
        node_id: int,
        config: Dict[str, Any]
    ) -> MathematicalReasoningModel:
        """
        Create a model for a specific node configuration.
        
        Args:
            node_id: Node identifier (0-4)
            config: Node configuration dictionary
            
        Returns:
            Configured mathematical reasoning model
        """
        # Extract configuration sections
        model_config = config.get('model_config', {})
        sota_config = config.get('sota_integration', {})
        pe_config = config.get(f'{model_config.get("positional_encoding", "rope")}_config', {})
        
        # Determine model name based on SOTA integration
        base_model = sota_config.get('base_model', 'deepseek-ai/deepseek-math-7b-instruct')
        
        # Setup LoRA configuration
        lora_config = {
            'r': sota_config.get('lora_rank', 64),
            'alpha': sota_config.get('lora_alpha', 16),
            'dropout': sota_config.get('lora_dropout', 0.1),
            'target_modules': sota_config.get('target_modules', [
                "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
            ])
        }
        
        # Setup quantization if needed
        quantization_config = {
            'load_in_4bit': True,
            'quant_type': 'nf4',
            'use_double_quant': True
        } if sota_config.get('use_quantization', True) else None
        
        # Create model
        model = MathematicalReasoningModel(
            model_name=base_model,
            pe_type=model_config.get('positional_encoding', 'rope'),
            pe_config=pe_config,
            use_lora=sota_config.get('use_lora', True),
            lora_config=lora_config,
            quantization_config=quantization_config,
            max_seq_len=model_config.get('max_seq_len', 4096),
            math_enhanced=True
        )
        
        logger.info(f"Created model for Node {node_id} with {model_config.get('positional_encoding')} PE")
        return model