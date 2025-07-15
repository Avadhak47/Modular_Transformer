"""
Configuration module for the modular transformer.
Provides configuration classes and factory functions.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
from src.config import ModelConfig


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    min_delta: float = 1e-4
    max_steps: int = 100000
    warmup_steps: int = 4000
    gradient_clip: float = 1.0
    optimizer: str = 'adamw'
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-8
    weight_decay: float = 0.01
    use_wandb: bool = False
    project_name: str = 'modular-transformer'
    experiment_name: str = 'default'
    checkpoint_dir: str = 'checkpoints'
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    patience: int = 5


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig
    training: TrainingConfig
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model': self.model.to_dict(),
            'training': self.training.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        model_config = ModelConfig.from_dict(config_dict['model'])
        training_config = TrainingConfig(**config_dict['training'])
        return cls(model=model_config, training=training_config)
    
    def save(self, filepath: str):
        """Save configuration to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "ExperimentConfig":
        """Load configuration from file."""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def get_config(pe_type: str = "sinusoidal") -> ExperimentConfig:
    """
    Get configuration for a specific positional encoding type.
    
    Args:
        pe_type: Positional encoding type
        
    Returns:
        ExperimentConfig with model and training settings
    """
    # Create model config with specified positional encoding
    model_config = ModelConfig(positional_encoding=pe_type)
    
    # Create training config
    training_config = TrainingConfig()
    
    # Customize training config based on positional encoding
    if pe_type == "rope":
        training_config.learning_rate = 1e-4
        training_config.experiment_name = f"transformer_rope"
    elif pe_type == "alibi":
        training_config.learning_rate = 1e-4
        training_config.experiment_name = f"transformer_alibi"
    elif pe_type == "diet":
        training_config.learning_rate = 1e-4
        training_config.experiment_name = f"transformer_diet"
    elif pe_type == "t5_relative":
        training_config.learning_rate = 1e-4
        training_config.experiment_name = f"transformer_t5_relative"
    elif pe_type == "nope":
        training_config.learning_rate = 1e-4
        training_config.experiment_name = f"transformer_nope"
    else:  # sinusoidal
        training_config.learning_rate = 1e-4
        training_config.experiment_name = f"transformer_sinusoidal"
    
    return ExperimentConfig(model=model_config, training=training_config)


def create_experiment_configs() -> Dict[str, ExperimentConfig]:
    """
    Create configurations for all positional encoding types.
    
    Returns:
        Dictionary mapping PE types to configurations
    """
    pe_types = ["sinusoidal", "rope", "alibi", "diet", "t5_relative", "nope"]
    configs = {}
    
    for pe_type in pe_types:
        configs[pe_type] = get_config(pe_type)
    
    return configs