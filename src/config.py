"""
Model configuration for the modular transformer.
"""
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ModelConfig:
    """Configuration class for the transformer model."""
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    vocab_size: int = 32000
    max_seq_len: int = 512
    dropout: float = 0.1
    positional_encoding: str = "sinusoidal"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.positional_encoding in [
            "sinusoidal", "rope", "alibi", "diet", "t5_relative", "nope"
        ], f"Unknown positional encoding: {self.positional_encoding}"
        assert self.d_model > 0, "d_model must be positive"
        assert self.n_heads > 0, "n_heads must be positive"
        assert self.d_ff > 0, "d_ff must be positive"
        assert self.n_encoder_layers > 0, "n_encoder_layers must be positive"
        assert self.n_decoder_layers > 0, "n_decoder_layers must be positive"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert 0 <= self.dropout <= 1, "dropout must be between 0 and 1"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "n_encoder_layers": self.n_encoder_layers,
            "n_decoder_layers": self.n_decoder_layers,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "positional_encoding": self.positional_encoding
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def get_preset(cls, size: str = "small", positional_encoding: str = "sinusoidal") -> "ModelConfig":
        """
        Return a ModelConfig for a given size preset.
        Presets:
        - small: ~current (d_model=512, n_heads=8, d_ff=2048, 6 layers)
        - medium: ~10x params (d_model=1600, n_heads=16, d_ff=6400, 12 layers)
        - large: ~100x params (d_model=5000, n_heads=32, d_ff=20000, 24 layers)
        """
        if size == "small":
            return cls(
                d_model=512,
                n_heads=8,
                d_ff=2048,
                n_encoder_layers=6,
                n_decoder_layers=6,
                vocab_size=32000,
                max_seq_len=512,
                dropout=0.1,
                positional_encoding=positional_encoding
            )
        elif size == "medium":
            return cls(
                d_model=1600,
                n_heads=16,
                d_ff=6400,
                n_encoder_layers=12,
                n_decoder_layers=12,
                vocab_size=32000,
                max_seq_len=1024,
                dropout=0.1,
                positional_encoding=positional_encoding
            )
        elif size == "large":
            return cls(
                d_model=5120,
                n_heads=32,
                d_ff=20000,
                n_encoder_layers=24,
                n_decoder_layers=24,
                vocab_size=32000,
                max_seq_len=2048,
                dropout=0.1,
                positional_encoding=positional_encoding
            )
        else:
            raise ValueError(f"Unknown model size: {size}")