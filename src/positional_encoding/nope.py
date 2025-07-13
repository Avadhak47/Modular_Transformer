"""
NoPE (No Positional Encoding) implementation.
"""
import torch
import torch.nn as nn
from typing import Dict, Any


class NoPositionalEncoding(nn.Module):
    """
    No Positional Encoding (NoPE) baseline.
    This class serves as a baseline that doesn't add any positional information.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize NoPE (effectively a pass-through with dropout).
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length (unused but kept for interface consistency)
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply no positional encoding - just return input with dropout.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Input tensor with dropout applied (no positional encoding added)
        """
        return self.dropout(x)
    
    def get_encoding_type(self) -> str:
        """Return the encoding type name."""
        return "nope"
    
    def can_extrapolate(self) -> bool:
        """NoPE can handle any sequence length since it doesn't encode position."""
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this positional encoding."""
        return {
            "type": self.get_encoding_type(),
            "d_model": self.d_model,
            "max_len": self.max_len,
            "can_extrapolate": self.can_extrapolate(),
            "parameters": 0,  # No additional parameters
            "description": "No positional encoding - relies on attention mechanism alone"
        }
    
    def reset_parameters(self):
        """Reset parameters (no-op for NoPE since there are no parameters)."""
        pass