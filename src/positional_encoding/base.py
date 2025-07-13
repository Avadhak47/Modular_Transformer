"""
Base positional encoding class for the modular transformer.
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BasePositionalEncoding(nn.Module, ABC):
    """
    Abstract base class for all positional encoding implementations.
    Provides a unified interface for different positional encoding strategies.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize base positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        
        # Validate inputs
        assert d_model > 0, "d_model must be positive"
        assert max_len > 0, "max_len must be positive" 
        assert 0 <= dropout <= 1, "dropout must be between 0 and 1"
    
    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Apply positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Positionally encoded tensor of same shape
        """
        pass
    
    @abstractmethod
    def get_encoding_type(self) -> str:
        """Return the name of the encoding type."""
        pass
    
    def can_extrapolate(self) -> bool:
        """
        Whether this encoding can handle sequences longer than max_len.
        Override in subclasses that support extrapolation.
        """
        return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this positional encoding."""
        return {
            "type": self.get_encoding_type(),
            "d_model": self.d_model,
            "max_len": self.max_len,
            "can_extrapolate": self.can_extrapolate(),
            "parameters": sum(p.numel() for p in self.parameters())
        }