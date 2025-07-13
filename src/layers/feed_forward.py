"""
Position-wise Feed-Forward Network implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network as described in the Transformer paper."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = 'relu'):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Set activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unknown activation function: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply position-wise feed-forward transformation."""
        return self.w2(self.dropout(self.activation(self.w1(x))))


class GatedFeedForward(nn.Module):
    """Gated Feed-Forward Network variant."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated feed-forward transformation."""
        gate = torch.sigmoid(self.w2(x))
        hidden = F.relu(self.w1(x))
        gated_hidden = gate * hidden
        return self.w3(self.dropout(gated_hidden))
