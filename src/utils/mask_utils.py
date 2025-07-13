"""
Utility functions for creating attention masks.
"""
import torch
from typing import Optional


def create_padding_mask(seq: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    Create padding mask for sequences.
    
    Args:
        seq: Input sequence tensor of shape (batch_size, seq_len)
        pad_token_id: ID of padding token
        
    Returns:
        Mask tensor of shape (batch_size, 1, 1, seq_len)
    """
    return (seq != pad_token_id).unsqueeze(1).unsqueeze(2)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal (lower triangular) mask for decoder self-attention.
    
    Args:
        seq_len: Sequence length
        device: Device to create tensor on
        
    Returns:
        Causal mask tensor of shape (1, 1, seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(0)


def combine_masks(mask1: Optional[torch.Tensor], mask2: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Combine two masks using logical AND.
    
    Args:
        mask1: First mask tensor
        mask2: Second mask tensor
        
    Returns:
        Combined mask or None if both inputs are None
    """
    if mask1 is None and mask2 is None:
        return None
    elif mask1 is None:
        return mask2
    elif mask2 is None:
        return mask1
    else:
        return mask1 & mask2


def create_look_ahead_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create look-ahead mask to prevent attention to future positions.
    
    Args:
        seq_len: Sequence length
        device: Device to create tensor on
        
    Returns:
        Look-ahead mask tensor
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask == 0
