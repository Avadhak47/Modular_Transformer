"""
DIET (Dynamic Iterative Embedding Transformation) Positional Encoding

A learnable positional encoding method that adapts to mathematical reasoning tasks.
"""

import torch
import torch.nn as nn
from typing import Optional


class DIETPositionalEncoding(nn.Module):
    """DIET Positional Encoding for mathematical reasoning."""
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 32768,
        compression_ratio: float = 0.25,
        math_enhanced: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.compression_ratio = compression_ratio
        self.math_enhanced = math_enhanced
        
        # Learnable position embeddings
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        
        # Compression layer
        compressed_dim = int(d_model * compression_ratio)
        self.compression = nn.Linear(d_model, compressed_dim)
        self.decompression = nn.Linear(compressed_dim, d_model)
        
        # Mathematical enhancement layers
        if math_enhanced:
            self.math_transform = nn.Linear(d_model, d_model)
            self.math_gate = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DIET positional encoding."""
        batch_size, seq_len, d_model = x.shape
        
        # Get position indices
        positions = torch.arange(seq_len, device=x.device)
        
        # Get position embeddings
        pos_embeddings = self.position_embeddings(positions)
        
        # Compress and decompress for efficiency
        compressed = self.compression(pos_embeddings)
        decompressed = self.decompression(compressed)
        
        # Apply mathematical enhancement
        if self.math_enhanced:
            enhanced = self.math_transform(decompressed)
            gate = torch.sigmoid(self.math_gate(x))
            decompressed = decompressed + gate * enhanced
        
        return x + decompressed