"""
Tests for transformer layer implementations.
"""
import pytest
import torch
from src.layers import (
    MultiHeadAttention,
    PositionWiseFeedForward,
    LayerNorm,
    TransformerEncoderLayer,
    TransformerDecoderLayer
)


class TestTransformerLayers:
    """Test cases for transformer layers."""
    
    def test_multi_head_attention(self):
        """Test multi-head attention layer."""
        d_model = 256
        n_heads = 8
        seq_len = 32
        batch_size = 2
        
        attention = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, attention_weights = attention(x, x, x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, n_heads, seq_len, seq_len)
    
    def test_feed_forward(self):
        """Test position-wise feed-forward network."""
        d_model = 256
        d_ff = 1024
        seq_len = 32
        batch_size = 2
        
        ff = PositionWiseFeedForward(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = ff(x)
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_layer_norm(self):
        """Test layer normalization."""
        d_model = 256
        seq_len = 32
        batch_size = 2
        
        layer_norm = LayerNorm(d_model)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = layer_norm(x)
        assert output.shape == (batch_size, seq_len, d_model)
        
        # Check normalization properties
        mean = output.mean(dim=-1)
        var = output.var(dim=-1, unbiased=False)
        
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-5)
    
    def test_encoder_layer(self):
        """Test transformer encoder layer."""
        d_model = 256
        n_heads = 8
        d_ff = 1024
        seq_len = 32
        batch_size = 2
        
        encoder_layer = TransformerEncoderLayer(d_model, n_heads, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = encoder_layer(x)
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_decoder_layer(self):
        """Test transformer decoder layer."""
        d_model = 256
        n_heads = 8
        d_ff = 1024
        seq_len = 32
        batch_size = 2
        
        decoder_layer = TransformerDecoderLayer(d_model, n_heads, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        encoder_output = torch.randn(batch_size, seq_len, d_model)
        
        output = decoder_layer(x, encoder_output)
        assert output.shape == (batch_size, seq_len, d_model)
