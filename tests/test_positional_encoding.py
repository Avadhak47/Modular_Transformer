"""
Tests for positional encoding implementations.
"""
import pytest
import torch
from src.positional_encoding import (
    SinusoidalPositionalEncoding,
    RotaryPositionalEncoding,
    ALiBiPositionalEncoding,
    NoPositionalEncoding
)


class TestPositionalEncodings:
    """Test cases for positional encoding implementations."""
    
    @pytest.fixture
    def input_tensor(self):
        """Sample input tensor for testing."""
        return torch.randn(2, 32, 256)  # (batch_size, seq_len, d_model)
    
    def test_sinusoidal_pe(self, input_tensor):
        """Test sinusoidal positional encoding."""
        d_model = input_tensor.size(-1)
        pe = SinusoidalPositionalEncoding(d_model)
        
        output = pe(input_tensor)
        assert output.shape == input_tensor.shape
        assert pe.get_encoding_type() == "sinusoidal"
    
    def test_rope_pe(self, input_tensor):
        """Test RoPE positional encoding."""
        d_model = input_tensor.size(-1)
        pe = RotaryPositionalEncoding(d_model)
        
        output = pe(input_tensor)
        assert output.shape == input_tensor.shape
        assert pe.get_encoding_type() == "rope"
        assert pe.can_extrapolate() == True
    
    def test_alibi_pe(self, input_tensor):
        """Test ALiBi positional encoding."""
        d_model = input_tensor.size(-1)
        n_heads = 8
        pe = ALiBiPositionalEncoding(d_model, n_heads)
        
        output = pe(input_tensor)
        assert output.shape == input_tensor.shape
        assert pe.get_encoding_type() == "alibi"
        assert pe.can_extrapolate() == True
        
        # Test bias generation
        seq_len = input_tensor.size(1)
        bias = pe.get_alibi_bias(seq_len, input_tensor.device)
        expected_bias_shape = (n_heads, seq_len, seq_len)
        assert bias.shape == expected_bias_shape
    
    def test_nope(self, input_tensor):
        """Test NoPE (no positional encoding)."""
        d_model = input_tensor.size(-1)
        pe = NoPositionalEncoding(d_model)
        
        output = pe(input_tensor)
        # Output should be the same as input (except for dropout)
        pe.eval()  # Disable dropout for testing
        with torch.no_grad():
            output_no_dropout = pe(input_tensor)
        
        # The output should be very close to input when dropout is disabled
        assert output_no_dropout.shape == input_tensor.shape
        assert pe.get_encoding_type() == "nope"
        assert pe.can_extrapolate() == True
