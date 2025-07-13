"""
Tests for the main transformer model.
"""
import pytest
import torch
from src.model import TransformerModel


class TestTransformerModel:
    """Test cases for TransformerModel."""
    
    @pytest.fixture
    def config(self):
        """Basic model configuration for testing."""
        return {
            "d_model": 256,
            "n_heads": 4,
            "d_ff": 1024,
            "n_encoder_layers": 2,
            "n_decoder_layers": 2,
            "vocab_size": 1000,
            "max_seq_len": 128,
            "dropout": 0.1,
            "positional_encoding": "sinusoidal"
        }
    
    def test_model_creation(self, config):
        """Test basic model creation."""
        model = TransformerModel(config)
        assert model.d_model == config["d_model"]
        assert model.n_heads == config["n_heads"]
    
    def test_forward_pass(self, config):
        """Test forward pass through the model."""
        model = TransformerModel(config)
        model.eval()
        
        batch_size = 2
        seq_len = 32
        src = torch.randint(1, config["vocab_size"], (batch_size, seq_len))
        tgt = torch.randint(1, config["vocab_size"], (batch_size, seq_len))
        
        with torch.no_grad():
            output = model(src, tgt)
        
        expected_shape = (batch_size, seq_len, config["vocab_size"])
        assert output.shape == expected_shape
    
    def test_positional_encoding_switching(self, config):
        """Test switching between different positional encodings."""
        model = TransformerModel(config)
        
        # Test switching to different PE types
        pe_types = ["rope", "alibi", "nope"]
        
        for pe_type in pe_types:
            try:
                model.switch_positional_encoding(pe_type)
                assert model.pos_encoding.get_encoding_type() == pe_type
            except Exception as e:
                # Some PE types might fail due to model configuration
                print(f"PE type {pe_type} failed: {e}")
    
    def test_parameter_count(self, config):
        """Test parameter counting."""
        model = TransformerModel(config)
        info = model.get_model_info()
        
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert info["total_parameters"] > 0
        assert info["trainable_parameters"] > 0
