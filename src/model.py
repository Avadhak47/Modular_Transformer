import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Any, Tuple

from .layers.attention import MultiHeadAttention
from .layers.feed_forward import PositionWiseFeedForward
from .layers.encoder import TransformerEncoder
from .layers.decoder import TransformerDecoder
from .layers.embedding import TokenEmbedding
from .positional_encoding.base import BasePositionalEncoding
from .positional_encoding.sinusoidal import SinusoidalPositionalEncoding
from .positional_encoding.sinusoidal import SinusoidalPositionalEncoding
from .positional_encoding.rope import RotaryPositionalEncoding
from .positional_encoding.alibi import ALiBiPositionalEncoding
from .positional_encoding.diet import DIETPositionalEncoding
from .positional_encoding.t5_relative import T5RelativePositionalEncoding
from .positional_encoding.nope import NoPositionalEncoding

class TransformerModel(nn.Module):
    """
    Complete Transformer model with modular positional encoding support.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Model dimensions
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.d_ff = config['d_ff']
        self.n_encoder_layers = config['n_encoder_layers']
        self.n_decoder_layers = config['n_decoder_layers']
        self.vocab_size = config['vocab_size']
        self.max_seq_len = config['max_seq_len']
        self.dropout = config['dropout']
        
        # Embeddings
        self.src_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.tgt_embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # Positional encoding (can be swapped)
        self.pos_encoding = self._create_positional_encoding(config)
        
        # Encoder and Decoder
        self.encoder = TransformerEncoder(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            n_layers=self.n_encoder_layers,
            dropout=self.dropout
        )
        
        self.decoder = TransformerDecoder(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            n_layers=self.n_decoder_layers,
            dropout=self.dropout
        )
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _create_positional_encoding(self, config: Dict[str, Any]) -> BasePositionalEncoding:
        """Create positional encoding based on configuration."""
        pe_type = config.get('positional_encoding', 'sinusoidal')
        
        if pe_type == 'sinusoidal':
            from .positional_encoding.sinusoidal import SinusoidalPositionalEncoding
            return SinusoidalPositionalEncoding(self.d_model, self.max_seq_len, self.dropout)
        elif pe_type == 'rope':
            from .positional_encoding.rope import RotaryPositionalEncoding
            return RotaryPositionalEncoding(self.d_model, self.max_seq_len, self.dropout)
        elif pe_type == 'alibi':
            from .positional_encoding.alibi import ALiBiPositionalEncoding
            return ALiBiPositionalEncoding(self.d_model, self.n_heads, self.max_seq_len, self.dropout)
        elif pe_type == 'diet':
            from .positional_encoding.diet import DIETPositionalEncoding
            return DIETPositionalEncoding(self.d_model, self.n_heads, self.max_seq_len, self.dropout)
        elif pe_type == 't5_relative':
            from .positional_encoding.t5_relative import T5RelativePositionalEncoding
            return T5RelativePositionalEncoding(self.d_model, self.n_heads, self.max_seq_len, self.dropout)
        elif pe_type == 'nope':
            from .positional_encoding.nope import NoPositionalEncoding
            return NoPositionalEncoding(self.d_model, self.max_seq_len, self.dropout)
        else:
            raise ValueError(f"Unknown positional encoding type: {pe_type}")
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.1)
    
    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        """
        Forward pass through the transformer.
        Compatible with HuggingFace-style interface for evaluation.
        """
        # If tgt is None, use src for both (for evaluation compatibility)
        if tgt is None:
            tgt = src
        
        # Embed and apply positional encoding
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        
        # Apply positional encoding
        src_encoded = self.pos_encoding(src_embedded)
        tgt_encoded = self.pos_encoding(tgt_embedded)
        
        # Encoder
        encoder_output = self.encoder(src_encoded, src_mask, self.pos_encoding)
        
        # Decoder
        decoder_output = self.decoder(tgt_encoded, encoder_output, src_mask, tgt_mask, self.pos_encoding)
        
        # Output projection
        logits = self.output_projection(decoder_output)
        
        # Return HuggingFace-style output if labels provided
        if labels is not None:
            # Calculate loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Shift logits and labels for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Return dict with loss and logits (HuggingFace style)
            return type('Output', (), {'loss': loss, 'logits': logits})()
        
        return logits
    
    def switch_positional_encoding(self, new_pe_type: str):
        """Switch to a different positional encoding method."""
        config = self.config.copy()
        config['positional_encoding'] = new_pe_type
        self.pos_encoding = self._create_positional_encoding(config)
        print(f"Switched to {new_pe_type} positional encoding")

    def get_model_info(self) -> dict:
        """Return model parameter statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        }
