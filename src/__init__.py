"""
Modular Transformer Package

A comprehensive implementation of the Transformer architecture with
interchangeable positional encoding methods.
"""

__version__ = "0.1.0"
__author__ = "Avadhesh"
__email__ = "your.email@example.com"

from .model import TransformerModel
from .config import ModelConfig

__all__ = [
    "TransformerModel",
    "ModelConfig",
]