"""
Compatibility wrapper for mathematical_reasoning_model.py

This module provides backward compatibility by re-exporting 
the mathematical reasoning model components.
"""

from .mathematical_reasoning_model import (
    MathematicalReasoningModel,
    create_mathematical_reasoning_model,
    MathematicalTokenizer
)

# Re-export for compatibility
__all__ = [
    'MathematicalReasoningModel',
    'create_mathematical_reasoning_model', 
    'MathematicalTokenizer'
]