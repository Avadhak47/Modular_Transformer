"""
Mathematical Adaptive Positional Encoding (MAPE)
A novel positional encoding method specifically designed for mathematical reasoning tasks.

Key innovations:
- Expression-aware hierarchical encoding
- Dynamic frequency adaptation based on mathematical content
- Operator precedence modeling
- Multi-scale position representation
- Mathematical symbol embedding integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple, List
import re


class MathAdaptivePositionalEncoding(nn.Module):
    """
    Mathematical Adaptive Positional Encoding (MAPE)
    
    A novel approach that adapts positional encoding based on mathematical content,
    incorporating operator precedence, expression hierarchy, and symbol semantics.
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 8192,
        num_hierarchy_levels: int = 4,
        symbol_vocab_size: int = 1000,
        operator_weight: float = 1.5,
        number_weight: float = 1.2,
        variable_weight: float = 1.3,
        function_weight: float = 1.4,
        bracket_weight: float = 2.0,
        learnable_frequencies: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_hierarchy_levels = num_hierarchy_levels
        self.symbol_vocab_size = symbol_vocab_size
        
        # Mathematical element weights
        self.operator_weight = operator_weight
        self.number_weight = number_weight
        self.variable_weight = variable_weight
        self.function_weight = function_weight
        self.bracket_weight = bracket_weight
        
        # Base frequencies for different mathematical elements
        self.base_frequencies = self._initialize_base_frequencies()
        
        # Learnable frequency adjustments
        if learnable_frequencies:
            self.frequency_adjustments = nn.Parameter(torch.ones(d_model // 2))
            self.hierarchy_frequency_scales = nn.Parameter(torch.ones(num_hierarchy_levels))
        else:
            self.register_buffer("frequency_adjustments", torch.ones(d_model // 2))
            self.register_buffer("hierarchy_frequency_scales", torch.ones(num_hierarchy_levels))
        
        # Mathematical symbol embeddings
        self.symbol_embeddings = nn.Embedding(symbol_vocab_size, d_model)
        
        # Operator precedence encoding
        self.precedence_encoder = OperatorPrecedenceEncoder(d_model)
        
        # Expression hierarchy encoder
        self.hierarchy_encoder = ExpressionHierarchyEncoder(d_model, num_hierarchy_levels)
        
        # Multi-scale position encoder
        self.multiscale_encoder = MultiScalePositionEncoder(d_model, max_seq_len)
        
        # Mathematical content classifier
        self.content_classifier = MathContentClassifier(d_model)
        
        # Adaptive combination weights
        self.combination_weights = nn.Parameter(torch.ones(5))  # 5 components
        
        # Cache for efficiency
        self.register_buffer("position_cache", torch.empty(0))
        self.register_buffer("hierarchy_cache", torch.empty(0))
        self.cached_seq_len = 0
    
    def _initialize_base_frequencies(self) -> torch.Tensor:
        """Initialize base frequencies for mathematical reasoning."""
        dim_pairs = self.d_model // 2
        
        # Create frequency spectrum optimized for mathematical patterns
        freq_exponents = torch.arange(0, dim_pairs, dtype=torch.float32) / dim_pairs
        
        # Mathematical frequency distribution (emphasizes different scales)
        base_freq = 1.0 / (10000.0 ** freq_exponents)
        
        # Add mathematical-specific frequency components
        math_freqs = torch.stack([
            base_freq,  # Standard frequencies
            base_freq * 2,  # Higher frequencies for operators
            base_freq * 0.5,  # Lower frequencies for numbers
            base_freq * 1.5,  # Medium frequencies for variables
        ], dim=0).mean(dim=0)
        
        return math_freq
    
    def _classify_mathematical_content(self, token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Classify tokens into mathematical categories."""
        batch_size, seq_len = token_ids.shape
        
        # Initialize classification masks
        classifications = {
            'operators': torch.zeros_like(token_ids, dtype=torch.bool),
            'numbers': torch.zeros_like(token_ids, dtype=torch.bool),
            'variables': torch.zeros_like(token_ids, dtype=torch.bool),
            'functions': torch.zeros_like(token_ids, dtype=torch.bool),
            'brackets': torch.zeros_like(token_ids, dtype=torch.bool),
            'text': torch.zeros_like(token_ids, dtype=torch.bool)
        }
        
        # Simplified classification based on token ranges
        # In practice, this would use tokenizer vocabulary mapping
        
        # Operators (assume certain token ranges)
        operator_mask = ((token_ids >= 40) & (token_ids <= 47)) | \
                       ((token_ids >= 60) & (token_ids <= 62)) | \
                       (token_ids == 94)  # +, -, *, /, <, =, >, ^
        classifications['operators'] = operator_mask
        
        # Numbers (digits and decimal points)
        number_mask = ((token_ids >= 48) & (token_ids <= 57)) | (token_ids == 46)  # 0-9, .
        classifications['numbers'] = number_mask
        
        # Variables (letters)
        variable_mask = ((token_ids >= 65) & (token_ids <= 90)) | \
                       ((token_ids >= 97) & (token_ids <= 122))  # A-Z, a-z
        classifications['variables'] = variable_mask
        
        # Brackets
        bracket_mask = (token_ids == 40) | (token_ids == 41) | \
                      (token_ids == 91) | (token_ids == 93) | \
                      (token_ids == 123) | (token_ids == 125)  # (), [], {}
        classifications['brackets'] = bracket_mask
        
        # Functions (heuristic: letters followed by opening bracket)
        function_mask = torch.zeros_like(token_ids, dtype=torch.bool)
        for i in range(seq_len - 1):
            if variable_mask[:, i].any() and (token_ids[:, i+1] == 40).any():
                function_mask[:, i] = True
        classifications['functions'] = function_mask
        
        # Everything else is text
        classifications['text'] = ~(operator_mask | number_mask | variable_mask | 
                                   bracket_mask | function_mask)
        
        return classifications
    
    def _compute_adaptive_frequencies(
        self, 
        classifications: Dict[str, torch.Tensor],
        seq_len: int
    ) -> torch.Tensor:
        """Compute adaptive frequencies based on mathematical content."""
        batch_size = list(classifications.values())[0].shape[0]
        
        # Start with base frequencies
        adaptive_freqs = self.base_frequencies.unsqueeze(0).expand(batch_size, seq_len, -1)
        
        # Apply content-specific frequency adjustments
        for content_type, mask in classifications.items():
            if mask.any():
                weight = getattr(self, f'{content_type}_weight', 1.0)
                mask_expanded = mask.unsqueeze(-1).expand(-1, -1, self.d_model // 2).float()
                adaptive_freqs = adaptive_freqs * (1.0 + (weight - 1.0) * mask_expanded)
        
        # Apply learnable adjustments
        adaptive_freqs = adaptive_freqs * self.frequency_adjustments.unsqueeze(0).unsqueeze(0)
        
        return adaptive_freqs
    
    def _compute_hierarchical_positions(
        self, 
        token_ids: torch.Tensor,
        classifications: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute hierarchical positions based on expression structure."""
        batch_size, seq_len = token_ids.shape
        
        # Initialize hierarchical positions
        hier_positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.float32)
        hier_positions = hier_positions.unsqueeze(0).expand(batch_size, -1)
        
        # Analyze bracket nesting for hierarchy
        bracket_levels = self._compute_bracket_levels(token_ids, classifications['brackets'])
        
        # Apply hierarchical adjustments
        for level in range(self.num_hierarchy_levels):
            level_mask = (bracket_levels == level)
            scale = self.hierarchy_frequency_scales[level]
            hier_positions = hier_positions + level_mask.float() * scale * 0.1
        
        return hier_positions
    
    def _compute_bracket_levels(
        self, 
        token_ids: torch.Tensor, 
        bracket_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute nesting levels based on bracket structure."""
        batch_size, seq_len = token_ids.shape
        levels = torch.zeros_like(token_ids, dtype=torch.long)
        
        # Simplified bracket level computation
        open_brackets = [40, 91, 123]  # (, [, {
        close_brackets = [41, 93, 125]  # ), ], }
        
        for b in range(batch_size):
            current_level = 0
            for i in range(seq_len):
                if token_ids[b, i].item() in open_brackets:
                    current_level += 1
                elif token_ids[b, i].item() in close_brackets:
                    current_level = max(0, current_level - 1)
                levels[b, i] = current_level
        
        return levels
    
    def forward(
        self, 
        x: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply Mathematical Adaptive Positional Encoding.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            token_ids: Token IDs for content analysis
            position_ids: Explicit position IDs (optional)
        
        Returns:
            Tensor with MAPE applied
        """
        batch_size, seq_len, d_model = x.shape
        
        if token_ids is None:
            # Fallback to standard positional encoding
            return self._apply_standard_encoding(x, position_ids)
        
        # Classify mathematical content
        classifications = self._classify_mathematical_content(token_ids)
        
        # Compute adaptive frequencies
        adaptive_freqs = self._compute_adaptive_frequencies(classifications, seq_len)
        
        # Compute hierarchical positions
        hier_positions = self._compute_hierarchical_positions(token_ids, classifications)
        
        # Generate position embeddings for each component
        components = []
        
        # 1. Base positional encoding with adaptive frequencies
        base_pos = self._generate_base_positions(hier_positions, adaptive_freqs)
        components.append(base_pos)
        
        # 2. Operator precedence encoding
        precedence_pos = self.precedence_encoder(token_ids, classifications)
        components.append(precedence_pos)
        
        # 3. Expression hierarchy encoding
        hierarchy_pos = self.hierarchy_encoder(token_ids, classifications)
        components.append(hierarchy_pos)
        
        # 4. Multi-scale position encoding
        multiscale_pos = self.multiscale_encoder(hier_positions)
        components.append(multiscale_pos)
        
        # 5. Symbol-specific embeddings
        symbol_pos = self._generate_symbol_positions(token_ids)
        components.append(symbol_pos)
        
        # Combine all components with learnable weights
        weights = F.softmax(self.combination_weights, dim=0)
        combined_pos = sum(w * comp for w, comp in zip(weights, components))
        
        # Add to input embeddings
        return x + combined_pos
    
    def _apply_standard_encoding(
        self, 
        x: torch.Tensor, 
        position_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Fallback to standard sinusoidal encoding."""
        batch_size, seq_len, d_model = x.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Standard sinusoidal encoding
        pe = torch.zeros(batch_size, seq_len, d_model, device=x.device)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=x.device).float() * 
            -(math.log(10000.0) / d_model)
        )
        
        pe[:, :, 0::2] = torch.sin(position_ids.unsqueeze(-1).float() * div_term)
        pe[:, :, 1::2] = torch.cos(position_ids.unsqueeze(-1).float() * div_term)
        
        return x + pe
    
    def _generate_base_positions(
        self, 
        positions: torch.Tensor, 
        freqs: torch.Tensor
    ) -> torch.Tensor:
        """Generate base positional encodings with adaptive frequencies."""
        batch_size, seq_len = positions.shape
        
        # Expand positions for frequency computation
        pos_expanded = positions.unsqueeze(-1)  # (batch, seq_len, 1)
        
        # Compute position * frequency
        pos_freqs = pos_expanded * freqs  # (batch, seq_len, d_model//2)
        
        # Generate sin/cos embeddings
        pe = torch.zeros(batch_size, seq_len, self.d_model, device=positions.device)
        pe[:, :, 0::2] = torch.sin(pos_freqs)
        pe[:, :, 1::2] = torch.cos(pos_freqs)
        
        return pe
    
    def _generate_symbol_positions(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Generate symbol-specific positional embeddings."""
        # Clamp token IDs to vocabulary size
        clamped_ids = torch.clamp(token_ids, 0, self.symbol_vocab_size - 1)
        
        # Get symbol embeddings
        symbol_embeds = self.symbol_embeddings(clamped_ids)
        
        return symbol_embeds * 0.1  # Scale down symbol contribution


class OperatorPrecedenceEncoder(nn.Module):
    """Encodes operator precedence information into positional embeddings."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Operator precedence levels (simplified)
        self.precedence_levels = {
            # High precedence
            42: 5,  # *
            47: 5,  # /
            94: 6,  # ^
            # Medium precedence  
            43: 3,  # +
            45: 3,  # -
            # Low precedence
            60: 2,  # <
            62: 2,  # >
            61: 1,  # =
        }
        
        self.precedence_embeddings = nn.Embedding(7, d_model)  # 6 levels + 0 for non-operators
    
    def forward(
        self, 
        token_ids: torch.Tensor, 
        classifications: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Encode operator precedence into positional embeddings."""
        batch_size, seq_len = token_ids.shape
        
        # Initialize precedence levels
        precedence_ids = torch.zeros_like(token_ids)
        
        # Assign precedence levels
        for token_id, level in self.precedence_levels.items():
            mask = (token_ids == token_id)
            precedence_ids[mask] = level
        
        # Get precedence embeddings
        precedence_pos = self.precedence_embeddings(precedence_ids)
        
        # Apply only to operators
        operator_mask = classifications['operators'].unsqueeze(-1).float()
        precedence_pos = precedence_pos * operator_mask
        
        return precedence_pos * 0.2  # Scale down precedence contribution


class ExpressionHierarchyEncoder(nn.Module):
    """Encodes mathematical expression hierarchy into positional embeddings."""
    
    def __init__(self, d_model: int, num_levels: int):
        super().__init__()
        self.d_model = d_model
        self.num_levels = num_levels
        
        self.level_embeddings = nn.Embedding(num_levels, d_model)
    
    def forward(
        self, 
        token_ids: torch.Tensor, 
        classifications: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Encode expression hierarchy into positional embeddings."""
        batch_size, seq_len = token_ids.shape
        
        # Compute expression levels based on brackets
        expression_levels = self._compute_expression_levels(token_ids, classifications)
        
        # Get hierarchy embeddings
        hierarchy_pos = self.level_embeddings(expression_levels)
        
        return hierarchy_pos * 0.15  # Scale down hierarchy contribution
    
    def _compute_expression_levels(
        self, 
        token_ids: torch.Tensor, 
        classifications: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute expression nesting levels."""
        batch_size, seq_len = token_ids.shape
        levels = torch.zeros_like(token_ids)
        
        for b in range(batch_size):
            current_level = 0
            for i in range(seq_len):
                if token_ids[b, i] in [40, 91, 123]:  # Opening brackets
                    current_level = min(current_level + 1, self.num_levels - 1)
                elif token_ids[b, i] in [41, 93, 125]:  # Closing brackets
                    current_level = max(0, current_level - 1)
                levels[b, i] = current_level
        
        return levels


class MultiScalePositionEncoder(nn.Module):
    """Encodes positions at multiple scales for mathematical reasoning."""
    
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Multiple scale encoders
        self.scales = [1, 2, 4, 8, 16]
        self.scale_weights = nn.Parameter(torch.ones(len(self.scales)))
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Generate multi-scale positional encodings."""
        batch_size, seq_len = positions.shape
        
        # Generate encodings at different scales
        scale_encodings = []
        
        for scale in self.scales:
            scaled_pos = positions / scale
            pe = self._sinusoidal_encoding(scaled_pos)
            scale_encodings.append(pe)
        
        # Combine scales with learnable weights
        weights = F.softmax(self.scale_weights, dim=0)
        combined = sum(w * enc for w, enc in zip(weights, scale_encodings))
        
        return combined * 0.1  # Scale down multi-scale contribution
    
    def _sinusoidal_encoding(self, positions: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal encoding for given positions."""
        batch_size, seq_len = positions.shape
        
        pe = torch.zeros(batch_size, seq_len, self.d_model, device=positions.device)
        
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=positions.device).float() * 
            -(math.log(10000.0) / self.d_model)
        )
        
        pos_expanded = positions.unsqueeze(-1).float()
        pe[:, :, 0::2] = torch.sin(pos_expanded * div_term)
        pe[:, :, 1::2] = torch.cos(pos_expanded * div_term)
        
        return pe


class MathContentClassifier(nn.Module):
    """Classifies mathematical content for adaptive encoding."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.classifier = nn.Linear(d_model, 6)  # 6 content types
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Classify mathematical content types."""
        return F.softmax(self.classifier(embeddings), dim=-1)


if __name__ == "__main__":
    # Test the implementation
    d_model = 512
    seq_len = 128
    batch_size = 2
    
    # Create MAPE encoder
    mape = MathAdaptivePositionalEncoding(d_model=d_model)
    
    # Test input
    x = torch.randn(batch_size, seq_len, d_model)
    token_ids = torch.randint(32, 126, (batch_size, seq_len))  # ASCII printable range
    
    # Apply MAPE
    x_encoded = mape(x, token_ids=token_ids)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_encoded.shape}")
    print(f"MAPE applied successfully!")
    
    # Test individual components
    print(f"Number of parameters: {sum(p.numel() for p in mape.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in mape.parameters() if p.requires_grad)}")