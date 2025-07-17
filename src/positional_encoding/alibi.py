"""
ALiBi (Attention with Linear Biases) - Enhanced for Mathematical Reasoning

Optimized implementation with:
- Adaptive slope learning for mathematical patterns
- Long sequence extrapolation capabilities
- Mathematical reasoning specific bias patterns
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class ALiBiPositionalBias(nn.Module):
    """
    Enhanced ALiBi Positional Bias for Mathematical Reasoning
    
    Features:
    - Learnable slopes optimized for mathematical patterns
    - Dynamic bias adjustment based on sequence content
    - Efficient computation for long sequences
    - Mathematical reasoning specific enhancements
    """
    
    def __init__(
        self,
        num_heads: int,
        max_seq_len: int = 32768,
        learnable_slopes: bool = True,
        math_enhanced: bool = True,
        use_symmetric_bias: bool = False
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.math_enhanced = math_enhanced
        self.use_symmetric_bias = use_symmetric_bias
        
        # Compute base slopes
        self.register_buffer('base_slopes', self._compute_base_slopes())
        
        # Learnable slope adjustments for mathematical reasoning
        if learnable_slopes:
            self.slope_adjustment = nn.Parameter(torch.zeros(num_heads))
            self.slope_scaling = nn.Parameter(torch.ones(num_heads))
        else:
            self.register_buffer('slope_adjustment', torch.zeros(num_heads))
            self.register_buffer('slope_scaling', torch.ones(num_heads))
        
        # Mathematical pattern specific biases
        if math_enhanced:
            self.math_bias_scale = nn.Parameter(torch.ones(num_heads))
            self.sequential_bias = nn.Parameter(torch.zeros(num_heads))
            self.hierarchical_bias = nn.Parameter(torch.zeros(num_heads))
        
        # Cache for bias matrix
        self._cached_bias = None
        self._cached_seq_len = 0
    
    def _compute_base_slopes(self) -> torch.Tensor:
        """Compute base slopes for ALiBi attention bias."""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        def get_slopes(n):
            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2**math.floor(math.log2(n))
                return (get_slopes_power_of_2(closest_power_of_2) + 
                       get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2])
        
        slopes = get_slopes(self.num_heads)
        return torch.tensor(slopes, dtype=torch.float32)
    
    def _get_bias_matrix(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or compute the bias matrix for given sequence length."""
        if self._cached_bias is not None and seq_len <= self._cached_seq_len:
            return self._cached_bias[:, :seq_len, :seq_len]
        
        # Create position difference matrix
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        position_diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [1, seq_len, seq_len]
        
        # Apply slopes
        effective_slopes = self.base_slopes * self.slope_scaling + self.slope_adjustment
        effective_slopes = effective_slopes.to(device).unsqueeze(1).unsqueeze(2)
        
        # Compute base bias
        bias = position_diff.unsqueeze(0) * effective_slopes  # [num_heads, seq_len, seq_len]
        
        # Mathematical reasoning enhancements
        if self.math_enhanced:
            # Scale bias for mathematical patterns
            bias = bias * self.math_bias_scale.unsqueeze(1).unsqueeze(2)
            
            # Add sequential reasoning bias (favor recent context)
            sequential_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            sequential_bias = (sequential_mask * 
                             self.sequential_bias.unsqueeze(1).unsqueeze(2))
            bias = bias + sequential_bias
            
            # Add hierarchical reasoning bias (favor certain distance patterns)
            hierarchical_pattern = self._create_hierarchical_pattern(seq_len, device)
            hierarchical_bias = (hierarchical_pattern * 
                                self.hierarchical_bias.unsqueeze(1).unsqueeze(2))
            bias = bias + hierarchical_bias
        
        # Apply causal mask (upper triangular part set to -inf)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        bias = bias.masked_fill(causal_mask.bool(), float('-inf'))
        
        # Cache the result
        self._cached_bias = bias
        self._cached_seq_len = seq_len
        
        return bias
    
    def _create_hierarchical_pattern(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create hierarchical attention pattern for mathematical reasoning."""
        # Create patterns that favor certain mathematical structure distances
        pattern = torch.zeros(seq_len, seq_len, device=device)
        
        for i in range(seq_len):
            for j in range(seq_len):
                distance = abs(i - j)
                
                # Favor mathematical structure distances (powers of 2, fibonacci-like)
                if distance in [1, 2, 3, 5, 8, 13, 21]:  # Fibonacci-like
                    pattern[i, j] = 0.1
                elif distance in [4, 16, 64, 256]:  # Powers of 4
                    pattern[i, j] = 0.05
                elif distance % 10 == 0:  # Decimal structure
                    pattern[i, j] = 0.02
        
        return pattern
    
    def forward(
        self, 
        attention_scores: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply ALiBi bias to attention scores.
        
        Args:
            attention_scores: Attention scores [batch, num_heads, seq_len, seq_len]
            seq_len: Sequence length (if different from tensor size)
            
        Returns:
            Biased attention scores
        """
        if seq_len is None:
            seq_len = attention_scores.size(-1)
        
        # Get bias matrix
        bias = self._get_bias_matrix(seq_len, attention_scores.device)
        
        # Add bias to attention scores
        return attention_scores + bias
    
    def extend_length(self, new_max_len: int):
        """Extend maximum sequence length and clear cache."""
        if new_max_len <= self.max_seq_len:
            return
        
        self.max_seq_len = new_max_len
        self._cached_bias = None
        self._cached_seq_len = 0
    
    def get_slopes(self) -> torch.Tensor:
        """Get effective slopes for analysis."""
        return self.base_slopes * self.slope_scaling + self.slope_adjustment


class AdaptiveALiBi(ALiBiPositionalBias):
    """
    Adaptive ALiBi that learns optimal bias patterns for mathematical reasoning.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, math_enhanced=True, **kwargs)
        
        # Additional adaptive parameters
        self.distance_embedding = nn.Embedding(self.max_seq_len, self.num_heads)
        self.content_adaptive_bias = nn.Linear(self.num_heads, self.num_heads)
        
        # Mathematical pattern detectors
        self.arithmetic_pattern_weight = nn.Parameter(torch.zeros(self.num_heads))
        self.algebraic_pattern_weight = nn.Parameter(torch.zeros(self.num_heads))
        self.geometric_pattern_weight = nn.Parameter(torch.zeros(self.num_heads))
    
    def _compute_adaptive_bias(
        self, 
        seq_len: int, 
        device: torch.device,
        content_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute adaptive bias based on content and mathematical patterns."""
        # Base bias computation
        base_bias = super()._get_bias_matrix(seq_len, device)
        
        # Distance-based adaptive bias
        distances = torch.arange(seq_len, device=device)
        distance_bias = self.distance_embedding(distances)  # [seq_len, num_heads]
        
        # Create distance bias matrix
        dist_bias_matrix = torch.zeros(self.num_heads, seq_len, seq_len, device=device)
        for i in range(seq_len):
            for j in range(seq_len):
                dist = abs(i - j)
                if dist < seq_len:
                    dist_bias_matrix[:, i, j] = distance_bias[dist]
        
        # Content-adaptive bias (if content features provided)
        if content_features is not None:
            content_bias = self.content_adaptive_bias(content_features.mean(dim=1))
            content_bias = content_bias.unsqueeze(1).unsqueeze(2)
            base_bias = base_bias + content_bias
        
        # Mathematical pattern-specific adjustments
        pattern_weights = (self.arithmetic_pattern_weight + 
                          self.algebraic_pattern_weight + 
                          self.geometric_pattern_weight) / 3.0
        pattern_bias = dist_bias_matrix * pattern_weights.unsqueeze(1).unsqueeze(2)
        
        return base_bias + pattern_bias
    
    def forward(
        self, 
        attention_scores: torch.Tensor,
        seq_len: Optional[int] = None,
        content_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Enhanced forward with adaptive bias computation."""
        if seq_len is None:
            seq_len = attention_scores.size(-1)
        
        # Get adaptive bias
        bias = self._compute_adaptive_bias(seq_len, attention_scores.device, content_features)
        
        return attention_scores + bias


class MathematicalALiBi(ALiBiPositionalBias):
    """
    ALiBi specifically optimized for mathematical reasoning patterns.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, math_enhanced=True, **kwargs)
        
        # Mathematical reasoning specific parameters
        self.equation_locality_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.proof_step_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.variable_reference_bias = nn.Parameter(torch.zeros(self.num_heads))
        
        # Learned mathematical distance functions
        self.math_distance_transform = nn.Linear(1, self.num_heads)
    
    def _create_mathematical_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create bias patterns specific to mathematical reasoning."""
        bias = torch.zeros(self.num_heads, seq_len, seq_len, device=device)
        
        for i in range(seq_len):
            for j in range(seq_len):
                distance = abs(i - j)
                
                # Equation locality: favor nearby tokens in equations
                if distance <= 5:
                    bias[:, i, j] += self.equation_locality_bias
                
                # Proof steps: favor certain step distances
                if distance in [1, 2, 3]:  # Adjacent proof steps
                    bias[:, i, j] += self.proof_step_bias
                
                # Variable references: favor medium-range dependencies
                if 3 < distance <= 20:
                    bias[:, i, j] += self.variable_reference_bias * (1.0 / distance)
                
                # Transform distance through learned function
                dist_tensor = torch.tensor([distance], dtype=torch.float32, device=device)
                dist_bias = self.math_distance_transform(dist_tensor)
                bias[:, i, j] += dist_bias
        
        return bias
    
    def _get_bias_matrix(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Enhanced bias matrix with mathematical reasoning patterns."""
        # Get base ALiBi bias
        base_bias = super()._get_bias_matrix(seq_len, device)
        
        # Add mathematical reasoning bias
        math_bias = self._create_mathematical_bias(seq_len, device)
        
        return base_bias + math_bias