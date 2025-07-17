import torch
import torch.nn as nn

class AlibiPlusPE(nn.Module):
    """ALiBi with learnable slope offset ("ALiBi+", simple extension)."""
    def __init__(self, num_heads: int, max_seq_len: int = 8192):
        super().__init__()
        slopes = torch.tensor([1.0 / (2 ** (i / num_heads)) for i in range(num_heads)])
        self.register_parameter("slopes", nn.Parameter(slopes))
        self.max_seq_len = max_seq_len

    def forward(self, attn_scores: torch.Tensor, seq_len: int):
        """Add ALiBi bias in-place to attention scores.
        attn_scores shape: (batch, heads, q_len, k_len)
        """
        device = attn_scores.device
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        bias = position_ids.unsqueeze(0) - position_ids.unsqueeze(1)  # (q_len, k_len)
        bias = bias.abs().unsqueeze(0).unsqueeze(0)  # (1,1,q_len,k_len)
        slopes = self.slopes.unsqueeze(-1).unsqueeze(-1)  # (heads,1,1)
        attn_scores -= slopes * bias  # broadcasting
        return attn_scores