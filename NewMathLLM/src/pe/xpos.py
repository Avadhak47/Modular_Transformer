import math
import torch
import torch.nn as nn

class XPOS(nn.Module):
    """Extrapolatable Rotary Positional Embedding (xPos)
    Paper: https://arxiv.org/abs/2307.05440
    Adapted to HuggingFace style. Supports dynamic sequence lengths > 4K.
    """
    def __init__(self, dim: int, base: float = 10000.0, scale_base: float = 512):
        super().__init__()
        self.dim = dim
        self.base = base
        self.scale_base = scale_base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)

        # xPos scaling to extend context length
        power = (torch.arange(seq_len, device=device) - (seq_len // 2)) / self.scale_base
        scale = torch.pow(self.base, power).unsqueeze(-1)
        emb = emb * scale
        return emb

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary(self, q, k, seq_len):
        """Apply xPos rotation to query / key matrices.
        Assumes shape (batch, heads, seq_len, head_dim)
        """
        emb = self.forward(seq_len, q.device)
        cos, sin = emb.cos(), emb.sin()
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,seq_len,dim)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        return q_rot, k_rot