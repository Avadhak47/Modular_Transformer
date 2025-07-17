from typing import Literal

import torch
from transformers import AutoModelForCausalLM

from .pe import XPOS, SinusoidalPE, AlibiPlusPE

PE_TYPE = Literal["xpos", "sinusoidal", "alibi+"]


def load_deepseek_with_pe(model_name: str, pe: PE_TYPE = "xpos", rope_scaling: Literal["linear", "dynamic"] = "linear"):
    """Load DeepSeekMath model and inject custom PE implementation.
    For RoPE-based architectures (Llama), we patch the rotary embedding tables.
    For ALiBi, we monkey-patch the attention forward method to add biases.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    if pe == "xpos":
        _patch_rope_with_xpos(model)
    elif pe == "sinusoidal":
        _patch_rope_with_sinusoidal(model)
    elif pe == "alibi+":
        _patch_with_alibi_plus(model)
    else:
        raise ValueError(f"Unknown pe {pe}")

    return model


def _patch_rope_with_xpos(model):
    for name, module in model.named_modules():
        if hasattr(module, "rotary_emb"):
            dim = module.rotary_emb.inv_freq.numel() * 2
            module.rotary_emb = XPOS(dim)

            def _apply_rotary(q, k, self_module=module):
                seq_len = q.size(-2)
                return self_module.rotary_emb.apply_rotary(q, k, seq_len)

            module.apply_rotary = _apply_rotary


def _patch_rope_with_sinusoidal(model):
    # For demonstration: embed_tokens weight add sinusoidal bias during forward
    class _Wrapped(torch.nn.Module):
        def __init__(self, orig, d_model):
            super().__init__()
            self.orig = orig
            self.pe = SinusoidalPE(d_model)

        def forward(self, input_ids=None, *args, **kwargs):
            emb = self.orig(input_ids, *args, **kwargs)
            return self.pe(emb)

    for name, module in model.named_modules():
        if name.endswith("embed_tokens"):
            d_model = module.embedding_dim
            parent = _get_parent(model, name)
            setattr(parent, name.split(".")[-1], _Wrapped(module, d_model))


def _patch_with_alibi_plus(model):
    for name, module in model.named_modules():
        if name.endswith("attention") and hasattr(module, "num_heads"):
            num_heads = module.num_heads
            pe_bias = AlibiPlusPE(num_heads)
            old_forward = module.forward

            def forward_with_alibi(*inputs, **kwargs):
                output = old_forward(*inputs, **kwargs)
                attn_scores = output[2] if isinstance(output, tuple) else None
                if attn_scores is not None:
                    seq_len = attn_scores.size(-1)
                    pe_bias(attn_scores, seq_len)
                return output

            module.forward = forward_with_alibi


def _get_parent(model, module_name):
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent