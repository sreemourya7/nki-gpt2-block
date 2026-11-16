"""PyTorch reference implementations.

Everything here is written in the plain token-major layout a PyTorch user
expects (``[T, D]``). The kernels are checked against these functions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def linear(x, W, b):
    """GPT-2 Conv1D: ``x @ W + b`` with ``W`` of shape ``[in, out]``."""
    return x @ W + b


def gelu_tanh(x):
    return F.gelu(x, approximate="tanh")


def mlp(x, W1, b1, W2, b2):
    return linear(gelu_tanh(linear(x, W1, b1)), W2, b2)


def layernorm(x, gamma, beta, eps=1e-5):
    return F.layer_norm(x, (x.shape[-1],), gamma, beta, eps)


def causal_attention(qkv, n_heads):
    """Causal multi-head self-attention on concatenated ``[T, 3*D]`` projections."""
    T, D3 = qkv.shape
    D = D3 // 3
    q, k, v = (t.view(T, n_heads, D // n_heads).transpose(0, 1) for t in qkv.split(D, dim=1))
    scores = q @ k.transpose(-1, -2) / (D // n_heads) ** 0.5
    mask = torch.ones(T, T, dtype=torch.bool, device=qkv.device).tril()
    probs = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
    return (probs @ v).transpose(0, 1).reshape(T, D)


def gpt2_block_intermediates(x, w, n_heads, eps=1e-5):
    """One pre-norm GPT-2 block, returning every stage (same names as ``block.run_block``).

    ``w`` maps the key names of ``block.WEIGHT_KEYS`` to torch tensors in
    PyTorch shapes (vectors are 1-D).
    """
    s = {}
    s["ln_1"] = layernorm(x, w["ln_1.g"], w["ln_1.b"], eps)
    s["qkv"] = linear(s["ln_1"], w["attn.c_attn.w"], w["attn.c_attn.b"])
    s["attention"] = causal_attention(s["qkv"], n_heads)
    s["attn_proj"] = linear(s["attention"], w["attn.c_proj.w"], w["attn.c_proj.b"])
    s["resid_mid"] = x + s["attn_proj"]
    s["ln_2"] = layernorm(s["resid_mid"], w["ln_2.g"], w["ln_2.b"], eps)
    s["mlp"] = mlp(
        s["ln_2"], w["mlp.c_fc.w"], w["mlp.c_fc.b"], w["mlp.c_proj.w"], w["mlp.c_proj.b"]
    )
    s["resid_out"] = s["resid_mid"] + s["mlp"]
    return s


def gpt2_block(x, w, n_heads, eps=1e-5):
    """One pre-norm GPT-2 block."""
    return gpt2_block_intermediates(x, w, n_heads, eps)["resid_out"]
