"""One GPT-2 decoder block assembled from the NKI kernels.

Data flow and layouts (TM = token-major ``[T, D]``, FM = feature-major ``[D, T]``)::

    x (TM) ── layernorm_T ──> FM ── linear_T (QKV) ──> FM ── causal_attention_T ──> FM
       │                                                                             │
       │                                              linear_T (output projection) <─┘
       └────────────── residual_add_T (FM -> TM) <──── FM
    x (TM) ── layernorm_T ──> FM ── mlp_T (fc + GELU + proj, hidden stays in SBUF) ──> FM
       └────────────── residual_add_T (FM -> TM) <──── FM

Each arrow is one kernel call, and tensors pass between kernels through HBM.
"""

from __future__ import annotations

import numpy as np

from . import sim
from .kernels import causal_attention_T, layernorm_T, linear_T, mlp_T, residual_add_T

WEIGHT_KEYS = (
    "ln_1.g", "ln_1.b",
    "attn.c_attn.w", "attn.c_attn.b",
    "attn.c_proj.w", "attn.c_proj.b",
    "ln_2.g", "ln_2.b",
    "mlp.c_fc.w", "mlp.c_fc.b",
    "mlp.c_proj.w", "mlp.c_proj.b",
)  # fmt: skip


def random_block_weights(d_model: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Random float32 weights with GPT-2 block shapes (for tests without a download)."""
    rng = np.random.default_rng(seed)

    def mat(i, o):
        return (rng.standard_normal((i, o)) * 0.05).astype(np.float32)

    def vec(n, scale=0.1, shift=0.0):
        return (rng.standard_normal(n) * scale + shift).astype(np.float32)

    return {
        "ln_1.g": vec(d_model, shift=1.0), "ln_1.b": vec(d_model),
        "attn.c_attn.w": mat(d_model, 3 * d_model), "attn.c_attn.b": vec(3 * d_model),
        "attn.c_proj.w": mat(d_model, d_model), "attn.c_proj.b": vec(d_model),
        "ln_2.g": vec(d_model, shift=1.0), "ln_2.b": vec(d_model),
        "mlp.c_fc.w": mat(d_model, 4 * d_model), "mlp.c_fc.b": vec(4 * d_model),
        "mlp.c_proj.w": mat(4 * d_model, d_model), "mlp.c_proj.b": vec(d_model),
    }  # fmt: skip


def prepare_weights(w: dict[str, np.ndarray], dtype) -> dict[str, np.ndarray]:
    """Convert float32 weights to what the kernels expect.

    Matrices are cast to the model dtype. Vectors (biases, LayerNorm scale and
    shift) stay float32 and become ``[N, 1]`` columns, because the engines
    consume them as float32 per-partition operands.
    """
    out = {}
    for key in WEIGHT_KEYS:
        a = np.asarray(w[key], dtype=np.float32)
        out[key] = a.astype(dtype) if a.ndim == 2 else np.ascontiguousarray(a.reshape(-1, 1))
    return out


def run_block(x, weights, n_heads, *, eps=1e-5, backend="pysim", target=sim.DEFAULT_TARGET,
              return_intermediates=False):  # fmt: skip
    """Run one GPT-2 block on the NKI kernels.

    Args:
        x: ``[T, D]`` hidden states, token-major, already in the model dtype.
        weights: output of :func:`prepare_weights` for the same dtype.
        n_heads: number of attention heads.
    Returns:
        ``[T, D]`` block output in the dtype of ``x``. With
        ``return_intermediates=True`` also a dict of every kernel output,
        converted to token-major so it lines up with the PyTorch reference.
    """
    w = weights

    def call(kernel, *args):
        return sim.run(kernel, *args, backend=backend, target=target)

    x = np.ascontiguousarray(x)
    n1 = call(layernorm_T, x, w["ln_1.g"], w["ln_1.b"], eps)
    qkv = call(linear_T, n1, w["attn.c_attn.w"], w["attn.c_attn.b"])
    att = call(causal_attention_T(n_heads), qkv)
    proj = call(linear_T, att, w["attn.c_proj.w"], w["attn.c_proj.b"])
    x_mid = call(residual_add_T, x, proj)
    n2 = call(layernorm_T, x_mid, w["ln_2.g"], w["ln_2.b"], eps)
    mlp_out = call(
        mlp_T, n2, w["mlp.c_fc.w"], w["mlp.c_fc.b"], w["mlp.c_proj.w"], w["mlp.c_proj.b"]
    )
    out = call(residual_add_T, x_mid, mlp_out)
    if not return_intermediates:
        return out
    inter = {
        "ln_1": n1.T, "qkv": qkv.T, "attention": att.T, "attn_proj": proj.T,
        "resid_mid": x_mid, "ln_2": n2.T, "mlp": mlp_out.T, "resid_out": out,
    }  # fmt: skip
    return out, inter
