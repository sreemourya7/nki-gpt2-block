import numpy as np
import pytest
from helpers import DTYPES, REL_TOL, as_dtype, randn, rel_err, rng, t

from nki_gpt2_block import reference as ref
from nki_gpt2_block.block import WEIGHT_KEYS, prepare_weights, random_block_weights, run_block

D, H = 256, 4


def _reference_weights(prepared):
    """What the kernels actually received, as float32 torch tensors in PyTorch shapes."""
    out = {}
    for key in WEIGHT_KEYS:
        a = prepared[key].astype(np.float32)
        out[key] = t(a[:, 0] if key.endswith((".b", ".g")) else a)
    return out


@pytest.mark.parametrize("dtype_name", DTYPES)
@pytest.mark.parametrize("T", [7, 200])
def test_block_matches_torch(dtype_name, T):
    w = prepare_weights(random_block_weights(D, seed=T), DTYPES[dtype_name])
    x, x32 = as_dtype(randn(rng(T), (T, D), 2.0), dtype_name)
    got = run_block(x, w, H)
    want = ref.gpt2_block(t(x32), _reference_weights(w), H).numpy()
    assert got.dtype == DTYPES[dtype_name]
    # Seven kernels compound their rounding, so allow twice the single-kernel bound.
    assert rel_err(got, want) <= 2 * REL_TOL[dtype_name]


def test_block_intermediates_line_up_with_reference():
    w = prepare_weights(random_block_weights(D, seed=1), np.float32)
    x = randn(rng(1), (33, D), 2.0)
    out, inter = run_block(x, w, H, return_intermediates=True)
    rw = _reference_weights(w)
    n1 = ref.layernorm(t(x), rw["ln_1.g"], rw["ln_1.b"])
    qkv = ref.linear(n1, rw["attn.c_attn.w"], rw["attn.c_attn.b"])
    att = ref.causal_attention(qkv, H)
    mid = t(x) + ref.linear(att, rw["attn.c_proj.w"], rw["attn.c_proj.b"])
    for name, want in {"ln_1": n1, "qkv": qkv, "attention": att, "resid_mid": mid}.items():
        assert rel_err(inter[name], want.numpy()) <= 5e-5, name
    np.testing.assert_array_equal(out, inter["resid_out"])
