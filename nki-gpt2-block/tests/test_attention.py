import numpy as np
import pytest
from helpers import DTYPES, as_dtype, assert_close, fm, randn, rng, t

from nki_gpt2_block import reference as ref
from nki_gpt2_block import sim
from nki_gpt2_block.kernels import causal_attention_T

D, H = 256, 4


# 1: single token. 129/300: partial query and key tiles. 512: the supported maximum.
@pytest.mark.parametrize("dtype_name", DTYPES)
@pytest.mark.parametrize("T", [1, 5, 129, 300, 512])
def test_attention_matches_torch(dtype_name, T):
    qkv, qkv32 = as_dtype(randn(rng(T), (T, 3 * D), 1.5), dtype_name)
    got = sim.run(causal_attention_T(H), fm(qkv))
    assert got.dtype == DTYPES[dtype_name] and got.shape == (D, T)
    assert_close(got.T, ref.causal_attention(t(qkv32), H).numpy(), dtype_name)


def test_reference_agrees_with_torch_builtin():
    """Guard the reference itself against torch's scaled_dot_product_attention."""
    import torch

    T = 37
    qkv = t(randn(rng(0), (T, 3 * D)))
    q, k, v = (x.view(T, H, D // H).transpose(0, 1) for x in qkv.split(D, dim=1))
    builtin = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    builtin = builtin.transpose(0, 1).reshape(T, D)
    assert torch.allclose(ref.causal_attention(qkv, H), builtin, atol=1e-5)


def test_attention_is_causal():
    """Changing tokens after position p must leave outputs up to p bit-identical."""
    T, p = 200, 140  # p sits inside the second query tile
    qkv = randn(rng(1), (T, 3 * D))
    changed = qkv.copy()
    changed[p + 1 :] = randn(rng(2), (T - p - 1, 3 * D), 5.0)
    a = sim.run(causal_attention_T(H), fm(qkv))
    b = sim.run(causal_attention_T(H), fm(changed))
    np.testing.assert_array_equal(a[:, : p + 1], b[:, : p + 1])
    assert not np.array_equal(a[:, p + 1 :], b[:, p + 1 :])


def test_attention_large_scores_do_not_overflow():
    """Scores far above exp()'s float32 range are fine because the row max is subtracted."""
    T = 64
    qkv = randn(rng(3), (T, 3 * D), 12.0)  # raw scores reach the thousands
    got = sim.run(causal_attention_T(H), fm(qkv))
    want = ref.causal_attention(t(qkv), H).numpy()
    assert np.isfinite(got).all()
    assert np.abs(got.T - want).max() / np.abs(want).max() < 1e-4


def test_first_token_attends_only_to_itself():
    T = 9
    qkv = randn(rng(4), (T, 3 * D))
    got = sim.run(causal_attention_T(H), fm(qkv))
    np.testing.assert_allclose(got[:, 0], qkv[0, 2 * D :], rtol=1e-6)  # output == its own V


def test_attention_rejects_long_sequences():
    with pytest.raises(AssertionError, match="exceeds the single-PSUM-tile limit"):
        sim.run(causal_attention_T(H), fm(randn(rng(5), (513, 3 * D))))
