import numpy as np
import pytest
from helpers import DTYPES, as_dtype, assert_close, fm, randn, rng, t

from nki_gpt2_block import reference as ref
from nki_gpt2_block import sim
from nki_gpt2_block.kernels import linear_gelu_T, linear_T, mlp_T


def _case(dtype_name, T, K, N, seed=0):
    r = rng(seed)
    x, x32 = as_dtype(randn(r, (T, K)), dtype_name)
    W, W32 = as_dtype(randn(r, (K, N), 0.05), dtype_name)
    b = randn(r, (N, 1))
    return x, x32, W, W32, b


# T=1: single token. 130: partial tile. 600: more tokens than one moving tile (512).
@pytest.mark.parametrize("dtype_name", DTYPES)
@pytest.mark.parametrize("T", [1, 130, 600])
def test_linear_matches_torch(dtype_name, T):
    x, x32, W, W32, b = _case(dtype_name, T, K=256, N=384)
    got = sim.run(linear_T, fm(x), W, b)
    assert got.dtype == DTYPES[dtype_name] and got.shape == (384, T)
    assert_close(got.T, ref.linear(t(x32), t(W32), t(b[:, 0])).numpy(), dtype_name)


@pytest.mark.parametrize("dtype_name", DTYPES)
def test_linear_fused_gelu_matches_torch(dtype_name):
    x, x32, W, W32, b = _case(dtype_name, T=130, K=256, N=128)
    got = sim.run(linear_gelu_T, fm(x), W, b)
    want = ref.gelu_tanh(ref.linear(t(x32), t(W32), t(b[:, 0]))).numpy()
    assert_close(got.T, want, dtype_name)


@pytest.mark.parametrize("dtype_name", DTYPES)
@pytest.mark.parametrize("T", [5, 600])
def test_mlp_matches_torch(dtype_name, T):
    r = rng(1)
    K, M = 256, 1024
    x, x32 = as_dtype(randn(r, (T, K)), dtype_name)
    W1, W1_32 = as_dtype(randn(r, (K, M), 0.05), dtype_name)
    W2, W2_32 = as_dtype(randn(r, (M, K), 0.05), dtype_name)
    b1, b2 = randn(r, (M, 1)), randn(r, (K, 1))
    got = sim.run(mlp_T, fm(x), W1, b1, W2, b2)
    want = ref.mlp(t(x32), t(W1_32), t(b1[:, 0]), t(W2_32), t(b2[:, 0])).numpy()
    assert_close(got.T, want, dtype_name)


def test_mlp_equals_two_separate_linears():
    """Fusing the two layers must not change the float32 result."""
    r = rng(2)
    x, W1, W2 = randn(r, (40, 256)), randn(r, (256, 512), 0.05), randn(r, (512, 256), 0.05)
    b1, b2 = randn(r, (512, 1)), randn(r, (256, 1))
    fused = sim.run(mlp_T, fm(x), W1, b1, W2, b2)
    hidden = sim.run(linear_gelu_T, fm(x), W1, b1)
    separate = sim.run(linear_T, hidden, W2, b2)
    np.testing.assert_array_equal(fused, separate)


def test_linear_rejects_bad_shapes():
    r = rng(3)
    x, W = randn(r, (4, 256)), randn(r, (256, 128))
    with pytest.raises(AssertionError, match="bias must have shape"):
        sim.run(linear_T, fm(x), W, randn(r, (128,)))
    with pytest.raises(AssertionError, match="multiples of 128"):
        sim.run(linear_T, fm(randn(r, (4, 100))), randn(r, (100, 128)), randn(r, (128, 1)))
