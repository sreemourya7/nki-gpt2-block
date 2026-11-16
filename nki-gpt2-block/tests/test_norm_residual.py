import numpy as np
import pytest
from helpers import DTYPES, as_dtype, assert_close, fm, randn, rng, t

from nki_gpt2_block import reference as ref
from nki_gpt2_block import sim
from nki_gpt2_block.kernels import layernorm_T, residual_add_T


@pytest.mark.parametrize("dtype_name", DTYPES)
@pytest.mark.parametrize("T", [1, 128, 300])
def test_layernorm_matches_torch(dtype_name, T):
    r = rng(0)
    D = 256
    # Non-zero mean and non-unit variance, so centering and scaling both matter.
    x, x32 = as_dtype(randn(r, (T, D), scale=3.0, shift=1.5), dtype_name)
    g, b = randn(r, (D, 1), 0.5, 1.0), randn(r, (D, 1))
    got = sim.run(layernorm_T, x, g, b)
    assert got.dtype == DTYPES[dtype_name] and got.shape == (D, T)
    assert_close(got.T, ref.layernorm(t(x32), t(g[:, 0]), t(b[:, 0])).numpy(), dtype_name)


def test_layernorm_large_offset_is_stable():
    """Variance is computed from centered values, so a large mean must not hurt it."""
    r = rng(1)
    D = 256
    x = randn(r, (16, D), scale=0.5, shift=300.0)
    g, b = np.ones((D, 1), np.float32), np.zeros((D, 1), np.float32)
    got = sim.run(layernorm_T, x, g, b)
    want = ref.layernorm(t(x), t(g[:, 0]), t(b[:, 0])).numpy()
    assert np.abs(got.T - want).max() < 1e-3


def test_layernorm_eps_is_used():
    D = 128
    x = np.zeros((4, D), np.float32)  # zero variance: output must be exactly beta, not NaN
    g, b = np.ones((D, 1), np.float32), np.full((D, 1), 0.25, np.float32)
    got = sim.run(layernorm_T, x, g, b, 1e-5)
    np.testing.assert_allclose(got, 0.25)


@pytest.mark.parametrize("dtype_name", DTYPES)
@pytest.mark.parametrize("T", [1, 130])
def test_residual_add_matches_numpy(dtype_name, T):
    r = rng(2)
    x, x32 = as_dtype(randn(r, (T, 256)), dtype_name)
    y, y32 = as_dtype(randn(r, (T, 256)), dtype_name)
    got = sim.run(residual_add_T, x, fm(y))
    assert got.dtype == DTYPES[dtype_name] and got.shape == (T, 256)
    assert_close(got, x32 + y32, dtype_name)
