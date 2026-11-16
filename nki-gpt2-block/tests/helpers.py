"""Shared test utilities."""

import ml_dtypes
import numpy as np
import torch

DTYPES = {"float32": np.float32, "bfloat16": ml_dtypes.bfloat16}

# Maximum allowed error, as a fraction of the largest reference magnitude.
# bfloat16 keeps 8 significand bits, so a single rounding is already ~0.4%.
REL_TOL = {"float32": 2e-5, "bfloat16": 2.5e-2}


def rng(seed=0):
    return np.random.default_rng(seed)


def randn(r, shape, scale=1.0, shift=0.0):
    return (r.standard_normal(shape) * scale + shift).astype(np.float32)


def as_dtype(a, dtype_name):
    """Round to the model dtype. Returns (kernel input, the same values as float32)."""
    cast = np.ascontiguousarray(a).astype(DTYPES[dtype_name])
    return cast, cast.astype(np.float32)


def fm(a):
    """Token-major -> feature-major (contiguous)."""
    return np.ascontiguousarray(a.T)


def t(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def rel_err(got, ref):
    got = np.asarray(got).astype(np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    assert got.shape == ref.shape, f"shape {got.shape} != reference {ref.shape}"
    assert np.isfinite(got).all(), "kernel output contains NaN or Inf"
    return float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-12))


def assert_close(got, ref, dtype_name):
    err = rel_err(got, ref)
    assert err <= REL_TOL[dtype_name], f"relative error {err:.3e} > {REL_TOL[dtype_name]:.1e}"
