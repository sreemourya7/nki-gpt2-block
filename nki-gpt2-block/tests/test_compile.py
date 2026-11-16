"""Compile every kernel with the real Neuron compiler and run the result.

These tests go beyond the Python simulator: neuronx-cc has to accept the
instruction stream for the target, allocate SBUF and PSUM, and emit a NEFF.
The compiled instructions are then executed on the compiler's own simulator
and compared against the Python simulator.

Skipped automatically when the neuronx-cc package is not installed.
"""

import numpy as np
import pytest
from helpers import DTYPES, as_dtype, fm, randn, rng

from nki_gpt2_block import sim
from nki_gpt2_block.kernels import (
    causal_attention_T,
    layernorm_T,
    linear_T,
    mlp_T,
    residual_add_T,
)

pytestmark = [
    pytest.mark.neuronxcc,
    pytest.mark.skipif(not sim.have_neuronx_cc(), reason="neuronx-cc is not installed"),
]

T, D, H = 130, 256, 4  # 130 tokens: one full and one partial token tile


def _cases(dtype_name):
    r = rng(0)
    x, _ = as_dtype(randn(r, (T, D), 2.0), dtype_name)
    y, _ = as_dtype(randn(r, (T, D)), dtype_name)
    W1, _ = as_dtype(randn(r, (D, 2 * D), 0.05), dtype_name)
    W2, _ = as_dtype(randn(r, (2 * D, D), 0.05), dtype_name)
    qkv, _ = as_dtype(randn(r, (T, 3 * D)), dtype_name)
    g, b = randn(r, (D, 1), 0.5, 1.0), randn(r, (D, 1))
    b1 = randn(r, (2 * D, 1))
    return {
        "linear": (linear_T, (fm(x), W1, b1)),
        "mlp": (mlp_T, (fm(x), W1, b1, W2, b)),
        "layernorm": (layernorm_T, (x, g, b)),
        "attention": (causal_attention_T(H), (fm(qkv),)),
        "residual_add": (residual_add_T, (x, fm(y))),
    }


@pytest.mark.parametrize("target", ["trn2", "trn3"])
@pytest.mark.parametrize("dtype_name", DTYPES)
@pytest.mark.parametrize("name", ["linear", "mlp", "layernorm", "attention", "residual_add"])
def test_kernel_compiles_and_matches_python_simulator(name, dtype_name, target, tmp_path):
    kernel, args = _cases(dtype_name)[name]
    compiled = sim.run(kernel, *args, backend="neuronx-cc", target=target, artifacts_dir=tmp_path)
    assert (tmp_path / "kernel.neff").exists(), "neuronx-cc did not emit a NEFF"
    assert sim.read_instruction_stats(tmp_path), "no instruction statistics were written"
    python_sim = sim.run(kernel, *args, backend="pysim", target=target)
    assert compiled.dtype == python_sim.dtype and compiled.shape == python_sim.shape
    # The two simulators may round differently by one unit in the last place.
    a, b = compiled.astype(np.float32), python_sim.astype(np.float32)
    ulp = 1e-5 if dtype_name == "float32" else 2.0**-7
    assert np.abs(a - b).max() <= ulp * max(np.abs(b).max(), 1.0)
