"""Residual add that also converts feature-major back to token-major."""

import nki
import nki.isa as nisa
import nki.language as nl

from ._common import PMAX, tiles


@nki.jit
def residual_add_T(x, yT):
    """``x + yT.T``.

    The residual stream is kept token-major (what LayerNorm consumes) while
    branch outputs are feature-major (what the linear kernels produce). The
    branch tile is transposed on the Tensor Engine into PSUM and the add reads
    it straight from there, so the add is also the PSUM eviction.

    Args:
        x:  ``[T, D]`` residual stream, token-major.
        yT: ``[D, T]`` branch output, feature-major.
    Returns:
        ``[T, D]`` in HBM, token-major, same dtype as ``x``.
    """
    T, D = x.shape
    assert tuple(yT.shape) == (D, T), f"expected yT of shape ({D}, {T}), got {tuple(yT.shape)}"
    assert D % PMAX == 0, "feature size must be a multiple of 128"
    assert x.dtype == yT.dtype
    out = nl.ndarray((T, D), dtype=x.dtype, buffer=nl.shared_hbm)
    for t0, tp in tiles(T, PMAX):
        for di in range(D // PMAX):
            y_sb = nl.ndarray((PMAX, tp), dtype=yT.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=y_sb, src=yT[di * PMAX : (di + 1) * PMAX, t0 : t0 + tp])
            y_ps = nl.ndarray((tp, PMAX), dtype=yT.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=y_ps, data=y_sb, engine=nisa.engine.tensor)
            x_sb = nl.ndarray((tp, PMAX), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_sb, src=x[t0 : t0 + tp, di * PMAX : (di + 1) * PMAX])
            o_sb = nl.ndarray((tp, PMAX), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=o_sb, data1=x_sb, data2=y_ps, op=nl.add)
            nisa.dma_copy(dst=out[t0 : t0 + tp, di * PMAX : (di + 1) * PMAX], src=o_sb)
    return out
