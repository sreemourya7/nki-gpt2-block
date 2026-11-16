"""LayerNorm kernel: token-major in, feature-major out.

Layout
------
LayerNorm reduces over features, and ``nisa.tensor_reduce`` only reduces free
dimensions, so the statistics need *tokens on the partition dimension*:
``x[t_tile, D]``. The linear layer that always follows wants features on the
partition dimension. The kernel therefore does the layout change itself with
Tensor Engine transposes (128x128 tiles, SBUF -> PSUM).

That transpose is also what makes the affine part free: after it, features sit
on partitions, so ``gamma`` and ``beta`` are per-partition vectors and the
``tensor_scalar`` that evicts PSUM to SBUF applies ``* gamma + beta`` in the
same instruction.

Per token tile the statistics take five instructions:

1. ``tensor_reduce(add, negate)``           -> -sum(x)
2. ``tensor_scalar(* 1/D)``                 -> -mean
3. ``activation(square, bias=-mean, reduce)``-> sum((x - mean)^2); the centering
   is fused into the square, no centered tile is kept
4. ``activation(rsqrt, scale=1/D, bias=eps)``-> 1/sqrt(var + eps)
5. ``tensor_scalar(+ -mean, * rstd)``        -> normalized tile
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ._common import PMAX, tiles


@nki.jit
def layernorm_T(x, gamma, beta, eps=1e-5):
    """LayerNorm over the last dimension.

    Args:
        x:     ``[T, D]`` input, token-major.
        gamma: ``[D, 1]`` float32 scale.
        beta:  ``[D, 1]`` float32 shift.
        eps:   variance epsilon (compile-time constant).
    Returns:
        ``[D, T]`` normalized output in HBM, feature-major, same dtype as ``x``.
    """
    T, D = x.shape
    assert D % PMAX == 0, "feature size must be a multiple of 128"
    assert tuple(gamma.shape) == (D, 1) and tuple(beta.shape) == (D, 1)
    assert gamma.dtype == nl.float32 and beta.dtype == nl.float32
    outT = nl.ndarray((D, T), dtype=x.dtype, buffer=nl.shared_hbm)

    # gamma/beta are loaded once and reused by every token tile.
    g_sb = nl.ndarray((PMAX, D // PMAX), dtype=nl.float32, buffer=nl.sbuf)
    b_sb = nl.ndarray((PMAX, D // PMAX), dtype=nl.float32, buffer=nl.sbuf)
    for di in range(D // PMAX):
        nisa.dma_copy(dst=g_sb[:, di : di + 1], src=gamma[di * PMAX : (di + 1) * PMAX, :])
        nisa.dma_copy(dst=b_sb[:, di : di + 1], src=beta[di * PMAX : (di + 1) * PMAX, :])

    for t0, tp in tiles(T, PMAX):
        x_sb = nl.ndarray((tp, D), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_sb, src=x[t0 : t0 + tp, :])

        neg_mean = nl.ndarray((tp, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=neg_mean, op=nl.add, data=x_sb, axis=1, negate=True, keepdims=True)
        nisa.tensor_scalar(dst=neg_mean, data=neg_mean, op0=nl.multiply, operand0=1.0 / D)

        sq = nl.ndarray(
            (tp, D), dtype=nl.float32, buffer=nl.sbuf
        )  # scratch, only the reduction is used
        sum_sq = nl.ndarray((tp, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=sq,
            op=nl.square,
            data=x_sb,
            bias=neg_mean,
            scale=1.0,
            reduce_op=nl.add,
            reduce_res=sum_sq,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )
        rstd = nl.ndarray((tp, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rstd, op=nl.rsqrt, data=sum_sq, bias=eps, scale=1.0 / D)

        # Normalized values stay float32 until the final eviction casts them.
        n_sb = nl.ndarray((tp, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=n_sb, data=x_sb, op0=nl.add, operand0=neg_mean, op1=nl.multiply, operand1=rstd
        )

        for di in range(D // PMAX):
            nT = nl.ndarray((PMAX, tp), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(
                dst=nT, data=n_sb[:, di * PMAX : (di + 1) * PMAX], engine=nisa.engine.tensor
            )
            o_sb = nl.ndarray((PMAX, tp), dtype=x.dtype, buffer=nl.sbuf)
            # Eviction with the affine fused: (nT * gamma) + beta, cast to x.dtype.
            nisa.tensor_scalar(
                dst=o_sb,
                data=nT,
                op0=nl.multiply,
                operand0=g_sb[:, di : di + 1],
                op1=nl.add,
                operand1=b_sb[:, di : di + 1],
            )
            nisa.dma_copy(dst=outT[di * PMAX : (di + 1) * PMAX, t0 : t0 + tp], src=o_sb)
    return outT
