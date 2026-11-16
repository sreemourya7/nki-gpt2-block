"""Linear layer and fused MLP kernels in feature-major layout.

Layout
------
Activations are passed *feature-major*: ``xT`` has shape ``[features, tokens]``.
``nc_matmul`` computes ``stationary.T @ moving`` and contracts along the
partition dimension, so with

    stationary = W[k_tile, n_tile]      (K on partitions, <=128 outputs free)
    moving     = xT[k_tile, t_tile]     (K on partitions, <=512 tokens free)

the PSUM result is ``yT[n_tile, t_tile]`` with *output features on the
partition dimension*. That is what makes the bias cheap: a per-feature bias is
a per-partition ``[128, 1]`` vector, which ``nisa.activation`` adds for free
while it evicts PSUM to SBUF. Bias add, the activation function, the
PSUM->SBUF move and the dtype cast are one instruction.

In token-major layout the bias would vary along the free dimension and need a
broadcast tile plus a separate ``tensor_tensor`` add.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ._common import MOVING_FMAX, PMAX, tiles

_ACTIVATIONS = {
    "none": nl.copy,
    "gelu_tanh": nl.gelu_apprx_tanh,  # GPT-2's "gelu_new"
    "gelu": nl.gelu,
}


def _load_feature_major(xT, t0, tt):
    """DMA ``xT[:, t0:t0+tt]`` into SBUF as ``[128, K/128, tt]`` (one moving tile per K chunk)."""
    K = xT.shape[0]
    x_sb = nl.ndarray((PMAX, K // PMAX, tt), dtype=xT.dtype, buffer=nl.sbuf)
    for ki in range(K // PMAX):
        nisa.dma_copy(dst=x_sb[:, ki, :], src=xT[ki * PMAX : (ki + 1) * PMAX, t0 : t0 + tt])
    return x_sb


def _linear_tile(dst, x_sb, W, b, ni, act_op):
    """Compute one ``[128, tt]`` output tile: ``act(W[:, n_tile].T @ x + b[n_tile])``.

    ``x_sb`` stays resident in SBUF; only the 128x128 weight tiles stream in.
    Partial products for the K chunks accumulate in a float32 PSUM tile.
    """
    n_chunks_k = x_sb.shape[1]
    tt = x_sb.shape[2]
    acc = nl.ndarray((PMAX, tt), dtype=nl.float32, buffer=nl.psum)
    for ki in range(n_chunks_k):
        w_sb = nl.ndarray((PMAX, PMAX), dtype=W.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_sb, src=W[ki * PMAX : (ki + 1) * PMAX, ni * PMAX : (ni + 1) * PMAX])
        nisa.nc_matmul(dst=acc, stationary=w_sb, moving=x_sb[:, ki, :], accumulate=(ki > 0))
    b_sb = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_sb, src=b[ni * PMAX : (ni + 1) * PMAX, :])
    # One instruction: bias add + activation + PSUM->SBUF eviction + cast to dst.dtype.
    nisa.activation(dst=dst, op=act_op, data=acc, bias=b_sb, scale=1.0)


def _check_linear_shapes(xT, W, b):
    K, _ = xT.shape
    Kw, N = W.shape
    assert K == Kw, f"xT has {K} features but W expects {Kw}"
    assert K % PMAX == 0 and N % PMAX == 0, "feature sizes must be multiples of 128"
    assert tuple(b.shape) == (N, 1), f"bias must have shape ({N}, 1), got {tuple(b.shape)}"
    assert b.dtype == nl.float32, "bias must be float32 (activation bias vector)"


def _make_linear(act_name):
    act_op = _ACTIVATIONS[act_name]

    @nki.jit
    def linear_T(xT, W, b):
        """``yT = act(W.T @ xT + b)``.

        Args:
            xT: ``[K, T]`` input, feature-major.
            W:  ``[K, N]`` weight (GPT-2 Conv1D layout: in x out).
            b:  ``[N, 1]`` float32 bias.
        Returns:
            ``[N, T]`` output in HBM, feature-major, same dtype as ``xT``.
        """
        _check_linear_shapes(xT, W, b)
        _, T = xT.shape
        N = W.shape[1]
        yT = nl.ndarray((N, T), dtype=xT.dtype, buffer=nl.shared_hbm)
        for t0, tt in tiles(T, MOVING_FMAX):
            x_sb = _load_feature_major(xT, t0, tt)
            for ni in range(N // PMAX):
                out = nl.ndarray((PMAX, tt), dtype=xT.dtype, buffer=nl.sbuf)
                _linear_tile(out, x_sb, W, b, ni, act_op)
                nisa.dma_copy(dst=yT[ni * PMAX : (ni + 1) * PMAX, t0 : t0 + tt], src=out)
        return yT

    return linear_T


linear_T = _make_linear("none")
linear_gelu_T = _make_linear("gelu_tanh")


@nki.jit
def mlp_T(xT, W1, b1, W2, b2):
    """GPT-2 MLP as one kernel: ``W2.T @ gelu_tanh(W1.T @ xT + b1) + b2``.

    The hidden activation ``[4K, T]`` never leaves SBUF: each hidden tile is
    written by the fused bias+GELU eviction of the first matmul and read back
    as the moving operand of the second, so the kernel saves one HBM store and
    one HBM load of the widest tensor in the block.

    Args:
        xT: ``[K, T]`` input, feature-major.
        W1: ``[K, M]``, b1: ``[M, 1]`` float32.
        W2: ``[M, N]``, b2: ``[N, 1]`` float32.
    Returns:
        ``[N, T]`` output in HBM, feature-major.
    """
    _check_linear_shapes(xT, W1, b1)
    M = W1.shape[1]
    assert W2.shape[0] == M, "W2 input size must match W1 output size"
    assert W2.shape[1] % PMAX == 0 and tuple(b2.shape) == (W2.shape[1], 1)
    assert b2.dtype == nl.float32
    _, T = xT.shape
    N = W2.shape[1]
    yT = nl.ndarray((N, T), dtype=xT.dtype, buffer=nl.shared_hbm)
    for t0, tt in tiles(T, MOVING_FMAX):
        x_sb = _load_feature_major(xT, t0, tt)
        h_sb = nl.ndarray((PMAX, M // PMAX, tt), dtype=xT.dtype, buffer=nl.sbuf)
        for mi in range(M // PMAX):
            _linear_tile(h_sb[:, mi, :], x_sb, W1, b1, mi, nl.gelu_apprx_tanh)
        for ni in range(N // PMAX):
            out = nl.ndarray((PMAX, tt), dtype=xT.dtype, buffer=nl.sbuf)
            _linear_tile(out, h_sb, W2, b2, ni, nl.copy)
            nisa.dma_copy(dst=yT[ni * PMAX : (ni + 1) * PMAX, t0 : t0 + tt], src=out)
    return yT
