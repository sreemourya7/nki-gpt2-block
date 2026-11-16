"""Causal multi-head self-attention kernel in feature-major layout.

Input is the output of the QKV projection, ``qkvT[3*D, T]``, feature-major.
For head ``h`` with head size ``d`` the rows ``h*d : (h+1)*d`` of each third
are ``qT``, ``kT`` and ``vT`` with shape ``[d, T]``.

Layout decisions
----------------
* **Scores** ``S = Q @ K.T`` contract over the head dimension ``d``. ``qT`` and
  ``kT`` already have ``d`` on partitions, so they are the stationary and
  moving operands as loaded: no transposes, and PSUM holds ``S[q_tile, keys]``
  with queries on partitions and keys on the free dimension.
* **Softmax** reduces over keys. Keys are on the free dimension, which is the
  only dimension ``tensor_reduce`` and the activation accumulator can reduce.
* **Context** ``P @ V`` contracts over keys, so now keys must be on partitions
  for both operands. ``P`` and ``vT`` are transposed on the Tensor Engine in
  128-wide tiles. The result is written as ``oT[d, q_tile]``, feature-major,
  ready for the output projection.

Softmax in three passes over the score tile
-------------------------------------------
1. ``tensor_scalar``  : PSUM -> SBUF eviction with the ``1/sqrt(d)`` scale fused.
2. ``affine_select``  : causal mask, generated on the fly (no mask tensor).
3. ``tensor_reduce``  : -max per query row.
4. ``activation(exp)``: ``exp(s - max)`` with the row sum accumulated by the
   same instruction.
5. ``reciprocal`` on the ``[q, 1]`` sums, then one ``tensor_scalar`` to normalize.

Causality is also used to skip work: for the query tile starting at ``q0`` only
keys ``< q0 + tq`` are loaded, scored and contracted.

Limit: all keys of one query tile live in a single PSUM tile, so ``T <= 512``.
Longer sequences need a running max/sum across key blocks (FlashAttention-style
online softmax); see the README.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ._common import MOVING_FMAX, PMAX, tiles

# Value written to masked positions. exp(MASK_VALUE - max) underflows to exactly 0.
MASK_VALUE = -1.0e30


def _make_attention(n_heads):
    @nki.jit
    def causal_attention_T(qkvT):
        """Causal multi-head self-attention.

        Args:
            qkvT: ``[3*D, T]`` concatenated Q, K, V projections, feature-major.
        Returns:
            ``[D, T]`` attention output (before the output projection) in HBM,
            feature-major, same dtype as ``qkvT``.
        """
        D3, T = qkvT.shape
        assert D3 % 3 == 0
        D = D3 // 3
        assert D % n_heads == 0
        d = D // n_heads
        assert d <= PMAX, "head size must fit the partition dimension"
        assert T <= MOVING_FMAX, f"T={T} exceeds the single-PSUM-tile limit of {MOVING_FMAX}"
        scale = 1.0 / (d**0.5)
        oT = nl.ndarray((D, T), dtype=qkvT.dtype, buffer=nl.shared_hbm)
        n_key_tiles = (T + PMAX - 1) // PMAX

        for h in range(n_heads):
            q_rows = slice(h * d, (h + 1) * d)
            k_rows = slice(D + h * d, D + (h + 1) * d)
            v_rows = slice(2 * D + h * d, 2 * D + (h + 1) * d)

            # kT[d, T]: moving operand for the score matmul, loaded once per head.
            k_sb = nl.ndarray((d, T), dtype=qkvT.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=k_sb, src=qkvT[k_rows, :])

            # V[keys, d] per key tile: transpose vT on the Tensor Engine once per head.
            v_sb = nl.ndarray((PMAX, n_key_tiles, d), dtype=qkvT.dtype, buffer=nl.sbuf)
            for ki, (k0, tk) in enumerate(tiles(T, PMAX)):
                vT_sb = nl.ndarray((d, tk), dtype=qkvT.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=vT_sb, src=qkvT[v_rows, k0 : k0 + tk])
                v_ps = nl.ndarray((tk, d), dtype=qkvT.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=v_ps, data=vT_sb, engine=nisa.engine.tensor)
                nisa.tensor_copy(dst=v_sb[0:tk, ki, :], src=v_ps)

            for q0, tq in tiles(T, PMAX):
                n_keys = q0 + tq  # causal: later keys can never be attended to

                q_sb = nl.ndarray((d, tq), dtype=qkvT.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=q_sb, src=qkvT[q_rows, q0 : q0 + tq])

                # S[tq, n_keys] = qT.T @ kT
                s_ps = nl.ndarray((tq, n_keys), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=s_ps, stationary=q_sb, moving=k_sb[:, 0:n_keys])

                s_sb = nl.ndarray((tq, n_keys), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=s_sb, data=s_ps, op0=nl.multiply, operand0=scale)

                # Keep key x for query (q0 + partition) when q0 + partition - x >= 0.
                m_sb = nl.ndarray((tq, n_keys), dtype=nl.float32, buffer=nl.sbuf)
                nisa.affine_select(
                    dst=m_sb,
                    pattern=[[-1, n_keys]],
                    channel_multiplier=1,
                    offset=q0,
                    on_true_tile=s_sb,
                    on_false_value=MASK_VALUE,
                    cmp_op=nl.greater_equal,
                )

                neg_max = nl.ndarray((tq, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(
                    dst=neg_max, op=nl.max, data=m_sb, axis=1, negate=True, keepdims=True
                )
                p_sb = nl.ndarray((tq, n_keys), dtype=nl.float32, buffer=nl.sbuf)
                row_sum = nl.ndarray((tq, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(
                    dst=p_sb,
                    op=nl.exp,
                    data=m_sb,
                    bias=neg_max,
                    scale=1.0,
                    reduce_op=nl.add,
                    reduce_res=row_sum,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce,
                )
                r_inv = nl.ndarray((tq, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.reciprocal(dst=r_inv, data=row_sum)
                nisa.tensor_scalar(dst=p_sb, data=p_sb, op0=nl.multiply, operand0=r_inv)

                # oT[d, tq] = sum over key tiles of V[keys, d].T @ P.T[keys, tq]
                o_ps = nl.ndarray((d, tq), dtype=nl.float32, buffer=nl.psum)
                for ki, (k0, tk_full) in enumerate(tiles(n_keys, PMAX)):
                    tk = tk_full
                    pT_ps = nl.ndarray((tk, tq), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_transpose(
                        dst=pT_ps, data=p_sb[:, k0 : k0 + tk], engine=nisa.engine.tensor
                    )
                    # The matmul moving operand is read in the model dtype, like the weights.
                    pT_sb = nl.ndarray((tk, tq), dtype=qkvT.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=pT_sb, src=pT_ps)
                    nisa.nc_matmul(
                        dst=o_ps, stationary=v_sb[0:tk, ki, :], moving=pT_sb, accumulate=(ki > 0)
                    )
                o_sb = nl.ndarray((d, tq), dtype=qkvT.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=o_sb, src=o_ps)
                nisa.dma_copy(dst=oT[q_rows, q0 : q0 + tq], src=o_sb)
        return oT

    return causal_attention_T


_CACHE = {}


def causal_attention_T(n_heads):
    """Return the attention kernel specialized for ``n_heads`` (a compile-time constant)."""
    if n_heads not in _CACHE:
        _CACHE[n_heads] = _make_attention(n_heads)
    return _CACHE[n_heads]
