"""Hardware tile limits shared by every kernel.

The numbers come from the NKI ISA documentation (nki 0.7):

* SBUF and PSUM have 128 partitions, so the partition dimension (dim 0) of
  any on-chip tile is at most 128.
* ``nc_matmul`` contracts along the partition dimension, takes a stationary
  tile with at most 128 free elements, and on NeuronCore-v3 (Trainium2) a
  moving tile with at most 512 free elements (one PSUM bank of float32).
  NeuronCore-v4 allows more, but 512 keeps the kernels valid on both.
"""

PMAX = 128  # partitions per on-chip tile
STATIONARY_FMAX = 128  # nc_matmul stationary free-dimension limit
MOVING_FMAX = 512  # nc_matmul moving free-dimension limit on NeuronCore-v3


def tiles(total: int, step: int):
    """Yield ``(start, size)`` pairs covering ``range(total)`` in ``step`` chunks."""
    for start in range(0, total, step):
        yield start, min(step, total - start)
