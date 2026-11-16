"""Run a kernel without Trainium hardware.

Two backends:

``pysim``
    ``nki.simulate``: NKI's NumPy-based functional simulator. Public API,
    fast, needs only the ``nki`` package.

``neuronx-cc``
    Compiles the kernel with the real Neuron compiler into a NEFF (the binary
    a Trainium device loads) and executes the compiled instructions on the
    compiler's own simulator. This checks things the Python simulator does
    not: that the instruction stream is legal for the target and that SBUF and
    PSUM allocation succeeds. It needs the ``neuronx-cc`` package and uses an
    *internal* NKI entry point (``StandaloneKernel(_enable_simulation=True)``)
    that may change between NKI releases.

Neither backend measures performance. Timing needs a real device.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

# NKI 0.7 warns that the default logical-NeuronCore setting will change. These
# kernels are written for a single core, so pin it unless the caller chose.
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "1")

import nki

BACKENDS = ("pysim", "neuronx-cc")
DEFAULT_TARGET = "trn2"


def have_neuronx_cc() -> bool:
    try:
        import neuronxcc  # noqa: F401
    except ImportError:
        return False
    return True


def run(kernel, *args, backend: str = "pysim", target: str = DEFAULT_TARGET, artifacts_dir=None):
    """Execute ``kernel(*args)`` on a simulator and return a NumPy array."""
    if backend == "pysim":
        prev = os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE")
        os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = target
        try:
            return np.asarray(nki.simulate(kernel)(*args))
        finally:
            if prev is None:
                del os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"]
            else:
                os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = prev
    if backend == "neuronx-cc":
        from nki.framework.compiled import StandaloneKernel  # internal API

        extra = {}
        if artifacts_dir is not None:
            Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
            extra["artifacts_dir"] = str(artifacts_dir)
        compiled = kernel._to_subclass(
            StandaloneKernel, _enable_simulation=True, target=target, **extra
        )
        return np.asarray(compiled(*args))
    raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")


def read_instruction_stats(artifacts_dir) -> dict[str, int]:
    """Parse the per-opcode instruction counts neuronx-cc writes next to the NEFF."""
    counts: dict[str, int] = {}
    for path in sorted(Path(artifacts_dir).glob("sg*/instruction_stats.txt")):
        for line in path.read_text().splitlines():
            m = re.match(r"^│\s*(\S+)\s*│\s*(\d+)\s*│$", line)
            if m:
                counts[m.group(1)] = counts.get(m.group(1), 0) + int(m.group(2))
    return counts
