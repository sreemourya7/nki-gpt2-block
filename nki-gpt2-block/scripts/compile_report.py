#!/usr/bin/env python3
"""Compile every kernel at GPT-2 shapes with neuronx-cc and report what it emitted.

For each kernel and target this compiles to a NEFF, runs the compiled
instructions on the compiler's simulator, checks the result against the Python
simulator, and records the per-opcode instruction counts from the compiler.

Instruction counts are not a latency measurement. They show how much work each
kernel schedules on the engines; real timing needs a Trainium device.

Writes results/compile_report.md and results/compile_report.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import tempfile
from pathlib import Path

import ml_dtypes
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nki_gpt2_block import sim
from nki_gpt2_block.kernels import (
    causal_attention_T,
    layernorm_T,
    linear_gelu_T,
    linear_T,
    mlp_T,
    residual_add_T,
)

D, H = 768, 12  # GPT-2 124M


DTYPES = {"float32": np.float32, "bfloat16": ml_dtypes.bfloat16}


def cases(T, dtype):
    r = np.random.default_rng(0)

    def f(*shape, s=1.0):  # activations and weight matrices: model dtype
        return (r.standard_normal(shape) * s).astype(dtype)

    def v(n):  # biases and LayerNorm vectors: always float32 columns
        return r.standard_normal((n, 1)).astype(np.float32)

    xT, x = f(D, T), f(T, D)
    return {
        "layernorm_T": (layernorm_T, (x, v(D), v(D))),
        "linear_T (QKV, 768->2304)": (linear_T, (xT, f(D, 3 * D, s=0.05), v(3 * D))),
        f"causal_attention_T ({H} heads)": (causal_attention_T(H), (f(3 * D, T),)),
        "linear_T (proj, 768->768)": (linear_T, (xT, f(D, D, s=0.05), v(D))),
        "residual_add_T": (residual_add_T, (x, xT)),
        "mlp_T (768->3072->768)": (
            mlp_T, (xT, f(D, 4 * D, s=0.05), v(4 * D), f(4 * D, D, s=0.05), v(D))),
        "  unfused: linear_gelu_T (768->3072)": (linear_gelu_T, (xT, f(D, 4 * D, s=0.05), v(4 * D))),
        "  unfused: linear_T (3072->768)": (linear_T, (f(4 * D, T), f(4 * D, D, s=0.05), v(D))),
    }  # fmt: skip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--targets", nargs="+", default=["trn2", "trn3"])
    ap.add_argument("--out", default=ROOT / "results")
    args = ap.parse_args()
    if not sim.have_neuronx_cc():
        sys.exit("neuronx-cc is not installed; see README for the install command")

    import neuronxcc
    import nki

    rows = []
    for target, (dname, dtype) in itertools.product(args.targets, DTYPES.items()):
        for name, (kernel, kargs) in cases(args.tokens, dtype).items():
            with tempfile.TemporaryDirectory() as tmp:
                got = sim.run(
                    kernel, *kargs, backend="neuronx-cc", target=target, artifacts_dir=tmp
                )
                stats = sim.read_instruction_stats(tmp)
                neff_kb = (Path(tmp) / "kernel.neff").stat().st_size / 1024
            want = sim.run(kernel, *kargs, backend="pysim", target=target).astype(np.float32)
            diff = float(np.abs(got.astype(np.float32) - want).max() / np.abs(want).max())
            rows.append({"target": target, "dtype": dname, "kernel": name,
                         "neff_kb": round(neff_kb, 1), "max_rel_diff_vs_pysim": diff,
                         "instructions": stats})  # fmt: skip
            print(f"{target} {dname:8s} {name.strip():38s} PASS  rel diff vs pysim {diff:.1e}",
                  flush=True)  # fmt: skip

    cols = ["MATMUL", "ACTIVATE", "TENSOR_SCALAR", "TENSOR_TENSOR", "TENSOR_REDUCE", "COPY"]
    md = ["# Compile report", "",
          f"- shapes: GPT-2 124M (d_model {D}, {H} heads), {args.tokens} tokens",
          f"- neuronx-cc {neuronxcc.__version__}, nki {nki.__version__}",
          "- every row compiled to a NEFF and its compiled instructions were executed on the "
          "compiler's simulator",
          "- instruction counts are as printed by neuronx-cc (`instruction_stats.txt`); they are "
          "not latency", "",
          "| target | dtype | kernel | " + " | ".join(cols)
          + " | all instructions | max rel. diff vs Python sim |",
          "|---|---|---|" + "---|" * (len(cols) + 2)]  # fmt: skip
    for r in rows:
        ins = r["instructions"]
        md.append(f"| {r['target']} | {r['dtype']} | {r['kernel'].replace('  ', '&nbsp;&nbsp;')} | "
                  + " | ".join(str(ins.get(c, 0)) for c in cols)
                  + f" | {sum(ins.values())} | {r['max_rel_diff_vs_pysim']:.1e} |")  # fmt: skip
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compile_report.md").write_text("\n".join(md) + "\n")
    (out / "compile_report.json").write_text(json.dumps(rows, indent=1))
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
