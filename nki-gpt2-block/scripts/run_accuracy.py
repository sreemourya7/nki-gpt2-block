#!/usr/bin/env python3
"""Accuracy report: GPT-2 on the NKI kernels versus float32 PyTorch.

Part A looks inside single blocks: every kernel output is compared with the
float32 reference at the real hidden states of the evaluation prompt.

Part B replaces all blocks with the NKI block and compares the final
next-token distribution with the float32 model. A PyTorch bfloat16 run of the
same model is included as a baseline, to separate "bfloat16 loses precision"
from "the kernels are wrong".

Writes results/accuracy.md and results/accuracy.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import ml_dtypes
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nki_gpt2_block import reference as ref
from nki_gpt2_block import sim
from nki_gpt2_block.block import prepare_weights, run_block
from nki_gpt2_block.gpt2 import Gpt2

DTYPES = {"float32": np.float32, "bfloat16": ml_dtypes.bfloat16}
STAGES = ["ln_1", "qkv", "attention", "attn_proj", "resid_mid", "ln_2", "mlp", "resid_out"]


def rel_err(got, want):
    got, want = np.asarray(got).astype(np.float32), np.asarray(want, dtype=np.float32)
    return float(np.abs(got - want).max() / np.abs(want).max())


def torch_block(model, i, x, torch_dtype):
    """Reference block in an arbitrary torch dtype (used for the bfloat16 baseline)."""
    w = {k: torch.from_numpy(v).to(torch_dtype) for k, v in model.block_weights(i).items()}
    with torch.no_grad():
        out = ref.gpt2_block_intermediates(
            torch.from_numpy(x).to(torch_dtype), w, model.n_head, model.eps
        )
    return {k: v.float().numpy() for k, v in out.items()}


def per_kernel_errors(model, hidden_in, layers, backend, target):
    rows = []
    for i in layers:
        x = hidden_in[i]
        want = torch_block(model, i, x, torch.float32)
        for name, dtype in DTYPES.items():
            w = prepare_weights(model.block_weights(i), dtype)
            _, got = run_block(x.astype(dtype), w, model.n_head, eps=model.eps, backend=backend,
                               target=target, return_intermediates=True)  # fmt: skip
            rows.append({"block": i, "run": f"NKI {name}",
                         **{s: rel_err(got[s], want[s]) for s in STAGES}})  # fmt: skip
        base = torch_block(model, i, x, torch.bfloat16)
        rows.append({"block": i, "run": "PyTorch bfloat16 (baseline)",
                     **{s: rel_err(base[s], want[s]) for s in STAGES}})  # fmt: skip
    return rows


def distribution_metrics(logits, ref_logits):
    lp = torch.log_softmax(torch.from_numpy(logits), -1)
    lq = torch.log_softmax(torch.from_numpy(ref_logits), -1)
    kl = (lq.exp() * (lq - lp)).sum(-1)
    top1 = logits.argmax(-1) == ref_logits.argmax(-1)
    return {
        "top1_agreement": float(top1.mean()),
        "positions": int(top1.size),
        "disagreements": int((~top1).sum()),
        "max_abs_logit_diff": float(np.abs(logits - ref_logits).max()),
        "mean_kl": float(kl.mean()),
        "max_kl": float(kl.max()),
    }


def end_to_end(model, ids, backend, target):
    ref_logits, _ = model.forward(ids)
    rows = []
    for name, dtype in DTYPES.items():
        t0 = time.time()
        blocks = {}
        for i in range(model.n_layer):
            w = prepare_weights(model.block_weights(i), dtype)
            blocks[i] = lambda x, w=w, dtype=dtype: run_block(
                x.astype(dtype), w, model.n_head, eps=model.eps, backend=backend, target=target
            )
        logits, _ = model.forward(ids, blocks)
        rows.append({"run": f"NKI kernels, {name}", **distribution_metrics(logits, ref_logits),
                     "seconds": round(time.time() - t0, 1)})  # fmt: skip
    baseline = {
        i: (lambda x, i=i: torch_block(model, i, x, torch.bfloat16)["resid_out"])
        for i in range(model.n_layer)
    }
    logits, _ = model.forward(ids, baseline)
    rows.append({"run": "PyTorch CPU, bfloat16 (baseline)",
                 **distribution_metrics(logits, ref_logits), "seconds": None})  # fmt: skip
    return rows


def to_markdown(meta, kernel_rows, e2e_rows):
    out = ["# Accuracy report", ""]
    out += [f"- {k}: {v}" for k, v in meta.items()]
    out += ["", "## Part A: per-kernel error inside a block", "",
            "Maximum absolute error divided by the largest reference magnitude, against the "
            "float32 PyTorch block with the original weights. The block input is the "
            "float32 model's real hidden state for the evaluation prompt. Errors accumulate "
            "from left to right because each kernel consumes the previous kernel's output. The "
            "PyTorch bfloat16 rows run the same block with PyTorch's own CPU bfloat16 ops.", "",
            "| block | run | " + " | ".join(STAGES) + " |",
            "|---|---|" + "---|" * len(STAGES)]  # fmt: skip
    for r in kernel_rows:
        out.append(
            f"| {r['block']} | {r['run']} | " + " | ".join(f"{r[s]:.1e}" for s in STAGES) + " |"
        )
    out += ["", f"## Part B: all {meta['blocks']} blocks on NKI kernels", "",
            "Next-token distributions over every prompt position, against the float32 "
            "PyTorch model. Embeddings, the final LayerNorm and the output projection run in "
            "float32 PyTorch in every row.", "",
            "| run | top-1 agreement | max abs logit diff | mean KL | max KL |", "|---|---|---|---|---|"]  # fmt: skip
    for r in e2e_rows:
        agree = (
            f"{r['positions'] - r['disagreements']}/{r['positions']} ({r['top1_agreement']:.1%})"
        )
        out.append(f"| {r['run']} | {agree} | {r['max_abs_logit_diff']:.2e} | "
                   f"{r['mean_kl']:.2e} | {r['max_kl']:.2e} |")  # fmt: skip
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model-dir", default=ROOT / "weights" / "gpt2")
    ap.add_argument("--tokens", type=int, default=0, help="use only the first N prompt tokens")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 5, 11], help="blocks for part A")
    ap.add_argument("--backend", choices=sim.BACKENDS, default="pysim")
    ap.add_argument("--target", default=sim.DEFAULT_TARGET)
    ap.add_argument("--out", default=ROOT / "results")
    args = ap.parse_args()

    model = Gpt2(args.model_dir)
    ids = json.loads((ROOT / "data" / "prompt.json").read_text())["token_ids"]
    if args.tokens:
        ids = ids[: args.tokens]

    import nki

    meta = {"model": "GPT-2 124M (openai-community/gpt2)", "blocks": model.n_layer,
            "prompt tokens": len(ids), "simulator backend": args.backend, "target": args.target,
            "nki": nki.__version__, "torch": torch.__version__}  # fmt: skip
    _, hidden = model.forward(ids)
    hidden_in = [model.embed(ids)] + hidden[:-1]

    print("part A: per-kernel errors ...", flush=True)
    kernel_rows = per_kernel_errors(model, hidden_in, args.layers, args.backend, args.target)
    print("part B: all blocks on NKI ...", flush=True)
    e2e_rows = end_to_end(model, ids, args.backend, args.target)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.backend == "pysim" else "_" + args.backend.replace("-", "_")
    md = to_markdown(meta, kernel_rows, e2e_rows)
    (out / f"accuracy{suffix}.md").write_text(md)
    (out / f"accuracy{suffix}.json").write_text(
        json.dumps({"meta": meta, "per_kernel": kernel_rows, "end_to_end": e2e_rows}, indent=1)
    )
    print(md)


if __name__ == "__main__":
    main()
