"""Loader and model-level tests.

The first test builds a tiny random checkpoint in the Hugging Face file layout,
so the loader and the block swap are exercised without any download. The
second uses the real GPT-2 weights when they are present.
"""

import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest
from helpers import rel_err
from safetensors.numpy import save_file

from nki_gpt2_block.block import prepare_weights, random_block_weights, run_block
from nki_gpt2_block.gpt2 import _BLOCK_KEYS, Gpt2

REAL = Path(__file__).resolve().parents[1] / "weights" / "gpt2"


def _nki_block(model, i, dtype):
    w = prepare_weights(model.block_weights(i), dtype)
    return lambda x: run_block(x.astype(dtype), w, model.n_head, eps=model.eps)


@pytest.fixture
def tiny_checkpoint(tmp_path):
    d, layers, vocab, ctx = 128, 2, 50, 32
    r = np.random.default_rng(0)
    tensors = {
        "wte.weight": r.standard_normal((vocab, d)).astype(np.float32),
        "wpe.weight": (r.standard_normal((ctx, d)) * 0.1).astype(np.float32),
        "ln_f.weight": np.ones(d, np.float32),
        "ln_f.bias": np.zeros(d, np.float32),
    }
    for i in range(layers):
        for ours, theirs in _BLOCK_KEYS.items():
            tensors[f"h.{i}.{theirs}"] = random_block_weights(d, seed=i)[ours]
    save_file(tensors, str(tmp_path / "model.safetensors"))
    cfg = {"n_layer": layers, "n_head": 2, "n_embd": d, "layer_norm_epsilon": 1e-5,
           "activation_function": "gelu_new"}  # fmt: skip
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return tmp_path


def test_all_blocks_swapped_on_tiny_checkpoint(tiny_checkpoint):
    model = Gpt2(tiny_checkpoint)
    ids = np.arange(20) % 50
    want, _ = model.forward(ids)
    got, _ = model.forward(ids, {i: _nki_block(model, i, np.float32) for i in range(2)})
    assert rel_err(got, want) < 1e-4
    assert (got.argmax(-1) == want.argmax(-1)).all()


@pytest.mark.gpt2
@pytest.mark.skipif(
    not (REAL / "model.safetensors").exists(), reason="GPT-2 weights not downloaded"
)
@pytest.mark.parametrize("dtype,min_agreement", [(np.float32, 1.0), (ml_dtypes.bfloat16, 0.95)])
def test_real_gpt2_all_blocks_on_nki(dtype, min_agreement):
    model = Gpt2(REAL)
    prompt = json.loads((REAL.parents[1] / "data" / "prompt.json").read_text())
    ids = prompt["token_ids"][:96]
    want, _ = model.forward(ids)
    blocks = {i: _nki_block(model, i, dtype) for i in range(model.n_layer)}
    got, _ = model.forward(ids, blocks)
    agreement = (got.argmax(-1) == want.argmax(-1)).mean()
    assert agreement >= min_agreement, f"top-1 agreement {agreement:.3f}"


@pytest.mark.gpt2
@pytest.mark.skipif(
    not (REAL / "model.safetensors").exists(), reason="GPT-2 weights not downloaded"
)
def test_reference_matches_transformers_library():
    """The float32 reference everything is measured against must be the real GPT-2."""
    transformers = pytest.importorskip("transformers")
    import torch

    model = Gpt2(REAL)
    ids = json.loads((REAL.parents[1] / "data" / "prompt.json").read_text())["token_ids"][:64]
    ours, _ = model.forward(ids)
    hf = transformers.GPT2LMHeadModel.from_pretrained(str(REAL)).eval()
    with torch.no_grad():
        theirs = hf(torch.tensor([ids])).logits[0].numpy()
    assert np.abs(ours - theirs).max() < 1e-3  # logits are O(100)
    assert (ours.argmax(-1) == theirs.argmax(-1)).all()
