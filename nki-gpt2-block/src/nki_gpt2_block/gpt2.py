"""Minimal GPT-2 loader and float32 PyTorch forward pass.

Reads the Hugging Face ``model.safetensors`` of a GPT-2 checkpoint directly,
so the project does not depend on ``transformers``. The forward pass can swap
any block for a custom implementation, which is how the NKI block is dropped
into the real model.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file

from . import reference as ref

# Our key -> key inside an "h.<i>." block of the checkpoint.
_BLOCK_KEYS = {
    "ln_1.g": "ln_1.weight", "ln_1.b": "ln_1.bias",
    "attn.c_attn.w": "attn.c_attn.weight", "attn.c_attn.b": "attn.c_attn.bias",
    "attn.c_proj.w": "attn.c_proj.weight", "attn.c_proj.b": "attn.c_proj.bias",
    "ln_2.g": "ln_2.weight", "ln_2.b": "ln_2.bias",
    "mlp.c_fc.w": "mlp.c_fc.weight", "mlp.c_fc.b": "mlp.c_fc.bias",
    "mlp.c_proj.w": "mlp.c_proj.weight", "mlp.c_proj.b": "mlp.c_proj.bias",
}  # fmt: skip

BlockFn = Callable[[np.ndarray], np.ndarray]


class Gpt2:
    """GPT-2 weights as float32 NumPy arrays plus a reference forward pass."""

    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "config.json").read_text())
        self.n_layer, self.n_head, self.d_model = cfg["n_layer"], cfg["n_head"], cfg["n_embd"]
        self.eps = cfg.get("layer_norm_epsilon", 1e-5)
        raw = load_file(str(model_dir / "model.safetensors"))
        # Some exports prefix every tensor with "transformer.".
        self.tensors = {
            k.removeprefix("transformer."): v.astype(np.float32) for k, v in raw.items()
        }
        if cfg.get("activation_function", "gelu_new") != "gelu_new":
            raise ValueError("only the tanh-approximated GELU (gelu_new) is implemented")

    def block_weights(self, i: int) -> dict[str, np.ndarray]:
        """Float32 weights of block ``i`` under the key names used by ``block.py``."""
        return {ours: self.tensors[f"h.{i}.{theirs}"] for ours, theirs in _BLOCK_KEYS.items()}

    def embed(self, token_ids) -> np.ndarray:
        """Token plus position embeddings: the input of block 0, ``[T, D]``."""
        ids = np.asarray(token_ids)
        return self.tensors["wte.weight"][ids] + self.tensors["wpe.weight"][: len(ids)]

    def reference_block(self, i: int, x: np.ndarray) -> np.ndarray:
        w = {k: torch.from_numpy(v) for k, v in self.block_weights(i).items()}
        with torch.no_grad():
            return ref.gpt2_block(torch.from_numpy(x), w, self.n_head, self.eps).numpy()

    def logits(self, hidden: np.ndarray) -> np.ndarray:
        """Final LayerNorm and the tied output embedding, in float32 PyTorch."""
        with torch.no_grad():
            h = ref.layernorm(
                torch.from_numpy(np.asarray(hidden, dtype=np.float32)),
                torch.from_numpy(self.tensors["ln_f.weight"]),
                torch.from_numpy(self.tensors["ln_f.bias"]),
                self.eps,
            )
            return (h @ torch.from_numpy(self.tensors["wte.weight"]).T).numpy()

    def forward(self, token_ids, block_fns: dict[int, BlockFn] | None = None):
        """Run the model and return ``(logits [T, vocab], hidden states per block)``.

        ``block_fns[i]``, when given, replaces block ``i``. It receives and
        returns float32 ``[T, D]`` arrays; everything else runs in float32 PyTorch.
        """
        block_fns = block_fns or {}
        x = self.embed(token_ids)
        hidden = []
        for i in range(self.n_layer):
            x = block_fns[i](x) if i in block_fns else self.reference_block(i, x)
            x = np.asarray(x, dtype=np.float32)
            hidden.append(x)
        return self.logits(x), hidden
