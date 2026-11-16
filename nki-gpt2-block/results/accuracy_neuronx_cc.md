# Accuracy report

- model: GPT-2 124M (openai-community/gpt2)
- blocks: 12
- prompt tokens: 205
- simulator backend: neuronx-cc
- target: trn2
- nki: 0.7.0+37713544100.gfede04bb
- torch: 2.15.0+cu132

## Part A: per-kernel error inside a block

Maximum absolute error divided by the largest reference magnitude, against the float32 PyTorch block with the original weights. The block input is the float32 model's real hidden state for the evaluation prompt. Errors accumulate from left to right because each kernel consumes the previous kernel's output. The PyTorch bfloat16 rows run the same block with PyTorch's own CPU bfloat16 ops.

| block | run | ln_1 | qkv | attention | attn_proj | resid_mid | ln_2 | mlp | resid_out |
|---|---|---|---|---|---|---|---|---|---|
| 0 | NKI float32 | 6.3e-07 | 9.3e-07 | 6.7e-07 | 6.9e-07 | 6.8e-07 | 4.5e-07 | 3.2e-07 | 3.2e-07 |
| 0 | NKI bfloat16 | 4.4e-03 | 3.8e-03 | 3.9e-03 | 3.7e-03 | 5.0e-03 | 3.8e-03 | 2.6e-03 | 2.8e-03 |
| 0 | PyTorch bfloat16 (baseline) | 7.2e-03 | 5.3e-03 | 5.1e-03 | 4.6e-03 | 7.1e-03 | 4.4e-03 | 2.6e-03 | 2.8e-03 |
| 5 | NKI float32 | 6.8e-07 | 9.9e-07 | 1.5e-06 | 1.6e-06 | 6.1e-09 | 6.0e-07 | 1.7e-06 | 1.6e-08 |
| 5 | NKI bfloat16 | 4.2e-03 | 5.4e-03 | 9.2e-03 | 1.2e-02 | 2.3e-03 | 1.8e-03 | 7.0e-03 | 8.8e-04 |
| 5 | PyTorch bfloat16 (baseline) | 6.6e-03 | 7.1e-03 | 1.3e-02 | 1.8e-02 | 2.3e-03 | 4.3e-03 | 7.9e-03 | 8.8e-04 |
| 11 | NKI float32 | 5.4e-07 | 5.6e-07 | 4.2e-07 | 2.8e-07 | 1.6e-06 | 1.1e-06 | 1.7e-06 | 1.8e-06 |
| 11 | NKI bfloat16 | 3.5e-03 | 6.9e-03 | 6.9e-03 | 3.2e-03 | 2.7e-02 | 1.5e-02 | 1.8e-02 | 3.2e-02 |
| 11 | PyTorch bfloat16 (baseline) | 4.0e-03 | 6.9e-03 | 6.9e-03 | 3.2e-03 | 2.7e-02 | 1.5e-02 | 1.8e-02 | 3.2e-02 |

## Part B: all 12 blocks on NKI kernels

Next-token distributions over every prompt position, against the float32 PyTorch model. Embeddings, the final LayerNorm and the output projection run in float32 PyTorch in every row.

| run | top-1 agreement | max abs logit diff | mean KL | max KL |
|---|---|---|---|---|
| NKI kernels, float32 | 205/205 (100.0%) | 3.81e-04 | 2.23e-08 | 6.04e-07 |
| NKI kernels, bfloat16 | 203/205 (99.0%) | 4.31e+00 | 2.05e-04 | 5.44e-03 |
| PyTorch CPU, bfloat16 (baseline) | 203/205 (99.0%) | 2.52e+00 | 2.55e-04 | 2.52e-03 |
