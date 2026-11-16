# Compile report

- shapes: GPT-2 124M (d_model 768, 12 heads), 128 tokens
- neuronx-cc 2.28.7631.0+8f7b0cf5, nki 0.7.0+37713544100.gfede04bb
- every row compiled to a NEFF and its compiled instructions were executed on the compiler's simulator
- instruction counts are as printed by neuronx-cc (`instruction_stats.txt`); they are not latency

| target | dtype | kernel | MATMUL | ACTIVATE | TENSOR_SCALAR | TENSOR_TENSOR | TENSOR_REDUCE | COPY | all instructions | max rel. diff vs Python sim |
|---|---|---|---|---|---|---|---|---|---|---|
| trn2 | float32 | layernorm_T | 6 | 3 | 7 | 0 | 1 | 0 | 65 | 5.0e-07 |
| trn2 | float32 | linear_T (QKV, 768->2304) | 216 | 18 | 0 | 0 | 0 | 0 | 655 | 2.6e-07 |
| trn2 | float32 | causal_attention_T (12 heads) | 72 | 24 | 12 | 0 | 12 | 36 | 346 | 3.7e-07 |
| trn2 | float32 | linear_T (proj, 768->768) | 72 | 6 | 0 | 0 | 0 | 0 | 230 | 2.4e-07 |
| trn2 | float32 | residual_add_T | 6 | 0 | 0 | 6 | 0 | 0 | 55 | 0.0e+00 |
| trn2 | float32 | mlp_T (768->3072->768) | 576 | 30 | 0 | 0 | 0 | 0 | 1607 | 6.1e-07 |
| trn2 | float32 | &nbsp;&nbsp;unfused: linear_gelu_T (768->3072) | 288 | 24 | 0 | 0 | 0 | 0 | 866 | 2.4e-06 |
| trn2 | float32 | &nbsp;&nbsp;unfused: linear_T (3072->768) | 288 | 6 | 0 | 0 | 0 | 0 | 806 | 2.6e-07 |
| trn2 | bfloat16 | layernorm_T | 6 | 3 | 7 | 0 | 1 | 0 | 65 | 1.6e-03 |
| trn2 | bfloat16 | linear_T (QKV, 768->2304) | 108 | 18 | 0 | 0 | 0 | 0 | 439 | 3.0e-05 |
| trn2 | bfloat16 | causal_attention_T (12 heads) | 48 | 24 | 12 | 0 | 12 | 12 | 300 | 5.8e-04 |
| trn2 | bfloat16 | linear_T (proj, 768->768) | 36 | 6 | 0 | 0 | 0 | 0 | 158 | 7.8e-06 |
| trn2 | bfloat16 | residual_add_T | 6 | 0 | 0 | 6 | 0 | 0 | 55 | 0.0e+00 |
| trn2 | bfloat16 | mlp_T (768->3072->768) | 288 | 30 | 0 | 0 | 0 | 0 | 1010 | 1.2e-04 |
| trn2 | bfloat16 | &nbsp;&nbsp;unfused: linear_gelu_T (768->3072) | 144 | 24 | 0 | 0 | 0 | 0 | 578 | 9.8e-04 |
| trn2 | bfloat16 | &nbsp;&nbsp;unfused: linear_T (3072->768) | 144 | 6 | 0 | 0 | 0 | 0 | 518 | 2.3e-06 |
| trn3 | float32 | layernorm_T | 6 | 3 | 7 | 0 | 1 | 0 | 65 | 5.0e-07 |
| trn3 | float32 | linear_T (QKV, 768->2304) | 216 | 18 | 0 | 0 | 0 | 0 | 655 | 2.6e-07 |
| trn3 | float32 | causal_attention_T (12 heads) | 72 | 24 | 12 | 0 | 12 | 36 | 344 | 3.7e-07 |
| trn3 | float32 | linear_T (proj, 768->768) | 72 | 6 | 0 | 0 | 0 | 0 | 230 | 2.4e-07 |
| trn3 | float32 | residual_add_T | 6 | 0 | 0 | 6 | 0 | 0 | 55 | 0.0e+00 |
| trn3 | float32 | mlp_T (768->3072->768) | 576 | 30 | 0 | 0 | 0 | 0 | 1610 | 6.1e-07 |
| trn3 | float32 | &nbsp;&nbsp;unfused: linear_gelu_T (768->3072) | 288 | 24 | 0 | 0 | 0 | 0 | 865 | 2.4e-06 |
| trn3 | float32 | &nbsp;&nbsp;unfused: linear_T (3072->768) | 288 | 6 | 0 | 0 | 0 | 0 | 806 | 2.6e-07 |
| trn3 | bfloat16 | layernorm_T | 6 | 3 | 7 | 0 | 1 | 0 | 65 | 1.6e-03 |
| trn3 | bfloat16 | linear_T (QKV, 768->2304) | 108 | 18 | 0 | 0 | 0 | 0 | 439 | 3.0e-05 |
| trn3 | bfloat16 | causal_attention_T (12 heads) | 48 | 24 | 12 | 0 | 12 | 12 | 300 | 5.8e-04 |
| trn3 | bfloat16 | linear_T (proj, 768->768) | 36 | 6 | 0 | 0 | 0 | 0 | 158 | 7.8e-06 |
| trn3 | bfloat16 | residual_add_T | 6 | 0 | 0 | 6 | 0 | 0 | 55 | 0.0e+00 |
| trn3 | bfloat16 | mlp_T (768->3072->768) | 288 | 30 | 0 | 0 | 0 | 0 | 1010 | 1.2e-04 |
| trn3 | bfloat16 | &nbsp;&nbsp;unfused: linear_gelu_T (768->3072) | 144 | 24 | 0 | 0 | 0 | 0 | 577 | 9.8e-04 |
| trn3 | bfloat16 | &nbsp;&nbsp;unfused: linear_T (3072->768) | 144 | 6 | 0 | 0 | 0 | 0 | 518 | 2.3e-06 |
