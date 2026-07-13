# Q8_0 GCN Repack benches (single GPU only, dense 4B model)

These a structured bench results for this commit stage. 
These results are useful to drive development but are meant to be useful for single-GPU only and qwen 4B model.

>>> REPACK_Q8_0=ON
[tuner] Using existing: /home/iacopo/Desktop/TheRock/.tmpvenv-vega/lib/python3.12/site-packages/_rocm_sdk_devel/share/rccl/tuner/rccl_tuner_gfx906.csv
=== Benchmark ===
Model: Qwen3-4B-Instruct-2507-Q8_0.gguf

>>> 
ggml_cuda_init: found 1 ROCm devices (Total VRAM: 32069 MiB):
  Device 0: AMD Instinct MI60 / MI50, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64, VRAM: 32069 MiB
llama-bench: benchmark 1/2: starting
llama-bench: benchmark 1/2: warmup prompt run
llama-bench: benchmark 1/2: prompt run 1/1
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |  fa | mmap |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --: | ---: | --------------: | -------------------: |
| qwen3 4B Q8_0                  |   3.98 GiB |     4.02 B | ROCm       |  99 |      24 |     2048 |   1 |    0 |          pp2048 |       1404.28 ± 0.00 |
llama-bench: benchmark 2/2: starting
llama-bench: benchmark 2/2: warmup generation run
llama-bench: benchmark 2/2: generation run 1/1
| qwen3 4B Q8_0                  |   3.98 GiB |     4.02 B | ROCm       |  99 |      24 |     2048 |   1 |    0 |           tg128 |        109.49 ± 0.00 |

build: 257287dff (9484)

Output saved to: bench_results.md



>>> REPACK_Q8_0=OFF

[tuner] Using existing: /home/iacopo/Desktop/TheRock/.tmpvenv-vega/lib/python3.12/site-packages/_rocm_sdk_devel/share/rccl/tuner/rccl_tuner_gfx906.csv
=== Benchmark ===
Model: Qwen3-4B-Instruct-2507-Q8_0.gguf

>>> 
ggml_cuda_init: found 1 ROCm devices (Total VRAM: 32069 MiB):
  Device 0: AMD Instinct MI60 / MI50, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64, VRAM: 32069 MiB
llama-bench: benchmark 1/2: starting
llama-bench: benchmark 1/2: warmup prompt run
[KSHARD-DIAG] env=0 gcn=1 small_k=0 has_fusion=1 has_ids=0 nchannels_dst=1 nsamples_dst=1 kb=1 type=8 ncols_x=2560 nrows_x=9728 bpr=80 bpi=16 vdr=2 qi=8 warp=64
llama-bench: benchmark 1/2: prompt run 1/1
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |  fa | mmap |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --: | ---: | --------------: | -------------------: |
| qwen3 4B Q8_0                  |   3.98 GiB |     4.02 B | ROCm       |  99 |      24 |     2048 |   1 |    0 |          pp2048 |       1294.27 ± 0.00 |
llama-bench: benchmark 2/2: starting
llama-bench: benchmark 2/2: warmup generation run
llama-bench: benchmark 2/2: generation run 1/1
| qwen3 4B Q8_0                  |   3.98 GiB |     4.02 B | ROCm       |  99 |      24 |     2048 |   1 |    0 |           tg128 |        105.42 ± 0.00 |

build: 257287dff (9484)

Output saved to: bench_results.md



================================================================================
KERNEL DISCOVERY REPORT - REPACK Q8 ENABLED - prompt processing only
================================================================================
Total GPU time: 2,867.18 ms | Kernels: 18 | Hot (>1%): 4

#    Kernel                                                Calls   Time(ms)      %
--------------------------------------------------------------------------------
1   * mmq_gemm_q8_0_repacked                                498    2109.91  73.6%
2   * flash_attn_tile                                        72     547.09  19.1%
3     flash_attn_combine_results                             72      68.18   2.4%
4     quantize_mmq_q8_1                                     498      39.61   1.4%
5     rms_norm_f32                                          144      24.31   0.8%
6     unary_gated_op_kernel                                  72      23.79   0.8%
7     rope_neox                                              72      16.03   0.6%
8     rms_norm_f32                                          146      14.13   0.5%
9     k_bin_bcast                                           144      12.61   0.4%
10    rope_neox                                              72       4.02   0.1%
11    k_set_rows                                             72       4.00   0.1%
12    __amd_rocclr_fillBufferAligned                         37       1.29   0.0%
13    mul_mat_vec_q8_0_repacked                               6       1.23   0.0%
14    __amd_rocclr_copyBuffer                               153       0.53   0.0%
15    flash_attn_mask_to_KV_max                              72       0.32   0.0%
16    mul_mat_vec_q8_0_repacked                               2       0.08   0.0%
17    quantize_q8_1                                           8       0.04   0.0%
18    k_get_rows_float                                        4       0.02   0.0%
--------------------------------------------------------------------------------

================================================================================
KERNEL DISCOVERY REPORT - DEFAULT - prompt processing only
================================================================================
Total GPU time: 3,114.89 ms | Kernels: 19 | Hot (>1%): 5

#    Kernel                                                Calls   Time(ms)      %
--------------------------------------------------------------------------------
1   * mul_mat_q<Q8>                                         356    1496.53  48.0%
2   * mul_mat_q<Q8>                                         142     847.99  27.2%
3   * flash_attn_tile                                        72     557.54  17.9%
4     flash_attn_combine_results                             72      68.16   2.2%
5     quantize_mmq_q8_1                                     498      40.03   1.3%
6     rms_norm_f32                                          144      25.44   0.8%
7     unary_gated_op_kernel                                  70      24.36   0.8%
8     rope_neox                                              72      16.78   0.5%
9     rms_norm_f32                                          146      14.27   0.5%
10    k_bin_bcast                                           142      12.43   0.4%
11    rope_neox                                              72       4.31   0.1%
12    k_set_rows                                             72       3.75   0.1%
13    mul_mat_vec_q<Q8>                                       2       1.22   0.0%
14    __amd_rocclr_fillBufferAligned                          1       0.89   0.0%
15    __amd_rocclr_copyBuffer                               153       0.55   0.0%
16    flash_attn_mask_to_KV_max                              72       0.34   0.0%
17    mul_mat_vec_q<Q8>                                       4       0.25   0.0%
18    quantize_q8_1                                           6       0.03   0.0%
19    k_get_rows_float                                        4       0.02   0.0%
--------------------------------------------------------------------------------


