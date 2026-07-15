# Native `mul_mat_q` (Q8_0) vs Repack `mmq_gemm_q8_0_repacked` — Kernel Execution & ISA Comparison

Generated: 2026-07-15

## Benchmark context (fact-checked from rocprofv3 + llama-bench runs)

| Field | Value |
|---|---|
| Tool | rocprofv3 --kernel-trace --stats (graphs disabled: GGML_CUDA_DISABLE_GRAPHS=1) |
| Device | AMD Instinct MI50/MI60, gfx906, Wave Size 64, 32069 MiB VRAM |
| Model | Qwen_Qwen3.5-4B-Q8_0.gguf (4.29 GiB, 4.33 B params, Q8_0 weights) |
| Test | llama-bench pp2048, ngl 99, fa 1, n_ubatch 2048, 1 prompt |
| Native prompt throughput | 1290.34 t/s (pp2048) |
| Repack prompt throughput | 1458.86 t/s (pp2048) |
| Native total GPU kernel time | 3144.98 ms (28 kernels) |
| Repack total GPU kernel time | 2756.06 ms (27 kernels) |

---

## 1. Side-by-side execution — kernels launched

### 1.1 GEMM kernel that performs the Q8_0 weight x F32 activation multiply

| Attribute | Native | Native (2nd tile) | Repack |
|---|---|---|---|
| Kernel | `mul_mat_q<(ggml_type)8, 64, false>` | `mul_mat_q<(ggml_type)8, 96, false>` | `mmq_gemm_q8_0_repacked<false, 2, 4>` |
| Source .s (gfx906) | `mmq-instance-q8_0-hip-amdgcn-amd-amdhsa-gfx906.s` | `mmq-instance-q8_0-hip-amdgcn-amd-amdhsa-gfx906.s` | `repack-gcn-hip-amdgcn-amd-amdhsa-gfx906.s` |
| Calls | 272 | 128 | 400 |
| Total GPU time | 1585.37 ms | 737.35 ms | 1936.28 ms |
| % of total GPU time | 50.41% | 23.45% | 70.26% |
| Workgroup size | 64 x 8 x 1 (512 threads) | 64 x 8 x 1 (512 threads) | 64 x 4 x 1 (256 threads) |
| Group (LDS) segment size | 48000 B | 52224 B | 32256 B |
| Private segment size | 28 B | 0 B | 0 B |

Native dispatch grids observed (Grid_Size_X x Grid_Size_Y x Grid_Size_Z):

- `mul_mat_q<(ggml_type)8, 64, false>`: 512x256x1, 4096x256x1, 4608x256x1, 2048x256x1 (272 calls)
- `mul_mat_q<(ggml_type)8, 96, false>`: 1280x176x1 (128 calls)
- `mmq_gemm_q8_0_repacked<false, 2, 4>`: 1024x64x1, 2560x64x1, 4096x64x1, 8192x64x1, 9216x64x1 (400 calls)

### 1.2 Complete kernel list per run (rank, short name, calls, time ms, % of GPU time)

**Native run** (total 3144.98 ms):

| # | Kernel | Calls | Time (ms) | % |
|---|---|---:|---:|---:|
| 1 | mul_mat_q<Q8> (x64) | 272 | 1585.37 | 50.41 |
| 2 | mul_mat_q<Q8> (x96) | 128 | 737.35 | 23.45 |
| 3 | gated_delta_net_chunked_cuda | 48 | 324.59 | 10.32 |
| 4 | concat_f32_non_cont | 48 | 175.72 | 5.59 |
| 5 | flash_attn_tile | 16 | 130.23 | 4.14 |
| 6 | quantize_mmq_q8_1 | 400 | 38.84 | 1.24 |
| 7 | unary_gated_op_kernel (silu) | 112 | 28.17 | 0.9 |
| 8 | ssm_conv_long_token_f32 | 48 | 24.07 | 0.77 |
| 9 | rocBLAS_gemm<MT64x64x16_ISA906> | 96 | 19.52 | 0.62 |
| 10 | rms_norm_f32<256> | 80 | 17.23 | 0.55 |
| 11 | rms_norm_f32<1024> | 130 | 12.87 | 0.41 |
| 12 | k_bin_bcast (add) | 176 | 11.78 | 0.37 |
| 13 | flash_attn_combine_results | 16 | 9.02 | 0.29 |
| 14 | l2_norm_f32 | 96 | 8.49 | 0.27 |
| 15 | cpy_scalar | 64 | 7.81 | 0.25 |
| 16 | rope_multi | 32 | 3.93 | 0.12 |
| 17 | unary_gated_op_kernel (sigmoid) | 16 | 2.36 | 0.07 |
| 18 | k_set_rows | 32 | 1.99 | 0.06 |
| 19 | mul_mat_vec_q<Q8> | 2 | 1.97 | 0.06 |
| 20 | __amd_rocclr_copyBuffer | 216 | 1.01 | 0.03 |
| 21 | k_get_rows_float | 98 | 0.84 | 0.03 |
| 22 | scale_f32 | 96 | 0.65 | 0.02 |
| 23 | __amd_rocclr_fillBufferAligned | 2 | 0.44 | 0.01 |
| 24 | k_bin_bcast (mul) | 48 | 0.21 | 0.01 |
| 25 | unary_op_kernel (sigmoid) | 48 | 0.21 | 0.01 |
| 26 | unary_op_kernel (softplus) | 48 | 0.21 | 0.01 |
| 27 | flash_attn_mask_to_KV_max | 16 | 0.08 | 0.0 |
| 28 | quantize_q8_1<32> | 2 | 0.01 | 0.0 |

**Repack run** (total 2756.06 ms):

| # | Kernel | Calls | Time (ms) | % |
|---|---|---:|---:|---:|
| 1 | mmq_gemm_q8_0_repacked | 400 | 1936.28 | 70.26 |
| 2 | gated_delta_net_chunked_cuda | 48 | 324.04 | 11.76 |
| 3 | concat_f32_non_cont | 48 | 174.03 | 6.31 |
| 4 | flash_attn_tile | 16 | 129.24 | 4.69 |
| 5 | quantize_mmq_q8_1 | 400 | 39.25 | 1.42 |
| 6 | unary_gated_op_kernel (silu) | 112 | 28.2 | 1.02 |
| 7 | ssm_conv_long_token_f32 | 48 | 24.02 | 0.87 |
| 8 | rocBLAS_gemm<MT64x64x16_ISA906> | 96 | 19.13 | 0.69 |
| 9 | rms_norm_f32<256> | 80 | 17.23 | 0.63 |
| 10 | rms_norm_f32<1024> | 130 | 13.4 | 0.49 |
| 11 | k_bin_bcast (add) | 176 | 12.11 | 0.44 |
| 12 | flash_attn_combine_results | 16 | 9.22 | 0.33 |
| 13 | l2_norm_f32 | 96 | 8.47 | 0.31 |
| 14 | cpy_scalar | 64 | 7.57 | 0.27 |
| 15 | rope_multi | 32 | 3.83 | 0.14 |
| 16 | unary_gated_op_kernel (sigmoid) | 16 | 2.35 | 0.09 |
| 17 | k_set_rows | 32 | 2.07 | 0.08 |
| 18 | mul_mat_vec_q8_0_repacked | 2 | 1.78 | 0.06 |
| 19 | __amd_rocclr_copyBuffer | 216 | 0.97 | 0.04 |
| 20 | k_get_rows_float | 98 | 0.86 | 0.03 |
| 21 | scale_f32 | 96 | 0.67 | 0.02 |
| 22 | __amd_rocclr_fillBufferAligned | 35 | 0.6 | 0.02 |
| 23 | k_bin_bcast (mul) | 48 | 0.23 | 0.01 |
| 24 | unary_op_kernel (sigmoid) | 48 | 0.21 | 0.01 |
| 25 | unary_op_kernel (softplus) | 48 | 0.21 | 0.01 |
| 26 | flash_attn_mask_to_KV_max | 16 | 0.08 | 0.0 |
| 27 | quantize_q8_1<32> | 2 | 0.01 | 0.0 |

### 1.3 Kernels common to both runs (non-GEMM)

The following kernels appeared in both runs with the same call counts and within ~1% time of each other: gated_delta_net_chunked_cuda (48, ~324 ms), concat_f32_non_cont (48, ~175 ms), flash_attn_tile (16, ~129-130 ms), quantize_mmq_q8_1 (400, ~39 ms), unary_gated_op_kernel (silu) (112, ~28 ms), ssm_conv_long_token_f32 (48, ~24 ms), rocBLAS_gemm<MT64x64x16_ISA906> (96, ~19-20 ms), rms_norm_f32<256> (80, ~17 ms), rms_norm_f32<1024> (130, ~13-13 ms), k_bin_bcast (add) (176, ~12 ms), flash_attn_combine_results (16, ~9 ms), l2_norm_f32 (96, ~8 ms), cpy_scalar (64, ~8 ms), rope_multi (32, ~4 ms), unary_gated_op_kernel (sigmoid) (16, ~2 ms), k_set_rows (32, ~2 ms), copyBuffer (216, ~1 ms), k_get_rows_float (98, ~1 ms), scale_f32 (96, ~1 ms), fillBufferAligned (2-35), k_bin_bcast (mul) (48, ~0.2 ms), unary_op_kernel sigmoid/softplus (48 each, ~0.2 ms), flash_attn_mask_to_KV_max (16, ~0.08 ms), quantize_q8_1<32> (2, ~0.01 ms).

Native-only GEMM-path kernels: `mul_mat_q<Q8>` (x64 + x96), `mul_mat_vec_q<Q8>` (2 calls). Repack-only GEMM-path kernel: `mmq_gemm_q8_0_repacked` (400 calls), `mul_mat_vec_q8_0_repacked` (2 calls).

---

## 2. Side-by-side ISA

All ISA below was extracted from the `*-hip-amdgcn-amd-amdhsa-gfx906.s` device assembly produced by the ROCm clang (`-save-temps --offload-arch=gfx906`) used to build `ggml-hip`. Instruction counts are computed directly from those `.s` files (every non-directive, non-label, non-comment line is tokenized to its opcode).

### 2.1 ISA metrics comparison

| Metric | Native x64 | Native x96 | Repack |
|---|---:|---:|---:|
| Source file | `mmq-instance-q8_0-hip-amdgcn-amd-amdhsa-gfx906.s` | `mmq-instance-q8_0-hip-amdgcn-amd-amdhsa-gfx906.s` | `repack-gcn-hip-amdgcn-amd-amdhsa-gfx906.s` |
| Mangled symbol | `_ZL9mul_mat_qIL9ggml_type8ELi64ELb0E...` | `_ZL9mul_mat_qIL9ggml_type8ELi96ELb0E...` | `_ZL22mmq_gemm_q8_0_repackedILb0ELi2ELi4E...` |
| VGPRs (num_vgpr) | 128 | 98 | 83 |
| SGPRs (numbered_sgpr) | 72 | 76 | 31 |
| AGPRs (num_agpr) | 0 | 0 | 0 |
| Total instructions (body) | 3431 | 4580 | 1165 |
| Vector-class instr | 1897 | 2671 | 788 |
| Scalar-class instr | 1011 | 1224 | 279 |
| LDS-class instr (ds_*) | 376 | 514 | 59 |
| Global-class instr (buffer_/global_) | 146 | 170 | 38 |
| dp4a (v_dot4_i32_i8) | 768 | 1152 | 256 |
| FMA (v_fma/v_fmac/v_fma_mix) | 96 | 144 | 64 |
| dp4a / FMA ratio | 8.00 | 8.00 | 4.00 |
| s_waitcnt total | 266 | 407 | 80 |
|   of which vmcnt(0) | 16 | 22 | 2 |
|   of which lgkmcnt | 220 | 353 | 75 |
| ds_read_* | 324 | 456 | 50 |
| ds_write_* | 52 | 58 | 9 |
| buffer/global load | 94 | 100 | 6 |
| buffer/global store | 52 | 70 | 32 |
| v_mov_b32 | 165 | 221 | 110 |

### 2.2 Top-15 instructions per kernel (count)

**mul_mat_q<(ggml_type)8, 64, false>**

| Instruction | Count |
|---|---:|
| `v_dot4_i32_i8` | 768 |
| `v_add_u32_e32` | 299 |
| `s_waitcnt` | 266 |
| `ds_read2_b32` | 240 |
| `v_mov_b32_e32` | 165 |
| `v_mul_f32_e32` | 102 |
| `v_cvt_f32_i32_e32` | 96 |
| `v_fmac_f32_e32` | 96 |
| `s_mul_i32` | 88 |
| `ds_read_b32` | 84 |
| `global_load_dword` | 80 |
| `s_add_i32` | 75 |
| `s_mul_hi_u32` | 55 |
| `v_add_co_u32_e32` | 49 |
| `v_addc_co_u32_e32` | 49 |

**mul_mat_q<(ggml_type)8, 96, false>**

| Instruction | Count |
|---|---:|
| `v_dot4_i32_i8` | 1152 |
| `s_waitcnt` | 407 |
| `v_add_u32_e32` | 406 |
| `ds_read2_b32` | 336 |
| `v_mov_b32_e32` | 221 |
| `v_mul_f32_e32` | 150 |
| `v_cvt_f32_i32_e32` | 144 |
| `v_fmac_f32_e32` | 144 |
| `ds_read_b32` | 120 |
| `global_load_dword` | 92 |
| `s_mul_i32` | 89 |
| `s_add_i32` | 77 |
| `global_store_dword` | 70 |
| `v_add_co_u32_e32` | 63 |
| `v_addc_co_u32_e32` | 63 |

**mmq_gemm_q8_0_repacked<false, 2, 4>**

| Instruction | Count |
|---|---:|
| `v_dot4_i32_i8` | 256 |
| `v_mov_b32_e32` | 110 |
| `s_waitcnt` | 80 |
| `s_cbranch_execz` | 53 |
| `v_lshlrev_b64` | 51 |
| `v_cmp_gt_u32_e32` | 49 |
| `v_add_co_u32_e32` | 48 |
| `v_addc_co_u32_e32` | 48 |
| `v_add_u32_e32` | 46 |
| `s_or_b64` | 37 |
| `s_and_saveexec_b64` | 36 |
| `v_mad_u64_u32` | 34 |
| `ds_read_b128` | 32 |
| `v_cvt_f32_i32_e32` | 32 |
| `v_fma_mix_f32` | 32 |

---

### 2.3 Native `mul_mat_q<(ggml_type)8, 64, false>` ISA body

Source: `mmq-instance-q8_0-hip-amdgcn-amd-amdhsa-gfx906.s`  |  Mangled: `mul_mat_q<(ggml_type)8, 64, false>`  |  Body lines: 3657  |  Total instructions: 3431

```asm
_ZL9mul_mat_qIL9ggml_type8ELi64ELb0EEvPKcPKiS4_S4_PfS5_15HIP_vector_typeIjLj3EEiiiiiS7_S7_iiiS7_S7_iiiS7_b: ; @_ZL9mul_mat_qIL9ggml_type8ELi64ELb0EEvPKcPKiS4_S4_PfS5_15HIP_vector_typeIjLj3EEiiiiiS7_S7_iiiS7_S7_iiiS7_b
; %bb.0:
	s_load_dwordx8 s[20:27], s[4:5], 0x30
	s_load_dwordx2 s[10:11], s[4:5], 0x50
	s_load_dwordx8 s[36:43], s[4:5], 0x5c
	s_load_dwordx2 s[34:35], s[4:5], 0xa0
	s_load_dwordx8 s[44:51], s[4:5], 0x80
	s_mov_b64 s[70:71], s[2:3]
	s_mov_b64 s[68:69], s[0:1]
	v_mov_b32_e32 v75, v1
	s_add_u32 s68, s68, s9
	v_lshl_add_u32 v62, v75, 6, v0
	s_addc_u32 s69, s69, 0
	v_cmp_gt_u32_e64 s[0:1], 64, v62
	s_and_saveexec_b64 s[2:3], s[0:1]
; %bb.1:                                ; %.lr.ph
	v_lshl_add_u32 v2, v62, 2, 0
	ds_write_b32 v2, v62
; %bb.2:                                ; %.critedge
	s_or_b64 exec, exec, s[2:3]
	s_load_dwordx8 s[12:19], s[4:5], 0x0
	s_load_dwordx4 s[28:31], s[4:5], 0x20
	s_waitcnt lgkmcnt(0)
	s_bitcmp1_b32 s35, 0
	s_cselect_b64 s[2:3], -1, 0
	s_and_b64 vcc, exec, s[2:3]
	s_barrier
	s_cbranch_vccnz .LBB28_8
; %bb.3:
	s_mul_hi_u32 s2, s36, s8
	s_add_i32 s2, s8, s2
	s_lshr_b32 s57, s2, s37
	s_mul_i32 s2, s57, s38
	s_sub_i32 s8, s8, s2
	s_lshl_b32 s7, s7, 6
	s_cmp_lg_u64 s[16:17], 0
	s_mov_b64 s[2:3], 0
	s_cbranch_scc0 .LBB28_9
; %bb.4:
	s_ashr_i32 s9, s8, 31
	s_lshl_b64 s[52:53], s[8:9], 2
	s_add_u32 s54, s18, s52
	s_addc_u32 s55, s19, s53
	s_load_dwordx2 s[52:53], s[54:55], 0x0
	s_mov_b64 s[54:55], 0
	s_mov_b32 s9, 0
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s33, s53, s52
	s_cmp_lt_i32 s7, s33
	s_mov_b32 s53, 0
	s_cbranch_scc0 .LBB28_17
; %bb.5:                                ; %.preheader
	s_and_saveexec_b64 s[54:55], s[0:1]
	s_cbranch_execz .LBB28_7
; %bb.6:                                ; %.lr.ph803
	v_or_b32_e32 v2, s7, v0
	v_add_u32_e32 v2, s52, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_mov_b32_e32 v4, s17
	v_add_co_u32_e32 v2, vcc, s16, v2
	v_addc_co_u32_e32 v3, vcc, v4, v3, vcc
	global_load_dword v2, v[2:3], off
	v_lshl_add_u32 v3, v62, 2, 0
	s_waitcnt vmcnt(0)
	ds_write_b32 v3, v2
.LBB28_7:                               ; %.critedge463
	s_or_b64 exec, exec, s[54:55]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_mov_b32 s53, s52
	s_mov_b32 s35, 0
	s_cbranch_execnz .LBB28_10
	s_branch .LBB28_18
.LBB28_8:
	s_mov_b64 s[2:3], 0
                                        ; implicit-def: $vgpr18
                                        ; implicit-def: $vgpr19
                                        ; implicit-def: $sgpr8_sgpr9
                                        ; implicit-def: $vgpr2
	s_cbranch_execnz .LBB28_37
	s_branch .LBB28_107
.LBB28_9:
	s_mul_i32 s9, s57, s48
	s_mul_i32 s33, s8, s40
	s_add_i32 s9, s33, s9
	s_mul_i32 s33, s57, s49
	s_mul_i32 s52, s27, s7
	s_mul_i32 s35, s8, s41
	s_add_i32 s33, s33, s52
	s_add_i32 s35, s33, s35
	s_mov_b32 s53, 0
	s_mov_b32 s33, s24
	s_cbranch_execz .LBB28_18
.LBB28_10:
	s_lshl_b32 s54, s6, 7
	s_cmp_lt_i32 s22, 1
	s_cbranch_scc1 .LBB28_19
; %bb.11:                               ; %.lr.ph.i
	s_add_i32 s2, s53, s7
	s_mul_i32 s2, s2, 36
	s_add_i32 s2, s2, s9
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 2
	v_lshrrev_b32_e32 v3, 4, v0
	s_add_u32 s55, s14, s2
	s_mul_hi_u32 s2, s8, s10
	v_lshl_add_u32 v3, v75, 2, v3
	s_movk_i32 s52, 0x104
	s_addc_u32 s56, s15, s3
	s_add_i32 s2, s8, s2
	v_and_b32_e32 v2, 15, v0
	v_mul_lo_u32 v36, s25, v3
	v_mad_u32_u24 v3, v3, s52, 0
	s_lshr_b32 s2, s2, s11
	v_lshl_add_u32 v37, v2, 2, v3
	v_lshrrev_b32_e32 v2, 3, v0
	s_mul_i32 s8, s2, s39
	s_mul_hi_u32 s2, s57, s42
	v_lshl_add_u32 v2, v75, 3, v2
	s_add_i32 s57, s57, s2
	v_and_b32_e32 v35, 7, v0
	v_mul_lo_u32 v38, s25, v2
	v_and_b32_e32 v3, 0x1ffc, v2
	v_lshlrev_b32_e32 v6, 5, v2
	v_add_u32_e32 v2, 64, v2
	s_lshr_b32 s2, s57, s43
	s_lshl_b32 s53, s25, 5
	v_lshlrev_b32_e32 v5, 2, v35
	v_and_b32_e32 v7, 0x3ffc, v2
	v_add_u32_e32 v9, 64, v0
	s_mul_i32 s3, s25, s54
	s_mul_i32 s9, s2, s47
	v_add3_u32 v3, 0, v3, v5
	v_add3_u32 v5, 0, v7, v5
	s_mul_i32 s2, s26, 36
	v_lshlrev_b32_e32 v7, 5, v0
	v_mul_u32_u24_e32 v8, 36, v75
	v_and_b32_e32 v9, 0x3fc, v9
	v_and_b32_e32 v11, 0x1fc, v0
	v_add_u32_e32 v50, s53, v36
	v_lshlrev_b32_e32 v4, 1, v35
	v_lshlrev_b32_e32 v2, 5, v2
	s_add_i32 s58, s9, s3
	s_ashr_i32 s3, s2, 31
	v_lshl_add_u32 v8, v8, 2, 0
	v_mad_u32_u24 v10, v0, s52, 0
	v_add3_u32 v9, v7, v9, 0
	v_add3_u32 v7, v7, v11, 0
	v_add_u32_e32 v51, s53, v50
	v_bfe_u32 v34, v0, 3, 1
	s_mov_b32 s57, 0
	s_add_i32 s58, s58, s8
	s_lshl_b64 s[8:9], s[2:3], 2
	v_add_u32_e32 v39, 0x110, v8
	v_add_u32_e32 v40, 0x2900, v10
	v_add_u32_e32 v41, 0x100, v8
	v_add_u32_e32 v42, v3, v6
	v_add_u32_e32 v43, 0xb300, v9
	v_add_u32_e32 v44, v5, v2
	v_add_u32_e32 v45, 0xab00, v7
	v_add_u32_e32 v46, 0x2980, v10
	v_add_u32_e32 v47, 0xb310, v9
	v_add_u32_e32 v48, 0xab10, v7
	v_lshlrev_b32_e32 v49, 1, v4
	s_movk_i32 s3, 0x1000
	v_mov_b32_e32 v32, 0
	v_add_u32_e32 v52, s53, v51
	v_lshl_add_u32 v53, s25, 6, v38
	v_lshl_add_u32 v54, v62, 2, 0
	v_lshlrev_b32_e32 v55, 2, v62
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, 0
.LBB28_12:                              ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB28_13 Depth 2
                                        ;     Child Loop BB28_15 Depth 2
	s_add_i32 s52, s58, s57
	s_mul_hi_i32 s53, s52, 34
	s_mul_i32 s52, s52, 34
	s_add_u32 s52, s12, s52
	s_addc_u32 s53, s13, s53
	v_mad_u64_u32 v[2:3], s[60:61], v34, 34, s[52:53]
	v_add_u32_e32 v66, 0x6800, v37
	v_add_u32_e32 v67, 0x8800, v37
	v_mad_i64_i32 v[4:5], s[60:61], v36, 34, v[2:3]
	v_mad_i64_i32 v[6:7], s[60:61], v50, 34, v[2:3]
	v_add_co_u32_e32 v4, vcc, v4, v49
	v_addc_co_u32_e32 v5, vcc, 0, v5, vcc
	v_add_co_u32_e32 v6, vcc, v6, v49
	v_addc_co_u32_e32 v7, vcc, 0, v7, vcc
	v_mad_i64_i32 v[8:9], s[60:61], v51, 34, v[2:3]
	global_load_dword v10, v[4:5], off offset:2
	global_load_dword v11, v[4:5], off offset:70
	global_load_dword v12, v[4:5], off offset:138
	global_load_dword v13, v[4:5], off offset:206
	global_load_dword v14, v[6:7], off offset:2
	global_load_dword v15, v[6:7], off offset:70
	global_load_dword v16, v[6:7], off offset:138
	global_load_dword v17, v[6:7], off offset:206
	v_mad_u64_u32 v[6:7], s[52:53], v35, 34, s[52:53]
	v_add_co_u32_e32 v4, vcc, v8, v49
	v_addc_co_u32_e32 v5, vcc, 0, v9, vcc
	v_mad_i64_i32 v[8:9], s[52:53], v38, 34, v[6:7]
	v_mad_i64_i32 v[6:7], s[52:53], v53, 34, v[6:7]
	v_mad_i64_i32 v[2:3], s[60:61], v52, 34, v[2:3]
	s_lshr_b32 s52, s57, 2
	s_mul_i32 s52, s2, s52
	s_ashr_i32 s53, s52, 31
	s_lshl_b64 s[52:53], s[52:53], 2
	v_add_co_u32_e32 v2, vcc, v2, v49
	s_add_u32 s52, s55, s52
	v_addc_co_u32_e32 v3, vcc, 0, v3, vcc
	s_addc_u32 s53, s56, s53
	global_load_dword v56, v[4:5], off offset:2
	global_load_dword v57, v[4:5], off offset:70
	global_load_dword v58, v[4:5], off offset:138
	global_load_dword v59, v[4:5], off offset:206
	global_load_dword v60, v[2:3], off offset:2
	global_load_dword v61, v[2:3], off offset:70
	global_load_dword v63, v[2:3], off offset:138
	global_load_dword v64, v[2:3], off offset:206
	s_nop 0
	global_load_ushort v8, v[8:9], off
	s_nop 0
	global_load_ushort v6, v[6:7], off
	v_mov_b32_e32 v2, s53
	v_add_co_u32_e32 v4, vcc, s52, v55
	v_addc_co_u32_e32 v5, vcc, 0, v2, vcc
	v_add_co_u32_e32 v2, vcc, s3, v4
	v_addc_co_u32_e32 v3, vcc, 0, v5, vcc
	v_add_co_u32_e32 v4, vcc, 0x2000, v4
	v_addc_co_u32_e32 v5, vcc, 0, v5, vcc
	global_load_dword v7, v55, s[52:53]
	global_load_dword v9, v55, s[52:53] offset:2048
	global_load_dword v65, v[2:3], off
	s_nop 0
	global_load_dword v2, v[2:3], off offset:2048
	s_nop 0
	global_load_dword v3, v[4:5], off
	v_add_u32_e32 v4, 0x2800, v37
	v_add_u32_e32 v5, 0x4800, v37
	s_mov_b32 s59, -8
	s_waitcnt vmcnt(6)
	v_cvt_f32_f16_e32 v8, v8
	s_waitcnt vmcnt(5)
	v_cvt_f32_f16_e32 v6, v6
	ds_write2_b32 v4, v10, v11 offset0:64 offset1:80
	ds_write2_b32 v4, v12, v13 offset0:96 offset1:112
	ds_write2_b32 v5, v14, v15 offset0:96 offset1:112
	ds_write2_b32 v5, v16, v17 offset0:128 offset1:144
	ds_write2_b32 v66, v56, v57 offset0:128 offset1:144
	ds_write2_b32 v66, v58, v59 offset0:160 offset1:176
	ds_write2_b32 v67, v60, v61 offset0:160 offset1:176
	ds_write2_b32 v67, v63, v64 offset0:192 offset1:208
	ds_write_b32 v42, v8 offset:43776
	ds_write_b32 v44, v6 offset:43776
	s_waitcnt vmcnt(3)
	ds_write2st64_b32 v54, v7, v9 offset0:1 offset1:9
	s_waitcnt vmcnt(1)
	ds_write2st64_b32 v54, v65, v2 offset0:17 offset1:25
	s_waitcnt vmcnt(0)
	ds_write_b32 v54, v3 offset:8448
	v_mov_b32_e32 v56, v45
	v_mov_b32_e32 v57, v43
	v_mov_b32_e32 v58, v41
	v_mov_b32_e32 v59, v40
	v_mov_b32_e32 v60, v39
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB28_13:                              ;   Parent Loop BB28_12 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ds_read_b32 v66, v58
	ds_read2_b32 v[12:13], v60 offset1:1
	ds_read2_b32 v[14:15], v60 offset0:2 offset1:3
	ds_read2_b32 v[16:17], v60 offset0:4 offset1:5
	ds_read2_b32 v[64:65], v60 offset0:6 offset1:7
	ds_read_b32 v61, v56
	ds_read2_b32 v[2:3], v59 offset1:1
	v_add_u32_e32 v68, 0x490, v60
	v_add_u32_e32 v70, 0x498, v60
	s_add_i32 s59, s59, 8
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v11, v61, v66
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v4, v2, v12, 0
	v_dot4_i32_i8 v6, v3, v13, v4
	ds_read2_b32 v[4:5], v59 offset0:2 offset1:3
	v_add_u32_e32 v56, 4, v56
	s_cmp_lt_u32 s59, 24
	ds_read2_b32 v[68:69], v68 offset1:1
	ds_read2_b32 v[70:71], v70 offset1:1
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v6, v4, v14, v6
	v_dot4_i32_i8 v8, v5, v15, v6
	ds_read2_b32 v[6:7], v59 offset0:4 offset1:5
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v8, v6, v16, v8
	v_dot4_i32_i8 v10, v7, v17, v8
	ds_read2_b32 v[8:9], v59 offset0:6 offset1:7
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v10, v8, v64, v10
	v_dot4_i32_i8 v10, v9, v65, v10
	v_cvt_f32_i32_e32 v10, v10
	v_fmac_f32_e32 v32, v11, v10
	v_add_u32_e32 v10, 0x4100, v59
	ds_read_b32 v63, v57
	ds_read2_b32 v[10:11], v10 offset1:1
	v_add_u32_e32 v57, 4, v57
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v12, v10, v12, 0
	v_dot4_i32_i8 v67, v11, v13, v12
	v_add_u32_e32 v12, 0x4108, v59
	ds_read2_b32 v[12:13], v12 offset1:1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v14, v12, v14, v67
	v_dot4_i32_i8 v67, v13, v15, v14
	v_add_u32_e32 v14, 0x4110, v59
	ds_read2_b32 v[14:15], v14 offset1:1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v16, v14, v16, v67
	v_dot4_i32_i8 v67, v15, v17, v16
	v_add_u32_e32 v16, 0x4118, v59
	ds_read2_b32 v[16:17], v16 offset1:1
	v_add_u32_e32 v59, 32, v59
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v64, v16, v64, v67
	v_dot4_i32_i8 v64, v17, v65, v64
	v_cvt_f32_i32_e32 v64, v64
	v_mul_f32_e32 v65, v63, v66
	v_add_u32_e32 v66, 0x488, v60
	ds_read2_b32 v[66:67], v66 offset1:1
	v_fmac_f32_e32 v33, v65, v64
	v_add_u32_e32 v64, 0x480, v60
	ds_read_b32 v72, v58 offset:1152
	ds_read2_b32 v[64:65], v64 offset1:1
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v74, v72, v61
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v73, v2, v64, 0
	v_dot4_i32_i8 v64, v10, v64, 0
	v_dot4_i32_i8 v64, v11, v65, v64
	v_dot4_i32_i8 v64, v12, v66, v64
	v_dot4_i32_i8 v64, v13, v67, v64
	v_dot4_i32_i8 v64, v14, v68, v64
	v_dot4_i32_i8 v64, v15, v69, v64
	v_dot4_i32_i8 v64, v16, v70, v64
	v_dot4_i32_i8 v73, v3, v65, v73
	v_dot4_i32_i8 v64, v17, v71, v64
	v_dot4_i32_i8 v73, v4, v66, v73
	v_cvt_f32_i32_e32 v64, v64
	v_dot4_i32_i8 v73, v5, v67, v73
	v_dot4_i32_i8 v73, v6, v68, v73
	v_dot4_i32_i8 v73, v7, v69, v73
	v_mul_f32_e32 v65, v72, v63
	v_dot4_i32_i8 v73, v8, v70, v73
	v_fmac_f32_e32 v30, v65, v64
	v_add_u32_e32 v64, 0x900, v60
	v_dot4_i32_i8 v73, v9, v71, v73
	ds_read_b32 v72, v58 offset:2304
	ds_read2_b32 v[64:65], v64 offset1:1
	v_cvt_f32_i32_e32 v73, v73
	v_add_u32_e32 v66, 0x908, v60
	ds_read2_b32 v[66:67], v66 offset1:1
	v_add_u32_e32 v68, 0x910, v60
	ds_read2_b32 v[68:69], v68 offset1:1
	v_fmac_f32_e32 v31, v74, v73
	v_add_u32_e32 v70, 0x918, v60
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v73, v2, v64, 0
	v_dot4_i32_i8 v64, v10, v64, 0
	ds_read2_b32 v[70:71], v70 offset1:1
	v_dot4_i32_i8 v64, v11, v65, v64
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v64, v12, v66, v64
	v_dot4_i32_i8 v64, v13, v67, v64
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v64, v14, v68, v64
	v_dot4_i32_i8 v64, v15, v69, v64
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v64, v16, v70, v64
	v_dot4_i32_i8 v73, v3, v65, v73
	v_dot4_i32_i8 v64, v17, v71, v64
	v_dot4_i32_i8 v73, v4, v66, v73
	v_cvt_f32_i32_e32 v64, v64
	v_dot4_i32_i8 v73, v5, v67, v73
	v_dot4_i32_i8 v73, v6, v68, v73
	v_dot4_i32_i8 v73, v7, v69, v73
	v_mul_f32_e32 v65, v72, v63
	v_dot4_i32_i8 v73, v8, v70, v73
	v_fmac_f32_e32 v28, v65, v64
	v_add_u32_e32 v64, 0xd80, v60
	v_dot4_i32_i8 v73, v9, v71, v73
	v_mul_f32_e32 v74, v72, v61
	ds_read_b32 v72, v58 offset:3456
	ds_read2_b32 v[64:65], v64 offset1:1
	v_cvt_f32_i32_e32 v73, v73
	v_add_u32_e32 v66, 0xd88, v60
	ds_read2_b32 v[66:67], v66 offset1:1
	v_add_u32_e32 v68, 0xd90, v60
	ds_read2_b32 v[68:69], v68 offset1:1
	v_fmac_f32_e32 v29, v74, v73
	v_add_u32_e32 v70, 0xd98, v60
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v73, v2, v64, 0
	v_dot4_i32_i8 v64, v10, v64, 0
	ds_read2_b32 v[70:71], v70 offset1:1
	v_dot4_i32_i8 v64, v11, v65, v64
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v64, v12, v66, v64
	v_dot4_i32_i8 v64, v13, v67, v64
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v64, v14, v68, v64
	v_dot4_i32_i8 v64, v15, v69, v64
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v64, v16, v70, v64
	v_dot4_i32_i8 v73, v3, v65, v73
	v_dot4_i32_i8 v64, v17, v71, v64
	v_dot4_i32_i8 v73, v4, v66, v73
	v_cvt_f32_i32_e32 v64, v64
	v_dot4_i32_i8 v73, v5, v67, v73
	v_dot4_i32_i8 v73, v6, v68, v73
	v_dot4_i32_i8 v73, v7, v69, v73
	v_mul_f32_e32 v65, v72, v63
	v_dot4_i32_i8 v73, v8, v70, v73
	v_fmac_f32_e32 v26, v65, v64
	v_add_u32_e32 v64, 0x1200, v60
	v_dot4_i32_i8 v73, v9, v71, v73
	v_mul_f32_e32 v74, v72, v61
	ds_read_b32 v72, v58 offset:4608
	ds_read2_b32 v[64:65], v64 offset1:1
	v_cvt_f32_i32_e32 v73, v73
	v_add_u32_e32 v66, 0x1208, v60
	ds_read2_b32 v[66:67], v66 offset1:1
	v_add_u32_e32 v68, 0x1210, v60
	ds_read2_b32 v[68:69], v68 offset1:1
	v_fmac_f32_e32 v27, v74, v73
	v_add_u32_e32 v70, 0x1218, v60
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v73, v2, v64, 0
	v_dot4_i32_i8 v64, v10, v64, 0
	ds_read2_b32 v[70:71], v70 offset1:1
	v_dot4_i32_i8 v64, v11, v65, v64
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v64, v12, v66, v64
	v_dot4_i32_i8 v64, v13, v67, v64
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v64, v14, v68, v64
	v_dot4_i32_i8 v64, v15, v69, v64
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v64, v16, v70, v64
	v_dot4_i32_i8 v73, v3, v65, v73
	v_dot4_i32_i8 v64, v17, v71, v64
	v_dot4_i32_i8 v73, v4, v66, v73
	v_cvt_f32_i32_e32 v64, v64
	v_dot4_i32_i8 v73, v5, v67, v73
	v_dot4_i32_i8 v73, v6, v68, v73
	v_dot4_i32_i8 v73, v7, v69, v73
	v_mul_f32_e32 v65, v72, v63
	v_dot4_i32_i8 v73, v8, v70, v73
	v_fmac_f32_e32 v24, v65, v64
	v_add_u32_e32 v64, 0x1680, v60
	v_dot4_i32_i8 v73, v9, v71, v73
	v_mul_f32_e32 v74, v72, v61
	ds_read_b32 v72, v58 offset:5760
	ds_read2_b32 v[64:65], v64 offset1:1
	v_cvt_f32_i32_e32 v73, v73
	v_add_u32_e32 v66, 0x1688, v60
	ds_read2_b32 v[66:67], v66 offset1:1
	v_add_u32_e32 v68, 0x1690, v60
	ds_read2_b32 v[68:69], v68 offset1:1
	v_fmac_f32_e32 v25, v74, v73
	v_add_u32_e32 v70, 0x1698, v60
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v73, v2, v64, 0
	v_dot4_i32_i8 v64, v10, v64, 0
	ds_read2_b32 v[70:71], v70 offset1:1
	v_dot4_i32_i8 v64, v11, v65, v64
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v64, v12, v66, v64
	v_dot4_i32_i8 v64, v13, v67, v64
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v64, v14, v68, v64
	v_dot4_i32_i8 v64, v15, v69, v64
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v64, v16, v70, v64
	v_dot4_i32_i8 v73, v3, v65, v73
	v_dot4_i32_i8 v64, v17, v71, v64
	v_dot4_i32_i8 v73, v4, v66, v73
	v_cvt_f32_i32_e32 v64, v64
	v_dot4_i32_i8 v73, v5, v67, v73
	v_dot4_i32_i8 v73, v6, v68, v73
	v_dot4_i32_i8 v73, v7, v69, v73
	v_mul_f32_e32 v65, v72, v63
	v_dot4_i32_i8 v73, v8, v70, v73
	v_fmac_f32_e32 v22, v65, v64
	v_add_u32_e32 v64, 0x1b00, v60
	v_dot4_i32_i8 v73, v9, v71, v73
	v_mul_f32_e32 v74, v72, v61
	ds_read_b32 v72, v58 offset:6912
	ds_read2_b32 v[64:65], v64 offset1:1
	v_cvt_f32_i32_e32 v73, v73
	v_add_u32_e32 v66, 0x1b08, v60
	ds_read2_b32 v[66:67], v66 offset1:1
	v_add_u32_e32 v68, 0x1b10, v60
	ds_read2_b32 v[68:69], v68 offset1:1
	v_fmac_f32_e32 v23, v74, v73
	v_add_u32_e32 v70, 0x1b18, v60
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v73, v2, v64, 0
	v_dot4_i32_i8 v64, v10, v64, 0
	ds_read2_b32 v[70:71], v70 offset1:1
	v_dot4_i32_i8 v64, v11, v65, v64
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v64, v12, v66, v64
	v_dot4_i32_i8 v64, v13, v67, v64
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v64, v14, v68, v64
	v_dot4_i32_i8 v64, v15, v69, v64
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v64, v16, v70, v64
	v_dot4_i32_i8 v64, v17, v71, v64
	v_cvt_f32_i32_e32 v64, v64
	v_dot4_i32_i8 v73, v3, v65, v73
	v_mul_f32_e32 v65, v72, v63
	v_mul_f32_e32 v74, v72, v61
	v_fmac_f32_e32 v20, v65, v64
	v_add_u32_e32 v64, 0x1f80, v60
	ds_read_b32 v72, v58 offset:8064
	ds_read2_b32 v[64:65], v64 offset1:1
	v_dot4_i32_i8 v73, v4, v66, v73
	v_add_u32_e32 v66, 0x1f88, v60
	v_dot4_i32_i8 v73, v5, v67, v73
	ds_read2_b32 v[66:67], v66 offset1:1
	v_dot4_i32_i8 v73, v6, v68, v73
	v_add_u32_e32 v68, 0x1f90, v60
	v_dot4_i32_i8 v73, v7, v69, v73
	ds_read2_b32 v[68:69], v68 offset1:1
	v_dot4_i32_i8 v73, v8, v70, v73
	v_add_u32_e32 v70, 0x1f98, v60
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v2, v2, v64, 0
	v_dot4_i32_i8 v73, v9, v71, v73
	ds_read2_b32 v[70:71], v70 offset1:1
	v_dot4_i32_i8 v2, v3, v65, v2
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v2, v4, v66, v2
	v_dot4_i32_i8 v2, v5, v67, v2
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v2, v6, v68, v2
	v_dot4_i32_i8 v2, v7, v69, v2
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v2, v8, v70, v2
	v_dot4_i32_i8 v2, v9, v71, v2
	v_cvt_f32_i32_e32 v2, v2
	v_mul_f32_e32 v3, v72, v61
	v_cvt_f32_i32_e32 v73, v73
	v_add_u32_e32 v60, 32, v60
	v_fmac_f32_e32 v19, v3, v2
	v_dot4_i32_i8 v2, v10, v64, 0
	v_dot4_i32_i8 v2, v11, v65, v2
	v_dot4_i32_i8 v2, v12, v66, v2
	v_dot4_i32_i8 v2, v13, v67, v2
	v_dot4_i32_i8 v2, v14, v68, v2
	v_dot4_i32_i8 v2, v15, v69, v2
	v_dot4_i32_i8 v2, v16, v70, v2
	v_dot4_i32_i8 v2, v17, v71, v2
	v_cvt_f32_i32_e32 v2, v2
	v_mul_f32_e32 v3, v72, v63
	v_fmac_f32_e32 v21, v74, v73
	v_add_u32_e32 v58, 4, v58
	v_fmac_f32_e32 v18, v3, v2
	s_cbranch_scc1 .LBB28_13
; %bb.14:                               ; %_ZL22vec_dot_q8_0_q8_1_dp4aILi64ELi128ELi8EEvPKiS1_Pfi.exit.i
                                        ;   in Loop: Header=BB28_12 Depth=1
	s_add_u32 s52, s52, s8
	s_addc_u32 s53, s53, s9
	v_mov_b32_e32 v2, s53
	v_add_co_u32_e32 v4, vcc, s52, v55
	v_addc_co_u32_e32 v5, vcc, 0, v2, vcc
	v_add_co_u32_e32 v2, vcc, s3, v4
	v_addc_co_u32_e32 v3, vcc, 0, v5, vcc
	v_add_co_u32_e32 v4, vcc, 0x2000, v4
	s_barrier
	v_addc_co_u32_e32 v5, vcc, 0, v5, vcc
	global_load_dword v6, v55, s[52:53]
	global_load_dword v7, v55, s[52:53] offset:2048
	global_load_dword v8, v[2:3], off
	s_nop 0
	global_load_dword v2, v[2:3], off offset:2048
	s_nop 0
	global_load_dword v3, v[4:5], off
	s_mov_b32 s52, -8
	v_mov_b32_e32 v56, v41
	v_mov_b32_e32 v57, v48
	v_mov_b32_e32 v58, v47
	v_mov_b32_e32 v59, v46
	v_mov_b32_e32 v60, v39
	s_waitcnt vmcnt(3)
	ds_write2st64_b32 v54, v6, v7 offset0:1 offset1:9
	s_waitcnt vmcnt(1)
	ds_write2st64_b32 v54, v8, v2 offset0:17 offset1:25
	s_waitcnt vmcnt(0)
	ds_write_b32 v54, v3 offset:8448
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB28_15:                              ;   Parent Loop BB28_12 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ds_read_b32 v66, v56
	ds_read2_b32 v[12:13], v60 offset1:1
	ds_read2_b32 v[14:15], v60 offset0:2 offset1:3
	ds_read2_b32 v[16:17], v60 offset0:4 offset1:5
	ds_read2_b32 v[64:65], v60 offset0:6 offset1:7
	ds_read_b32 v61, v57
	ds_read2_b32 v[2:3], v59 offset1:1
	v_add_u32_e32 v68, 0x490, v60
	v_add_u32_e32 v70, 0x498, v60
	s_add_i32 s52, s52, 8
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v11, v61, v66
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v4, v2, v12, 0
	v_dot4_i32_i8 v6, v3, v13, v4
	ds_read2_b32 v[4:5], v59 offset0:2 offset1:3
	v_add_u32_e32 v57, 4, v57
	s_cmp_lt_u32 s52, 24
	ds_read2_b32 v[68:69], v68 offset1:1
	ds_read2_b32 v[70:71], v70 offset1:1
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v6, v4, v14, v6
	v_dot4_i32_i8 v8, v5, v15, v6
	ds_read2_b32 v[6:7], v59 offset0:4 offset1:5
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v8, v6, v16, v8
	v_dot4_i32_i8 v10, v7, v17, v8
	ds_read2_b32 v[8:9], v59 offset0:6 offset1:7
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v10, v8, v64, v10
	v_dot4_i32_i8 v10, v9, v65, v10
	v_cvt_f32_i32_e32 v10, v10
	v_fmac_f32_e32 v32, v11, v10
	v_add_u32_e32 v10, 0x4100, v59
	ds_read_b32 v63, v58
	ds_read2_b32 v[10:11], v10 offset1:1
	v_add_u32_e32 v58, 4, v58
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v12, v10, v12, 0
	v_dot4_i32_i8 v67, v11, v13, v12
	v_add_u32_e32 v12, 0x4108, v59
	ds_read2_b32 v[12:13], v12 offset1:1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v14, v12, v14, v67
	v_dot4_i32_i8 v67, v13, v15, v14
	v_add_u32_e32 v14, 0x4110, v59
	ds_read2_b32 v[14:15], v14 offset1:1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v16, v14, v16, v67
	v_dot4_i32_i8 v67, v15, v17, v16
	v_add_u32_e32 v16, 0x4118, v59
	ds_read2_b32 v[16:17], v16 offset1:1
	v_add_u32_e32 v59, 32, v59
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v64, v16, v64, v67
	v_dot4_i32_i8 v64, v17, v65, v64
	v_cvt_f32_i32_e32 v64, v64
	v_mul_f32_e32 v65, v63, v66
	v_add_u32_e32 v66, 0x488, v60
	ds_read2_b32 v[66:67], v66 offset1:1
	v_fmac_f32_e32 v33, v65, v64
	v_add_u32_e32 v64, 0x480, v60
	ds_read_b32 v72, v56 offset:1152
	ds_read2_b32 v[64:65], v64 offset1:1
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v74, v72, v61
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v73, v2, v64, 0
	v_dot4_i32_i8 v64, v10, v64, 0
	v_dot4_i32_i8 v64, v11, v65, v64
	v_dot4_i32_i8 v64, v12, v66, v64
	v_dot4_i32_i8 v64, v13, v67, v64
	v_dot4_i32_i8 v64, v14, v68, v64
	v_dot4_i32_i8 v64, v15, v69, v64
	v_dot4_i32_i8 v64, v16, v70, v64
	v_dot4_i32_i8 v73, v3, v65, v73
	v_dot4_i32_i8 v64, v17, v71, v64
	v_dot4_i32_i8 v73, v4, v66, v73
	v_cvt_f32_i32_e32 v64, v64
	v_dot4_i32_i8 v73, v5, v67, v73
	v_dot4_i32_i8 v73, v6, v68, v73
	v_dot4_i32_i8 v73, v7, v69, v73
	v_mul_f32_e32 v65, v72, v63
	v_dot4_i32_i8 v73, v8, v70, v73
	v_fmac_f32_e32 v30, v65, v64
	v_add_u32_e32 v64, 0x900, v60
	v_dot4_i32_i8 v73, v9, v71, v73
	ds_read_b32 v72, v56 offset:2304
	ds_read2_b32 v[64:65], v64 offset1:1
	v_cvt_f32_i32_e32 v73, v73
	v_add_u32_e32 v66, 0x908, v60
	ds_read2_b32 v[66:67], v66 offset1:1
	v_add_u32_e32 v68, 0x910, v60
	ds_read2_b32 v[68:69], v68 offset1:1
	v_fmac_f32_e32 v31, v74, v73
	v_add_u32_e32 v70, 0x918, v60
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v73, v2, v64, 0
	v_dot4_i32_i8 v64, v10, v64, 0
	ds_read2_b32 v[70:71], v70 offset1:1
	v_dot4_i32_i8 v64, v11, v65, v64
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v64, v12, v66, v64
	v_dot4_i32_i8 v64, v13, v67, v64
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v64, v14, v68, v64
	v_dot4_i32_i8 v64, v15, v69, v64
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v64, v16, v70, v64
	v_dot4_i32_i8 v73, v3, v65, v73
	v_dot4_i32_i8 v64, v17, v71, v64
	v_dot4_i32_i8 v73, v4, v66, v73
	v_cvt_f32_i32_e32 v64, v64
	v_dot4_i32_i8 v73, v5, v67, v73
	v_dot4_i32_i8 v73, v6, v68, v73
	v_dot4_i32_i8 v73, v7, v69, v73
	v_mul_f32_e32 v65, v72, v63
	v_dot4_i32_i8 v73, v8, v70, v73
	v_fmac_f32_e32 v28, v65, v64
	v_add_u32_e32 v64, 0xd80, v60
	v_dot4_i32_i8 v73, v9, v71, v73
	v_mul_f32_e32 v74, v72, v61
	ds_read_b32 v72, v56 offset:3456
	ds_read2_b32 v[64:65], v64 offset1:1
	v_cvt_f32_i32_e32 v73, v73
	v_add_u32_e32 v66, 0xd88, v60
	ds_read2_b32 v[66:67], v66 offset1:1
	v_add_u32_e32 v68, 0xd90, v60
	ds_read2_b32 v[68:69], v68 offset1:1
	v_fmac_f32_e32 v29, v74, v73
	v_add_u32_e32 v70, 0xd98, v60
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v73, v2, v64, 0
	v_dot4_i32_i8 v64, v10, v64, 0
	ds_read2_b32 v[70:71], v70 offset1:1
	v_dot4_i32_i8 v64, v11, v65, v64
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v64, v12, v66, v64
	v_dot4_i32_i8 v64, v13, v67, v64
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v64, v14, v68, v64
	v_dot4_i32_i8 v64, v15, v69, v64
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v64, v16, v70, v64
	v_dot4_i32_i8 v73, v3, v65, v73
	v_dot4_i32_i8 v64, v17, v71, v64
	v_dot4_i32_i8 v73, v4, v66, v73
	v_cvt_f32_i32_e32 v64, v64
	v_dot4_i32_i8 v73, v5, v67, v73
	v_dot4_i32_i8 v73, v6, v68, v73
	v_dot4_i32_i8 v73, v7, v69, v73
	v_mul_f32_e32 v65, v72, v63
	v_dot4_i32_i8 v73, v8, v70, v73
	v_fmac_f32_e32 v26, v65, v64
	v_add_u32_e32 v64, 0x1200, v60
	v_dot4_i32_i8 v73, v9, v71, v73
	v_mul_f32_e32 v74, v72, v61
	ds_read_b32 v72, v56 offset:4608
	ds_read2_b32 v[64:65], v64 offset1:1
	v_cvt_f32_i32_e32 v73, v73
	v_add_u32_e32 v66, 0x1208, v60
	ds_read2_b32 v[66:67], v66 offset1:1
	v_add_u32_e32 v68, 0x1210, v60
	ds_read2_b32 v[68:69], v68 offset1:1
	v_fmac_f32_e32 v27, v74, v73
	v_add_u32_e32 v70, 0x1218, v60
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v73, v2, v64, 0
	v_dot4_i32_i8 v64, v10, v64, 0
	ds_read2_b32 v[70:71], v70 offset1:1
	v_dot4_i32_i8 v64, v11, v65, v64
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v64, v12, v66, v64
	v_dot4_i32_i8 v64, v13, v67, v64
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v64, v14, v68, v64
	v_dot4_i32_i8 v64, v15, v69, v64
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v64, v16, v70, v64
	v_dot4_i32_i8 v73, v3, v65, v73
	v_dot4_i32_i8 v64, v17, v71, v64
	v_dot4_i32_i8 v73, v4, v66, v73
	v_cvt_f32_i32_e32 v64, v64
	v_dot4_i32_i8 v73, v5, v67, v73
	v_dot4_i32_i8 v73, v6, v68, v73
	v_dot4_i32_i8 v73, v7, v69, v73
	v_mul_f32_e32 v65, v72, v63
	v_dot4_i32_i8 v73, v8, v70, v73
	v_fmac_f32_e32 v24, v65, v64
	v_add_u32_e32 v64, 0x1680, v60
	v_dot4_i32_i8 v73, v9, v71, v73
	v_mul_f32_e32 v74, v72, v61
	ds_read_b32 v72, v56 offset:5760
	ds_read2_b32 v[64:65], v64 offset1:1
	v_cvt_f32_i32_e32 v73, v73
	v_add_u32_e32 v66, 0x1688, v60
	ds_read2_b32 v[66:67], v66 offset1:1
	v_add_u32_e32 v68, 0x1690, v60
	ds_read2_b32 v[68:69], v68 offset1:1
	v_fmac_f32_e32 v25, v74, v73
	v_add_u32_e32 v70, 0x1698, v60
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v73, v2, v64, 0
	v_dot4_i32_i8 v64, v10, v64, 0
	ds_read2_b32 v[70:71], v70 offset1:1
	v_dot4_i32_i8 v64, v11, v65, v64
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v64, v12, v66, v64
	v_dot4_i32_i8 v64, v13, v67, v64
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v64, v14, v68, v64
	v_dot4_i32_i8 v64, v15, v69, v64
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v64, v16, v70, v64
	v_dot4_i32_i8 v73, v3, v65, v73
	v_dot4_i32_i8 v64, v17, v71, v64
	v_dot4_i32_i8 v73, v4, v66, v73
	v_cvt_f32_i32_e32 v64, v64
	v_dot4_i32_i8 v73, v5, v67, v73
	v_dot4_i32_i8 v73, v6, v68, v73
	v_dot4_i32_i8 v73, v7, v69, v73
	v_mul_f32_e32 v65, v72, v63
	v_dot4_i32_i8 v73, v8, v70, v73
	v_fmac_f32_e32 v22, v65, v64
	v_add_u32_e32 v64, 0x1b00, v60
	v_dot4_i32_i8 v73, v9, v71, v73
	v_mul_f32_e32 v74, v72, v61
	ds_read_b32 v72, v56 offset:6912
	ds_read2_b32 v[64:65], v64 offset1:1
	v_cvt_f32_i32_e32 v73, v73
	v_add_u32_e32 v66, 0x1b08, v60
	ds_read2_b32 v[66:67], v66 offset1:1
	v_add_u32_e32 v68, 0x1b10, v60
	ds_read2_b32 v[68:69], v68 offset1:1
	v_fmac_f32_e32 v23, v74, v73
	v_add_u32_e32 v70, 0x1b18, v60
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v73, v2, v64, 0
	v_dot4_i32_i8 v64, v10, v64, 0
	ds_read2_b32 v[70:71], v70 offset1:1
	v_dot4_i32_i8 v64, v11, v65, v64
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v64, v12, v66, v64
	v_dot4_i32_i8 v64, v13, v67, v64
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v64, v14, v68, v64
	v_dot4_i32_i8 v64, v15, v69, v64
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v64, v16, v70, v64
	v_dot4_i32_i8 v64, v17, v71, v64
	v_cvt_f32_i32_e32 v64, v64
	v_dot4_i32_i8 v73, v3, v65, v73
	v_mul_f32_e32 v65, v72, v63
	v_mul_f32_e32 v74, v72, v61
	v_fmac_f32_e32 v20, v65, v64
	v_add_u32_e32 v64, 0x1f80, v60
	ds_read_b32 v72, v56 offset:8064
	ds_read2_b32 v[64:65], v64 offset1:1
	v_dot4_i32_i8 v73, v4, v66, v73
	v_add_u32_e32 v66, 0x1f88, v60
	v_dot4_i32_i8 v73, v5, v67, v73
	ds_read2_b32 v[66:67], v66 offset1:1
	v_dot4_i32_i8 v73, v6, v68, v73
	v_add_u32_e32 v68, 0x1f90, v60
	v_dot4_i32_i8 v73, v7, v69, v73
	ds_read2_b32 v[68:69], v68 offset1:1
	v_dot4_i32_i8 v73, v8, v70, v73
	v_add_u32_e32 v70, 0x1f98, v60
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v2, v2, v64, 0
	v_dot4_i32_i8 v73, v9, v71, v73
	ds_read2_b32 v[70:71], v70 offset1:1
	v_dot4_i32_i8 v2, v3, v65, v2
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v2, v4, v66, v2
	v_dot4_i32_i8 v2, v5, v67, v2
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v2, v6, v68, v2
	v_dot4_i32_i8 v2, v7, v69, v2
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v2, v8, v70, v2
	v_dot4_i32_i8 v2, v9, v71, v2
	v_cvt_f32_i32_e32 v2, v2
	v_mul_f32_e32 v3, v72, v61
	v_cvt_f32_i32_e32 v73, v73
	v_add_u32_e32 v60, 32, v60
	v_fmac_f32_e32 v19, v3, v2
	v_dot4_i32_i8 v2, v10, v64, 0
	v_dot4_i32_i8 v2, v11, v65, v2
	v_dot4_i32_i8 v2, v12, v66, v2
	v_dot4_i32_i8 v2, v13, v67, v2
	v_dot4_i32_i8 v2, v14, v68, v2
	v_dot4_i32_i8 v2, v15, v69, v2
	v_dot4_i32_i8 v2, v16, v70, v2
	v_dot4_i32_i8 v2, v17, v71, v2
	v_cvt_f32_i32_e32 v2, v2
	v_mul_f32_e32 v3, v72, v63
	v_fmac_f32_e32 v21, v74, v73
	v_add_u32_e32 v56, 4, v56
	v_fmac_f32_e32 v18, v3, v2
	s_cbranch_scc1 .LBB28_15
; %bb.16:                               ; %_ZL22vec_dot_q8_0_q8_1_dp4aILi64ELi128ELi8EEvPKiS1_Pfi.exit104.i
                                        ;   in Loop: Header=BB28_12 Depth=1
	s_add_i32 s57, s57, 8
	s_cmp_ge_i32 s57, s22
	s_barrier
	s_cbranch_scc0 .LBB28_12
	s_branch .LBB28_20
.LBB28_17:                              ; %Flow2627
	s_mov_b32 s35, 0
	s_and_b64 vcc, exec, s[54:55]
	s_cbranch_vccnz .LBB28_10
.LBB28_18:
                                        ; implicit-def: $vgpr18
                                        ; implicit-def: $vgpr19
                                        ; implicit-def: $sgpr8_sgpr9
                                        ; implicit-def: $vgpr2
	s_branch .LBB28_107
.LBB28_19:
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v32, 0
.LBB28_20:                              ; %._crit_edge.i
	s_not_b32 s2, s7
	s_add_i32 s33, s33, s2
	v_cmp_ge_i32_e32 vcc, s33, v75
	s_mov_b64 s[2:3], 0
                                        ; implicit-def: $sgpr8_sgpr9
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[52:53], vcc
	s_cbranch_execz .LBB28_36
; %bb.21:                               ; %.preheader.i.i
	v_lshl_add_u32 v3, v75, 2, 0
	ds_read_b32 v2, v3
	s_add_i32 s2, s35, s54
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 2
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[4:5], s[8:9], v2, s27, v[0:1]
	s_add_u32 s8, s28, s2
	s_addc_u32 s9, s29, s3
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_mov_b32_e32 v2, s9
	v_add_co_u32_e32 v4, vcc, s8, v4
	v_addc_co_u32_e32 v5, vcc, v2, v5, vcc
	v_add_u32_e32 v2, 8, v75
	v_cmp_ge_u32_e32 vcc, s33, v2
	s_mov_b64 s[2:3], 0
	global_store_dword v[4:5], v32, off
	global_store_dword v[4:5], v33, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[54:55], vcc
	s_cbranch_execz .LBB28_35
; %bb.22:                               ; %.preheader.1.i.i
	ds_read_b32 v2, v3 offset:32
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[4:5], s[2:3], v2, s27, v[0:1]
	v_add_u32_e32 v5, 16, v75
	v_cmp_ge_u32_e32 vcc, s33, v5
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_mov_b32_e32 v2, s9
	v_add_co_u32_e64 v4, s[2:3], s8, v4
	v_addc_co_u32_e64 v5, s[2:3], v2, v5, s[2:3]
	s_mov_b64 s[2:3], 0
	global_store_dword v[4:5], v31, off
	global_store_dword v[4:5], v30, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[56:57], vcc
	s_cbranch_execz .LBB28_34
; %bb.23:                               ; %.preheader.2.i.i
	ds_read_b32 v2, v3 offset:64
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[4:5], s[2:3], v2, s27, v[0:1]
	v_add_u32_e32 v5, 24, v75
	v_cmp_ge_u32_e32 vcc, s33, v5
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_mov_b32_e32 v2, s9
	v_add_co_u32_e64 v4, s[2:3], s8, v4
	v_addc_co_u32_e64 v5, s[2:3], v2, v5, s[2:3]
	s_mov_b64 s[2:3], 0
	global_store_dword v[4:5], v29, off
	global_store_dword v[4:5], v28, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[58:59], vcc
	s_cbranch_execz .LBB28_33
; %bb.24:                               ; %.preheader.3.i.i
	ds_read_b32 v2, v3 offset:96
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[4:5], s[2:3], v2, s27, v[0:1]
	v_add_u32_e32 v5, 32, v75
	v_cmp_ge_u32_e32 vcc, s33, v5
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_mov_b32_e32 v2, s9
	v_add_co_u32_e64 v4, s[2:3], s8, v4
	v_addc_co_u32_e64 v5, s[2:3], v2, v5, s[2:3]
	s_mov_b64 s[2:3], 0
	global_store_dword v[4:5], v27, off
	global_store_dword v[4:5], v26, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[60:61], vcc
	s_cbranch_execz .LBB28_32
; %bb.25:                               ; %.preheader.4.i.i
	ds_read_b32 v2, v3 offset:128
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[4:5], s[2:3], v2, s27, v[0:1]
	v_add_u32_e32 v5, 40, v75
	v_cmp_ge_u32_e32 vcc, s33, v5
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_mov_b32_e32 v2, s9
	v_add_co_u32_e64 v4, s[2:3], s8, v4
	v_addc_co_u32_e64 v5, s[2:3], v2, v5, s[2:3]
	s_mov_b64 s[2:3], 0
	global_store_dword v[4:5], v25, off
	global_store_dword v[4:5], v24, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[62:63], vcc
	s_cbranch_execz .LBB28_31
; %bb.26:                               ; %.preheader.5.i.i
	ds_read_b32 v2, v3 offset:160
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[4:5], s[2:3], v2, s27, v[0:1]
	v_add_u32_e32 v5, 48, v75
	v_cmp_ge_u32_e32 vcc, s33, v5
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_mov_b32_e32 v2, s9
	v_add_co_u32_e64 v4, s[2:3], s8, v4
	v_addc_co_u32_e64 v5, s[2:3], v2, v5, s[2:3]
	s_mov_b64 s[2:3], 0
	global_store_dword v[4:5], v23, off
	global_store_dword v[4:5], v22, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[64:65], vcc
	s_cbranch_execz .LBB28_30
; %bb.27:                               ; %.preheader.6.i.i
	ds_read_b32 v2, v3 offset:192
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[4:5], s[2:3], v2, s27, v[0:1]
	v_add_u32_e32 v5, 56, v75
	v_cmp_ge_u32_e32 vcc, s33, v5
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_mov_b32_e32 v2, s9
	v_add_co_u32_e64 v4, s[2:3], s8, v4
	v_addc_co_u32_e64 v5, s[2:3], v2, v5, s[2:3]
	s_mov_b64 s[2:3], 0
	global_store_dword v[4:5], v21, off
	global_store_dword v[4:5], v20, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[66:67], vcc
	s_cbranch_execz .LBB28_29
; %bb.28:                               ; %.preheader.7.i.i
	ds_read_b32 v2, v3 offset:224
	s_mov_b64 s[2:3], exec
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v2, v2, s27
.LBB28_29:                              ; %Flow2636
	s_or_b64 exec, exec, s[66:67]
	s_and_b64 s[2:3], s[2:3], exec
.LBB28_30:                              ; %Flow2635
	s_or_b64 exec, exec, s[64:65]
	s_and_b64 s[2:3], s[2:3], exec
.LBB28_31:                              ; %Flow2634
	s_or_b64 exec, exec, s[62:63]
	s_and_b64 s[2:3], s[2:3], exec
.LBB28_32:                              ; %Flow2633
	s_or_b64 exec, exec, s[60:61]
	s_and_b64 s[2:3], s[2:3], exec
.LBB28_33:                              ; %Flow2632
	s_or_b64 exec, exec, s[58:59]
	s_and_b64 s[2:3], s[2:3], exec
.LBB28_34:                              ; %Flow2631
	s_or_b64 exec, exec, s[56:57]
	s_and_b64 s[2:3], s[2:3], exec
.LBB28_35:                              ; %Flow2630
	s_or_b64 exec, exec, s[54:55]
	s_and_b64 s[2:3], s[2:3], exec
.LBB28_36:                              ; %Flow2629
	s_or_b64 exec, exec, s[52:53]
	s_branch .LBB28_107
.LBB28_37:                              ; %.split
	s_add_i32 s7, s23, 0x7f
	s_ashr_i32 s8, s7, 31
	s_lshr_b32 s8, s8, 25
	s_add_i32 s7, s7, s8
	s_ashr_i32 s8, s7, 7
	s_load_dword s7, s[4:5], 0xa8
	s_mul_i32 s4, s22, s8
	s_mul_i32 s4, s4, s38
	s_mul_i32 s23, s4, s46
	s_mul_i32 s23, s23, s34
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_u32_e32 v2, s7
	s_mul_hi_u32 s4, s23, s6
	s_cmp_lg_u32 s4, 0
	s_mul_i32 s33, s23, s6
	s_cbranch_scc0 .LBB28_103
; %bb.38:
	v_madmk_f32 v3, 0, 0x4f800000, v2
	v_rcp_f32_e32 v3, v3
                                        ; implicit-def: $vgpr4
	s_sub_u32 s5, 0, s7
	s_subb_u32 s35, 0, 0
	v_mul_f32_e32 v3, 0x5f7ffffc, v3
	v_mul_f32_e32 v4, 0x2f800000, v3
	v_trunc_f32_e32 v4, v4
	v_madmk_f32 v3, v4, 0xcf800000, v3
	v_cvt_u32_f32_e32 v4, v4
	v_cvt_u32_f32_e32 v3, v3
	v_readfirstlane_b32 s52, v4
	v_readfirstlane_b32 s53, v3
	s_mul_hi_u32 s55, s5, s53
	s_mul_i32 s56, s5, s52
	s_mul_i32 s54, s35, s53
	s_add_i32 s55, s55, s56
	s_add_i32 s55, s55, s54
	s_mul_i32 s57, s5, s53
	s_mul_i32 s56, s53, s55
	s_mul_hi_u32 s58, s53, s57
	s_mul_hi_u32 s54, s53, s55
	s_add_u32 s56, s58, s56
	s_addc_u32 s54, 0, s54
	s_mul_hi_u32 s59, s52, s57
	s_mul_i32 s57, s52, s57
	s_add_u32 s56, s56, s57
	s_mul_hi_u32 s58, s52, s55
	s_addc_u32 s54, s54, s59
	s_addc_u32 s56, s58, 0
	s_mul_i32 s55, s52, s55
	s_add_u32 s54, s54, s55
	s_addc_u32 s55, 0, s56
	s_add_u32 s53, s53, s54
	s_addc_u32 s52, s52, s55
	s_mul_i32 s54, s5, s52
	s_mul_hi_u32 s55, s5, s53
	s_add_i32 s54, s55, s54
	s_mul_i32 s35, s35, s53
	s_add_i32 s54, s54, s35
	s_mul_i32 s5, s5, s53
	s_mul_hi_u32 s55, s52, s5
	s_mul_i32 s56, s52, s5
	s_mul_i32 s58, s53, s54
	s_mul_hi_u32 s5, s53, s5
	s_mul_hi_u32 s57, s53, s54
	s_add_u32 s5, s5, s58
	s_addc_u32 s57, 0, s57
	s_add_u32 s5, s5, s56
	s_mul_hi_u32 s35, s52, s54
	s_addc_u32 s5, s57, s55
	s_addc_u32 s35, s35, 0
	s_mul_i32 s54, s52, s54
	s_add_u32 s5, s5, s54
	s_addc_u32 s35, 0, s35
	s_add_u32 s5, s53, s5
	s_addc_u32 s35, s52, s35
	s_mul_i32 s53, s33, s35
	s_mul_hi_u32 s54, s33, s5
	s_mul_hi_u32 s52, s33, s35
	s_add_u32 s53, s54, s53
	s_addc_u32 s52, 0, s52
	s_mul_hi_u32 s55, s4, s5
	s_mul_i32 s5, s4, s5
	s_add_u32 s5, s53, s5
	s_mul_hi_u32 s54, s4, s35
	s_addc_u32 s5, s52, s55
	s_addc_u32 s52, s54, 0
	s_mul_i32 s35, s4, s35
	s_add_u32 s5, s5, s35
	s_addc_u32 s35, 0, s52
	s_mul_i32 s35, s7, s35
	s_mul_hi_u32 s54, s7, s5
	s_add_u32 s52, s5, 1
	s_add_u32 s53, s5, 2
	s_add_i32 s54, s54, s35
	s_mul_i32 s35, s7, s5
	s_sub_u32 s35, s33, s35
	s_subb_u32 s4, s4, s54
	s_sub_u32 s54, s35, s7
	s_subb_u32 s55, s4, 0
	s_cmp_ge_u32 s54, s7
	s_cselect_b32 s54, -1, 0
	s_cmp_eq_u32 s55, 0
	s_cselect_b32 s54, s54, -1
	s_cmp_lg_u32 s54, 0
	s_cselect_b32 s52, s53, s52
	s_cmp_ge_u32 s35, s7
	s_cselect_b32 s35, -1, 0
	s_cmp_eq_u32 s4, 0
	s_cselect_b32 s4, s35, -1
	s_cmp_lg_u32 s4, 0
	s_cselect_b32 s4, s52, s5
	v_cvt_f32_u32_e32 v3, s7
	s_cbranch_execnz .LBB28_40
.LBB28_39:
	v_rcp_iflag_f32_e32 v4, v3
	s_sub_i32 s4, 0, s7
	v_mul_f32_e32 v4, 0x4f7ffffe, v4
	v_cvt_u32_f32_e32 v4, v4
	v_readfirstlane_b32 s5, v4
	s_mul_i32 s4, s4, s5
	s_mul_hi_u32 s4, s5, s4
	s_add_i32 s5, s5, s4
	s_mul_hi_u32 s4, s33, s5
	s_mul_i32 s8, s4, s7
	s_sub_i32 s8, s33, s8
	s_add_i32 s5, s4, 1
	s_sub_i32 s9, s8, s7
	s_cmp_ge_u32 s8, s7
	s_cselect_b32 s4, s5, s4
	s_cselect_b32 s8, s9, s8
	s_add_i32 s5, s4, 1
	s_cmp_ge_u32 s8, s7
	s_cselect_b32 s4, s5, s4
.LBB28_40:
	s_add_i32 s5, s6, 1
	s_mul_hi_u32 s33, s23, s5
	s_cmp_lg_u32 s33, 0
	s_mul_i32 s5, s23, s5
	s_cbranch_scc0 .LBB28_104
; %bb.41:
	v_mov_b32_e32 v4, 0x4f800000
	v_mac_f32_e32 v2, 0, v4
	v_rcp_f32_e32 v2, v2
	s_sub_u32 s23, 0, s7
	s_subb_u32 s35, 0, 0
	v_mul_f32_e32 v2, 0x5f7ffffc, v2
	v_mul_f32_e32 v4, 0x2f800000, v2
	v_trunc_f32_e32 v4, v4
	v_madmk_f32 v2, v4, 0xcf800000, v2
	v_cvt_u32_f32_e32 v4, v4
	v_cvt_u32_f32_e32 v2, v2
	v_readfirstlane_b32 s52, v4
	v_readfirstlane_b32 s53, v2
	s_mul_hi_u32 s55, s23, s53
	s_mul_i32 s56, s23, s52
	s_mul_i32 s54, s35, s53
	s_add_i32 s55, s55, s56
	s_add_i32 s55, s55, s54
	s_mul_i32 s57, s23, s53
	s_mul_i32 s56, s53, s55
	s_mul_hi_u32 s58, s53, s57
	s_mul_hi_u32 s54, s53, s55
	s_add_u32 s56, s58, s56
	s_addc_u32 s54, 0, s54
	s_mul_hi_u32 s59, s52, s57
	s_mul_i32 s57, s52, s57
	s_add_u32 s56, s56, s57
	s_mul_hi_u32 s58, s52, s55
	s_addc_u32 s54, s54, s59
	s_addc_u32 s56, s58, 0
	s_mul_i32 s55, s52, s55
	s_add_u32 s54, s54, s55
	s_addc_u32 s55, 0, s56
	s_add_u32 s53, s53, s54
	s_addc_u32 s52, s52, s55
	s_mul_i32 s54, s23, s52
	s_mul_hi_u32 s55, s23, s53
	s_add_i32 s54, s55, s54
	s_mul_i32 s35, s35, s53
	s_add_i32 s54, s54, s35
	s_mul_i32 s23, s23, s53
	s_mul_hi_u32 s55, s52, s23
	s_mul_i32 s56, s52, s23
	s_mul_i32 s58, s53, s54
	s_mul_hi_u32 s23, s53, s23
	s_mul_hi_u32 s57, s53, s54
	s_add_u32 s23, s23, s58
	s_addc_u32 s57, 0, s57
	s_add_u32 s23, s23, s56
	s_mul_hi_u32 s35, s52, s54
	s_addc_u32 s23, s57, s55
	s_addc_u32 s35, s35, 0
	s_mul_i32 s54, s52, s54
	s_add_u32 s23, s23, s54
	s_addc_u32 s35, 0, s35
	s_add_u32 s23, s53, s23
	s_addc_u32 s35, s52, s35
	s_mul_i32 s53, s5, s35
	s_mul_hi_u32 s54, s5, s23
	s_mul_hi_u32 s52, s5, s35
	s_add_u32 s53, s54, s53
	s_addc_u32 s52, 0, s52
	s_mul_hi_u32 s55, s33, s23
	s_mul_i32 s23, s33, s23
	s_add_u32 s23, s53, s23
	s_mul_hi_u32 s54, s33, s35
	s_addc_u32 s23, s52, s55
	s_addc_u32 s52, s54, 0
	s_mul_i32 s35, s33, s35
	s_add_u32 s23, s23, s35
	s_addc_u32 s35, 0, s52
	s_mul_i32 s35, s7, s35
	s_mul_hi_u32 s54, s7, s23
	s_add_u32 s52, s23, 1
	s_add_u32 s53, s23, 2
	s_add_i32 s54, s54, s35
	s_mul_i32 s35, s7, s23
	s_sub_u32 s35, s5, s35
	s_subb_u32 s33, s33, s54
	s_sub_u32 s54, s35, s7
	s_subb_u32 s55, s33, 0
	s_cmp_ge_u32 s54, s7
	s_cselect_b32 s54, -1, 0
	s_cmp_eq_u32 s55, 0
	s_cselect_b32 s54, s54, -1
	s_cmp_lg_u32 s54, 0
	s_cselect_b32 s52, s53, s52
	s_cmp_ge_u32 s35, s7
	s_cselect_b32 s35, -1, 0
	s_cmp_eq_u32 s33, 0
	s_cselect_b32 s33, s35, -1
	s_cmp_lg_u32 s33, 0
	s_cselect_b32 s52, s52, s23
	s_cbranch_execnz .LBB28_43
.LBB28_42:
	v_rcp_iflag_f32_e32 v2, v3
	s_sub_i32 s8, 0, s7
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_readfirstlane_b32 s9, v2
	s_mul_i32 s8, s8, s9
	s_mul_hi_u32 s8, s9, s8
	s_add_i32 s9, s9, s8
	s_mul_hi_u32 s8, s5, s9
	s_mul_i32 s23, s8, s7
	s_sub_i32 s5, s5, s23
	s_add_i32 s9, s8, 1
	s_sub_i32 s23, s5, s7
	s_cmp_ge_u32 s5, s7
	s_cselect_b32 s8, s9, s8
	s_cselect_b32 s5, s23, s5
	s_add_i32 s9, s8, 1
	s_cmp_ge_u32 s5, s7
	s_cselect_b32 s52, s9, s8
.LBB28_43:
	s_mul_hi_u32 s5, s4, s20
	s_add_i32 s5, s5, s4
	s_lshr_b32 s5, s5, s21
	s_mul_i32 s5, s5, s22
	s_sub_i32 s5, s4, s5
	s_and_b32 s5, s5, 7
	s_sub_i32 s33, s4, s5
	s_mul_hi_u32 s4, s52, s20
	s_add_i32 s4, s4, s52
	s_lshr_b32 s4, s4, s21
	s_mul_i32 s4, s4, s22
	s_sub_i32 s4, s52, s4
	s_and_b32 s4, s4, 7
	s_sub_i32 s35, s52, s4
	s_mul_hi_u32 s4, s33, s20
	s_add_i32 s4, s33, s4
	s_lshr_b32 s4, s4, s21
	s_mul_i32 s4, s4, s22
	s_sub_i32 s7, s33, s4
	s_sub_i32 s4, s35, s4
	s_min_u32 s23, s22, s4
	s_cmp_lt_i32 s33, s35
	s_cselect_b64 s[54:55], -1, 0
	s_cmp_le_u32 s22, s4
	s_cselect_b64 s[4:5], -1, 0
	s_and_b64 s[4:5], s[54:55], s[4:5]
	s_andn2_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB28_72
; %bb.44:                               ; %.lr.ph808
	v_lshlrev_b32_e32 v3, 2, v75
	v_lshrrev_b32_e32 v4, 4, v0
	v_add_u32_e32 v4, v3, v4
	v_mul_lo_u32 v67, s25, v4
	s_movk_i32 s54, 0x104
	v_add_u32_e32 v11, 0, v3
	v_add_u32_e32 v3, 8, v75
	v_and_b32_e32 v2, 15, v0
	v_mad_u32_u24 v4, v4, s54, 0
	buffer_store_dword v3, off, s[68:71], 0 offset:4 ; 4-byte Folded Spill
	v_add_u32_e32 v3, 16, v75
	s_cmp_lg_u64 s[16:17], 0
	v_lshl_add_u32 v68, v2, 2, v4
	v_lshrrev_b32_e32 v2, 3, v0
	buffer_store_dword v3, off, s[68:71], 0 offset:8 ; 4-byte Folded Spill
	v_add_u32_e32 v3, 24, v75
	s_cselect_b64 s[4:5], -1, 0
	s_lshl_b32 s8, s25, 5
	v_lshl_add_u32 v2, v75, 3, v2
	buffer_store_dword v3, off, s[68:71], 0 offset:12 ; 4-byte Folded Spill
	v_add_u32_e32 v3, 32, v75
	v_and_b32_e32 v65, 7, v0
	v_add_u32_e32 v69, s8, v67
	v_mul_lo_u32 v72, s25, v2
	v_and_b32_e32 v4, 0x1ffc, v2
	v_lshlrev_b32_e32 v7, 5, v2
	v_add_u32_e32 v2, 64, v2
	v_mul_u32_u24_e32 v9, 36, v75
	buffer_store_dword v3, off, s[68:71], 0 offset:16 ; 4-byte Folded Spill
	v_add_u32_e32 v3, 40, v75
	v_add_u32_e32 v70, s8, v69
	v_lshlrev_b32_e32 v6, 2, v65
	v_and_b32_e32 v8, 0x3ffc, v2
	v_add_u32_e32 v10, 64, v0
	buffer_store_dword v3, off, s[68:71], 0 offset:20 ; 4-byte Folded Spill
	v_lshl_add_u32 v3, v9, 2, 0
	v_add_u32_e32 v71, s8, v70
	v_add3_u32 v4, 0, v4, v6
	v_add3_u32 v6, 0, v8, v6
	s_mul_i32 s8, s26, 36
	v_lshlrev_b32_e32 v8, 5, v0
	v_add_u32_e32 v80, 0x110, v3
	v_add_u32_e32 v82, 0x100, v3
	v_and_b32_e32 v3, 0x3fc, v10
	v_and_b32_e32 v10, 0x1fc, v0
	v_lshlrev_b32_e32 v5, 1, v65
	v_lshlrev_b32_e32 v2, 5, v2
	s_ashr_i32 s9, s8, 31
	v_mad_u32_u24 v9, v0, s54, 0
	v_add3_u32 v3, v8, v3, 0
	v_add3_u32 v8, v8, v10, 0
	v_lshl_add_u32 v63, v62, 2, 0
	v_bfe_u32 v64, v0, 3, 1
	v_mov_b32_e32 v1, 0
	v_lshl_add_u32 v73, s25, 6, v72
	s_lshl_b64 s[52:53], s[8:9], 2
	v_add_u32_e32 v81, 0x2900, v9
	v_add_u32_e32 v83, 0xb300, v3
	v_add_u32_e32 v84, 0xab00, v8
	v_add_u32_e32 v85, 0x2980, v9
	v_add_u32_e32 v86, 0xb310, v3
	v_add_u32_e32 v87, 0xab10, v8
	v_lshlrev_b32_e32 v88, 1, v5
	v_add_u32_e32 v89, v4, v7
	v_add_u32_e32 v90, v6, v2
	s_movk_i32 s9, 0x1000
	buffer_store_dword v11, off, s[68:71], 0 ; 4-byte Folded Spill
	s_branch .LBB28_47
.LBB28_45:                              ; %Flow2613
                                        ;   in Loop: Header=BB28_47 Depth=1
	s_or_b64 exec, exec, s[54:55]
.LBB28_46:                              ; %_ZL22mul_mat_q_process_tileIL9ggml_type8ELi64ELb0ELb0EEvPKciPKiS4_PfS5_iiiiiii.exit602
                                        ;   in Loop: Header=BB28_47 Depth=1
	s_add_i32 s7, s33, s22
	s_mul_hi_u32 s23, s7, s20
	s_add_i32 s7, s7, s23
	s_lshr_b32 s7, s7, s21
	s_mul_i32 s33, s7, s22
	s_sub_i32 s56, s35, s33
	s_cmp_gt_i32 s35, s33
	s_cselect_b64 s[54:55], -1, 0
	s_cmp_le_u32 s22, s56
	s_cselect_b64 s[58:59], -1, 0
	s_and_b64 s[58:59], s[54:55], s[58:59]
	s_mov_b32 s7, 0
	s_and_b64 vcc, exec, s[58:59]
	s_mov_b32 s23, s22
	s_cbranch_vccz .LBB28_71
.LBB28_47:                              ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB28_56 Depth 2
                                        ;       Child Loop BB28_57 Depth 3
                                        ;       Child Loop BB28_59 Depth 3
	s_mul_hi_u32 s54, s33, s20
	s_add_i32 s54, s33, s54
	s_lshr_b32 s54, s54, s21
	s_mul_hi_u32 s55, s54, s50
	s_add_i32 s55, s54, s55
	s_lshr_b32 s55, s55, s51
	s_mul_i32 s56, s55, s34
	s_sub_i32 s56, s54, s56
	s_mul_hi_u32 s54, s55, s36
	s_add_i32 s54, s55, s54
	s_lshr_b32 s57, s54, s37
	s_mul_i32 s54, s57, s38
	s_sub_i32 s54, s55, s54
	s_mul_hi_u32 s55, s57, s44
	s_add_i32 s55, s57, s55
	s_lshr_b32 s60, s55, s45
	s_mul_i32 s55, s60, s46
	s_sub_i32 s61, s57, s55
	s_lshl_b32 s58, s56, 6
	s_andn2_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB28_53
; %bb.48:                               ;   in Loop: Header=BB28_47 Depth=1
	s_ashr_i32 s55, s54, 31
	s_lshl_b64 s[56:57], s[54:55], 2
	s_add_u32 s56, s18, s56
	s_addc_u32 s57, s19, s57
	global_load_dwordx2 v[2:3], v1, s[56:57]
	s_mov_b64 s[56:57], 0
	s_mov_b32 s55, 0
	s_mov_b32 s62, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s59, v2
	v_subrev_u32_e32 v99, s59, v3
	v_cmp_lt_i32_e32 vcc, s58, v99
	s_cbranch_vccz .LBB28_52
; %bb.49:                               ;   in Loop: Header=BB28_47 Depth=1
	s_barrier
	s_and_saveexec_b64 s[56:57], s[0:1]
	s_cbranch_execz .LBB28_51
; %bb.50:                               ; %.lr.ph804
                                        ;   in Loop: Header=BB28_47 Depth=1
	v_or_b32_e32 v2, s58, v0
	v_add_u32_e32 v2, s59, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_mov_b32_e32 v4, s17
	v_add_co_u32_e32 v2, vcc, s16, v2
	v_addc_co_u32_e32 v3, vcc, v4, v3, vcc
	global_load_dword v2, v[2:3], off
	s_waitcnt vmcnt(0)
	ds_write_b32 v63, v2
.LBB28_51:                              ; %.critedge465
                                        ;   in Loop: Header=BB28_47 Depth=1
	s_or_b64 exec, exec, s[56:57]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_mov_b64 s[56:57], -1
	s_mov_b32 s62, s59
.LBB28_52:                              ; %Flow2617
                                        ;   in Loop: Header=BB28_47 Depth=1
	s_mov_b32 s59, 0
	s_and_b64 vcc, exec, s[56:57]
	s_cbranch_vccz .LBB28_46
	s_branch .LBB28_54
.LBB28_53:                              ;   in Loop: Header=BB28_47 Depth=1
	s_mul_i32 s55, s61, s48
	s_mul_i32 s56, s54, s40
	s_mul_i32 s57, s54, s41
	s_mul_i32 s59, s58, s27
	s_add_i32 s55, s55, s56
	s_mul_i32 s56, s61, s49
	s_add_i32 s57, s57, s59
	v_mov_b32_e32 v99, s24
	s_add_i32 s59, s57, s56
	s_mov_b32 s62, 0
	s_cbranch_execz .LBB28_46
.LBB28_54:                              ;   in Loop: Header=BB28_47 Depth=1
	s_lshl_b32 s56, s60, 7
	v_mov_b32_e32 v91, 0
	s_cmp_ge_i32 s7, s23
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v106, 0
	s_cbranch_scc1 .LBB28_62
; %bb.55:                               ; %.lr.ph.i513
                                        ;   in Loop: Header=BB28_47 Depth=1
	s_add_i32 s57, s62, s58
	s_mul_i32 s57, s57, 36
	s_add_i32 s62, s57, s55
	s_ashr_i32 s63, s62, 31
	s_lshl_b64 s[62:63], s[62:63], 2
	s_add_u32 s57, s14, s62
	s_mul_hi_u32 s62, s54, s10
	s_addc_u32 s60, s15, s63
	s_add_i32 s54, s54, s62
	s_mul_hi_u32 s62, s61, s42
	s_lshr_b32 s54, s54, s11
	s_add_i32 s61, s61, s62
	s_mul_i32 s55, s56, s25
	s_mul_i32 s54, s54, s39
	s_lshr_b32 s61, s61, s43
	s_mul_i32 s61, s61, s47
	s_add_i32 s54, s54, s55
	v_mov_b32_e32 v1, v75
	s_add_i32 s61, s54, s61
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v91, 0
.LBB28_56:                              ;   Parent Loop BB28_47 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB28_57 Depth 3
                                        ;       Child Loop BB28_59 Depth 3
	s_add_i32 s54, s61, s7
	s_mul_hi_i32 s55, s54, 34
	s_mul_i32 s54, s54, 34
	s_add_u32 s54, s12, s54
	s_addc_u32 s55, s13, s55
	v_mad_u64_u32 v[2:3], s[62:63], v64, 34, s[54:55]
	v_lshlrev_b32_e32 v108, 2, v62
	v_add_u32_e32 v27, 0x6800, v68
	v_mad_i64_i32 v[4:5], s[62:63], v67, 34, v[2:3]
	v_mad_i64_i32 v[6:7], s[62:63], v69, 34, v[2:3]
	v_add_co_u32_e32 v4, vcc, v4, v88
	v_addc_co_u32_e32 v5, vcc, 0, v5, vcc
	v_add_co_u32_e32 v6, vcc, v6, v88
	v_addc_co_u32_e32 v7, vcc, 0, v7, vcc
	v_mad_i64_i32 v[8:9], s[62:63], v70, 34, v[2:3]
	global_load_dword v10, v[4:5], off offset:2
	global_load_dword v11, v[4:5], off offset:70
	global_load_dword v12, v[4:5], off offset:138
	global_load_dword v13, v[4:5], off offset:206
	global_load_dword v14, v[6:7], off offset:2
	global_load_dword v15, v[6:7], off offset:70
	global_load_dword v16, v[6:7], off offset:138
	global_load_dword v17, v[6:7], off offset:206
	v_mad_u64_u32 v[6:7], s[54:55], v65, 34, s[54:55]
	v_add_co_u32_e32 v4, vcc, v8, v88
	v_addc_co_u32_e32 v5, vcc, 0, v9, vcc
	v_mad_i64_i32 v[8:9], s[54:55], v72, 34, v[6:7]
	v_mad_i64_i32 v[6:7], s[54:55], v73, 34, v[6:7]
	s_ashr_i32 s54, s7, 31
	s_lshr_b32 s54, s54, 30
	s_add_i32 s54, s7, s54
	v_mad_i64_i32 v[2:3], s[62:63], v71, 34, v[2:3]
	s_ashr_i32 s54, s54, 2
	s_mul_i32 s54, s8, s54
	s_ashr_i32 s55, s54, 31
	s_lshl_b64 s[54:55], s[54:55], 2
	v_add_co_u32_e32 v2, vcc, v2, v88
	s_add_u32 s54, s57, s54
	v_addc_co_u32_e32 v3, vcc, 0, v3, vcc
	s_addc_u32 s55, s60, s55
	global_load_dword v18, v[4:5], off offset:2
	global_load_dword v19, v[4:5], off offset:70
	global_load_dword v20, v[4:5], off offset:138
	global_load_dword v21, v[4:5], off offset:206
	global_load_dword v22, v[2:3], off offset:2
	global_load_dword v23, v[2:3], off offset:70
	global_load_dword v24, v[2:3], off offset:138
	global_load_dword v25, v[2:3], off offset:206
	s_nop 0
	global_load_ushort v8, v[8:9], off
	s_nop 0
	global_load_ushort v6, v[6:7], off
	v_mov_b32_e32 v2, s55
	v_add_co_u32_e32 v4, vcc, s54, v108
	v_addc_co_u32_e32 v5, vcc, 0, v2, vcc
	v_add_co_u32_e32 v2, vcc, s9, v4
	v_addc_co_u32_e32 v3, vcc, 0, v5, vcc
	v_add_co_u32_e32 v4, vcc, 0x2000, v4
	v_addc_co_u32_e32 v5, vcc, 0, v5, vcc
	global_load_dword v7, v108, s[54:55]
	global_load_dword v9, v108, s[54:55] offset:2048
	global_load_dword v26, v[2:3], off
	s_nop 0
	global_load_dword v2, v[2:3], off offset:2048
	s_nop 0
	global_load_dword v3, v[4:5], off
	v_add_u32_e32 v4, 0x2800, v68
	v_add_u32_e32 v5, 0x4800, v68
	s_mov_b32 s62, -8
	v_mov_b32_e32 v109, v84
	v_mov_b32_e32 v110, v83
	v_mov_b32_e32 v111, v82
	v_mov_b32_e32 v112, v81
	v_mov_b32_e32 v113, v80
	s_waitcnt vmcnt(21)
	ds_write2_b32 v4, v10, v11 offset0:64 offset1:80
	s_waitcnt vmcnt(19)
	ds_write2_b32 v4, v12, v13 offset0:96 offset1:112
	s_waitcnt vmcnt(17)
	ds_write2_b32 v5, v14, v15 offset0:96 offset1:112
	s_waitcnt vmcnt(15)
	ds_write2_b32 v5, v16, v17 offset0:128 offset1:144
	s_waitcnt vmcnt(13)
	ds_write2_b32 v27, v18, v19 offset0:128 offset1:144
	s_waitcnt vmcnt(11)
	ds_write2_b32 v27, v20, v21 offset0:160 offset1:176
	v_add_u32_e32 v4, 0x8800, v68
	s_waitcnt vmcnt(6)
	v_cvt_f32_f16_e32 v5, v8
	s_waitcnt vmcnt(5)
	v_cvt_f32_f16_e32 v6, v6
	ds_write2_b32 v4, v22, v23 offset0:160 offset1:176
	ds_write2_b32 v4, v24, v25 offset0:192 offset1:208
	ds_write_b32 v89, v5 offset:43776
	ds_write_b32 v90, v6 offset:43776
	s_waitcnt vmcnt(3)
	ds_write2st64_b32 v63, v7, v9 offset0:1 offset1:9
	s_waitcnt vmcnt(1)
	ds_write2st64_b32 v63, v26, v2 offset0:17 offset1:25
	s_waitcnt vmcnt(0)
	ds_write_b32 v63, v3 offset:8448
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB28_57:                              ;   Parent Loop BB28_47 Depth=1
                                        ;     Parent Loop BB28_56 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ds_read2_b32 v[28:29], v113 offset1:1
	ds_read2_b32 v[12:13], v113 offset0:2 offset1:3
	ds_read2_b32 v[8:9], v113 offset0:4 offset1:5
	ds_read2_b32 v[2:3], v113 offset0:6 offset1:7
	ds_read2_b32 v[14:15], v112 offset1:1
	ds_read2_b32 v[10:11], v112 offset0:2 offset1:3
	ds_read2_b32 v[6:7], v112 offset0:4 offset1:5
	ds_read2_b32 v[4:5], v112 offset0:6 offset1:7
	v_add_u32_e32 v16, 0x4100, v112
	v_add_u32_e32 v17, 0x4108, v112
	v_add_u32_e32 v18, 0x4110, v112
	v_add_u32_e32 v19, 0x4118, v112
	v_add_u32_e32 v20, 0x480, v113
	v_add_u32_e32 v21, 0x488, v113
	v_add_u32_e32 v22, 0x490, v113
	v_add_u32_e32 v23, 0x498, v113
	v_add_u32_e32 v24, 0x900, v113
	v_add_u32_e32 v25, 0x908, v113
	v_add_u32_e32 v30, 0x910, v113
	v_add_u32_e32 v31, 0x918, v113
	v_add_u32_e32 v36, 0xd80, v113
	v_add_u32_e32 v38, 0xd88, v113
	v_add_u32_e32 v39, 0xd90, v113
	v_add_u32_e32 v42, 0xd98, v113
	v_add_u32_e32 v43, 0x1200, v113
	v_add_u32_e32 v48, 0x1208, v113
	v_add_u32_e32 v49, 0x1210, v113
	v_add_u32_e32 v60, 0x1218, v113
	ds_read2_b32 v[58:59], v16 offset1:1
	ds_read2_b32 v[46:47], v17 offset1:1
	ds_read2_b32 v[26:27], v18 offset1:1
	ds_read2_b32 v[16:17], v19 offset1:1
	ds_read2_b32 v[32:33], v20 offset1:1
	ds_read2_b32 v[50:51], v21 offset1:1
	ds_read2_b32 v[44:45], v22 offset1:1
	ds_read2_b32 v[18:19], v23 offset1:1
	ds_read2_b32 v[34:35], v24 offset1:1
	ds_read2_b32 v[52:53], v25 offset1:1
	ds_read2_b32 v[40:41], v30 offset1:1
	ds_read2_b32 v[20:21], v31 offset1:1
	ds_read2_b32 v[36:37], v36 offset1:1
	ds_read2_b32 v[54:55], v38 offset1:1
	ds_read2_b32 v[38:39], v39 offset1:1
	ds_read2_b32 v[22:23], v42 offset1:1
	ds_read2_b32 v[42:43], v43 offset1:1
	ds_read2_b32 v[56:57], v48 offset1:1
	ds_read2_b32 v[30:31], v49 offset1:1
	ds_read2_b32 v[24:25], v60 offset1:1
	s_waitcnt lgkmcnt(14)
	v_dot4_i32_i8 v48, v14, v28, 0
	v_dot4_i32_i8 v28, v58, v28, 0
	v_dot4_i32_i8 v114, v15, v29, v48
	v_dot4_i32_i8 v115, v59, v29, v28
	v_dot4_i32_i8 v28, v14, v32, 0
	v_dot4_i32_i8 v29, v58, v32, 0
	v_dot4_i32_i8 v116, v15, v33, v28
	v_dot4_i32_i8 v117, v59, v33, v29
	s_waitcnt lgkmcnt(11)
	v_dot4_i32_i8 v28, v14, v34, 0
	v_dot4_i32_i8 v29, v58, v34, 0
	v_dot4_i32_i8 v118, v15, v35, v28
	v_dot4_i32_i8 v119, v59, v35, v29
	s_waitcnt lgkmcnt(7)
	v_dot4_i32_i8 v28, v14, v36, 0
	v_dot4_i32_i8 v29, v58, v36, 0
	v_dot4_i32_i8 v120, v15, v37, v28
	v_dot4_i32_i8 v121, v59, v37, v29
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v28, v14, v42, 0
	v_dot4_i32_i8 v29, v58, v42, 0
	v_add_u32_e32 v61, 0x1680, v113
	v_dot4_i32_i8 v122, v15, v43, v28
	v_dot4_i32_i8 v123, v59, v43, v29
	v_add_u32_e32 v28, 0x1688, v113
	v_add_u32_e32 v29, 0x1690, v113
	v_add_u32_e32 v36, 0x1698, v113
	ds_read2_b32 v[34:35], v61 offset1:1
	ds_read2_b32 v[60:61], v28 offset1:1
	ds_read2_b32 v[32:33], v29 offset1:1
	ds_read2_b32 v[28:29], v36 offset1:1
	v_add_u32_e32 v48, 0x1b18, v113
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v36, v14, v34, 0
	v_dot4_i32_i8 v34, v58, v34, 0
	v_dot4_i32_i8 v124, v15, v35, v36
	v_dot4_i32_i8 v125, v59, v35, v34
	v_add_u32_e32 v34, 0x1b00, v113
	v_add_u32_e32 v35, 0x1b08, v113
	v_add_u32_e32 v36, 0x1b10, v113
	ds_read2_b32 v[42:43], v34 offset1:1
	ds_read2_b32 v[126:127], v35 offset1:1
	ds_read2_b32 v[36:37], v36 offset1:1
	ds_read2_b32 v[34:35], v48 offset1:1
	v_add_u32_e32 v74, 0x1f98, v113
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v48, v14, v42, 0
	v_dot4_i32_i8 v42, v58, v42, 0
	v_dot4_i32_i8 v75, v15, v43, v48
	v_dot4_i32_i8 v66, v59, v43, v42
	v_add_u32_e32 v42, 0x1f80, v113
	v_add_u32_e32 v43, 0x1f88, v113
	v_add_u32_e32 v48, 0x1f90, v113
	ds_read2_b32 v[78:79], v42 offset1:1
	ds_read2_b32 v[76:77], v43 offset1:1
	ds_read2_b32 v[48:49], v48 offset1:1
	ds_read2_b32 v[42:43], v74 offset1:1
	s_add_i32 s62, s62, 8
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v14, v14, v78, 0
	v_dot4_i32_i8 v14, v15, v79, v14
	v_dot4_i32_i8 v15, v58, v78, 0
	v_dot4_i32_i8 v58, v10, v12, v114
	v_dot4_i32_i8 v12, v46, v12, v115
	v_dot4_i32_i8 v15, v59, v79, v15
	v_dot4_i32_i8 v59, v47, v13, v12
	v_dot4_i32_i8 v12, v10, v50, v116
	v_dot4_i32_i8 v58, v11, v13, v58
	v_dot4_i32_i8 v13, v46, v50, v117
	v_dot4_i32_i8 v74, v11, v51, v12
	v_dot4_i32_i8 v12, v10, v52, v118
	v_dot4_i32_i8 v78, v47, v51, v13
	v_dot4_i32_i8 v13, v46, v52, v119
	v_dot4_i32_i8 v79, v11, v53, v12
	v_dot4_i32_i8 v12, v10, v54, v120
	v_dot4_i32_i8 v114, v47, v53, v13
	v_dot4_i32_i8 v13, v46, v54, v121
	v_dot4_i32_i8 v115, v11, v55, v12
	v_dot4_i32_i8 v12, v10, v56, v122
	v_dot4_i32_i8 v116, v47, v55, v13
	v_dot4_i32_i8 v13, v46, v56, v123
	v_dot4_i32_i8 v54, v11, v57, v12
	v_dot4_i32_i8 v12, v10, v60, v124
	v_dot4_i32_i8 v55, v47, v57, v13
	v_dot4_i32_i8 v13, v46, v60, v125
	v_dot4_i32_i8 v52, v11, v61, v12
	v_dot4_i32_i8 v12, v10, v126, v75
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v10, v10, v76, v14
	v_dot4_i32_i8 v53, v47, v61, v13
	v_dot4_i32_i8 v13, v46, v126, v66
	v_dot4_i32_i8 v50, v11, v127, v12
	v_dot4_i32_i8 v10, v11, v77, v10
	v_dot4_i32_i8 v11, v46, v76, v15
	v_dot4_i32_i8 v12, v6, v8, v58
	v_dot4_i32_i8 v8, v26, v8, v59
	v_dot4_i32_i8 v51, v47, v127, v13
	v_dot4_i32_i8 v11, v47, v77, v11
	v_dot4_i32_i8 v12, v7, v9, v12
	v_dot4_i32_i8 v8, v27, v9, v8
	v_dot4_i32_i8 v9, v6, v44, v74
	v_dot4_i32_i8 v15, v26, v44, v78
	v_dot4_i32_i8 v9, v7, v45, v9
	v_dot4_i32_i8 v15, v27, v45, v15
	v_dot4_i32_i8 v44, v6, v40, v79
	v_dot4_i32_i8 v45, v26, v40, v114
	v_dot4_i32_i8 v56, v6, v38, v115
	v_dot4_i32_i8 v60, v26, v38, v116
	v_dot4_i32_i8 v54, v6, v30, v54
	v_dot4_i32_i8 v30, v26, v30, v55
	v_dot4_i32_i8 v52, v6, v32, v52
	v_dot4_i32_i8 v32, v26, v32, v53
	v_dot4_i32_i8 v50, v6, v36, v50
	v_dot4_i32_i8 v36, v26, v36, v51
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v6, v6, v48, v10
	v_dot4_i32_i8 v10, v26, v48, v11
	v_dot4_i32_i8 v40, v7, v41, v44
	v_dot4_i32_i8 v41, v27, v41, v45
	v_dot4_i32_i8 v38, v7, v39, v56
	v_dot4_i32_i8 v39, v27, v39, v60
	v_dot4_i32_i8 v54, v7, v31, v54
	v_dot4_i32_i8 v30, v27, v31, v30
	v_dot4_i32_i8 v52, v7, v33, v52
	v_dot4_i32_i8 v32, v27, v33, v32
	v_dot4_i32_i8 v50, v7, v37, v50
	v_dot4_i32_i8 v36, v27, v37, v36
	v_dot4_i32_i8 v11, v4, v2, v12
	v_dot4_i32_i8 v6, v7, v49, v6
	v_dot4_i32_i8 v7, v27, v49, v10
	ds_read_b32 v57, v111
	ds_read_b32 v13, v109
	ds_read_b32 v14, v110
	ds_read_b32 v59, v111 offset:1152
	ds_read_b32 v58, v111 offset:2304
	ds_read_b32 v47, v111 offset:3456
	ds_read_b32 v46, v111 offset:4608
	ds_read_b32 v45, v111 offset:5760
	ds_read_b32 v44, v111 offset:6912
	ds_read_b32 v56, v111 offset:8064
	v_dot4_i32_i8 v10, v5, v3, v11
	v_dot4_i32_i8 v2, v16, v2, v8
	v_dot4_i32_i8 v8, v4, v18, v9
	v_dot4_i32_i8 v9, v16, v18, v15
	v_dot4_i32_i8 v11, v4, v20, v40
	v_dot4_i32_i8 v12, v16, v20, v41
	v_dot4_i32_i8 v15, v4, v22, v38
	v_dot4_i32_i8 v18, v16, v22, v39
	v_dot4_i32_i8 v20, v4, v24, v54
	v_dot4_i32_i8 v22, v16, v24, v30
	v_dot4_i32_i8 v24, v4, v28, v52
	v_dot4_i32_i8 v26, v16, v28, v32
	v_dot4_i32_i8 v27, v4, v34, v50
	v_dot4_i32_i8 v28, v16, v34, v36
	s_waitcnt lgkmcnt(10)
	v_dot4_i32_i8 v4, v4, v42, v6
	v_dot4_i32_i8 v6, v16, v42, v7
	v_cvt_f32_i32_e32 v7, v10
	v_dot4_i32_i8 v2, v17, v3, v2
	v_dot4_i32_i8 v3, v5, v19, v8
	v_dot4_i32_i8 v8, v17, v19, v9
	v_dot4_i32_i8 v9, v5, v21, v11
	v_dot4_i32_i8 v10, v17, v21, v12
	v_dot4_i32_i8 v11, v5, v23, v15
	v_dot4_i32_i8 v12, v17, v23, v18
	v_dot4_i32_i8 v15, v5, v25, v20
	v_dot4_i32_i8 v16, v17, v25, v22
	v_dot4_i32_i8 v18, v5, v29, v24
	v_dot4_i32_i8 v19, v17, v29, v26
	v_dot4_i32_i8 v20, v5, v35, v27
	v_dot4_i32_i8 v21, v17, v35, v28
	v_dot4_i32_i8 v4, v5, v43, v4
	v_dot4_i32_i8 v5, v17, v43, v6
	v_cvt_f32_i32_e32 v2, v2
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v6, v8
	v_cvt_f32_i32_e32 v8, v9
	v_cvt_f32_i32_e32 v9, v10
	v_cvt_f32_i32_e32 v10, v11
	v_cvt_f32_i32_e32 v11, v12
	v_cvt_f32_i32_e32 v12, v15
	v_cvt_f32_i32_e32 v15, v16
	v_cvt_f32_i32_e32 v16, v18
	v_cvt_f32_i32_e32 v17, v19
	v_cvt_f32_i32_e32 v18, v20
	v_cvt_f32_i32_e32 v19, v21
	v_cvt_f32_i32_e32 v4, v4
	v_cvt_f32_i32_e32 v5, v5
	s_waitcnt lgkmcnt(8)
	v_mul_f32_e32 v60, v13, v57
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v57, v14, v57
	s_waitcnt lgkmcnt(6)
	v_mul_f32_e32 v31, v59, v13
	v_mul_f32_e32 v55, v59, v14
	s_waitcnt lgkmcnt(5)
	v_mul_f32_e32 v59, v58, v13
	v_mul_f32_e32 v58, v58, v14
	s_waitcnt lgkmcnt(4)
	v_mul_f32_e32 v33, v47, v13
	v_mul_f32_e32 v47, v47, v14
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v53, v46, v13
	v_mul_f32_e32 v46, v46, v14
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v37, v45, v13
	v_mul_f32_e32 v45, v45, v14
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v51, v44, v13
	v_mul_f32_e32 v44, v44, v14
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v13, v56, v13
	v_mul_f32_e32 v14, v56, v14
	v_add_u32_e32 v113, 32, v113
	v_add_u32_e32 v112, 32, v112
	v_add_u32_e32 v111, 4, v111
	v_add_u32_e32 v110, 4, v110
	v_add_u32_e32 v109, 4, v109
	s_cmp_lt_u32 s62, 24
	v_fmac_f32_e32 v106, v60, v7
	v_fmac_f32_e32 v107, v57, v2
	v_fmac_f32_e32 v105, v31, v3
	v_fmac_f32_e32 v104, v55, v6
	v_fmac_f32_e32 v103, v59, v8
	v_fmac_f32_e32 v102, v58, v9
	v_fmac_f32_e32 v101, v33, v10
	v_fmac_f32_e32 v100, v47, v11
	v_fmac_f32_e32 v98, v53, v12
	v_fmac_f32_e32 v97, v46, v15
	v_fmac_f32_e32 v96, v37, v16
	v_fmac_f32_e32 v95, v45, v17
	v_fmac_f32_e32 v94, v51, v18
	v_fmac_f32_e32 v93, v44, v19
	v_fmac_f32_e32 v92, v13, v4
	v_fmac_f32_e32 v91, v14, v5
	s_cbranch_scc1 .LBB28_57
; %bb.58:                               ; %_ZL22vec_dot_q8_0_q8_1_dp4aILi64ELi128ELi8EEvPKiS1_Pfi.exit.i568
                                        ;   in Loop: Header=BB28_56 Depth=2
	s_add_u32 s54, s54, s52
	s_addc_u32 s55, s55, s53
	v_mov_b32_e32 v2, s55
	v_add_co_u32_e32 v4, vcc, s54, v108
	v_addc_co_u32_e32 v5, vcc, 0, v2, vcc
	v_add_co_u32_e32 v2, vcc, s9, v4
	v_addc_co_u32_e32 v3, vcc, 0, v5, vcc
	v_add_co_u32_e32 v4, vcc, 0x2000, v4
	s_barrier
	v_addc_co_u32_e32 v5, vcc, 0, v5, vcc
	global_load_dword v6, v108, s[54:55]
	global_load_dword v7, v108, s[54:55] offset:2048
	global_load_dword v8, v[2:3], off
	s_nop 0
	global_load_dword v2, v[2:3], off offset:2048
	s_nop 0
	global_load_dword v3, v[4:5], off
	s_mov_b32 s54, -8
	v_mov_b32_e32 v108, v82
	v_mov_b32_e32 v109, v87
	v_mov_b32_e32 v110, v86
	v_mov_b32_e32 v111, v85
	v_mov_b32_e32 v112, v80
	s_waitcnt vmcnt(3)
	ds_write2st64_b32 v63, v6, v7 offset0:1 offset1:9
	s_waitcnt vmcnt(1)
	ds_write2st64_b32 v63, v8, v2 offset0:17 offset1:25
	s_waitcnt vmcnt(0)
	ds_write_b32 v63, v3 offset:8448
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB28_59:                              ;   Parent Loop BB28_47 Depth=1
                                        ;     Parent Loop BB28_56 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ds_read2_b32 v[24:25], v112 offset1:1
	ds_read2_b32 v[12:13], v112 offset0:2 offset1:3
	ds_read2_b32 v[8:9], v112 offset0:4 offset1:5
	ds_read2_b32 v[2:3], v112 offset0:6 offset1:7
	ds_read2_b32 v[14:15], v111 offset1:1
	ds_read2_b32 v[10:11], v111 offset0:2 offset1:3
	ds_read2_b32 v[6:7], v111 offset0:4 offset1:5
	ds_read2_b32 v[4:5], v111 offset0:6 offset1:7
	v_add_u32_e32 v16, 0x4100, v111
	v_add_u32_e32 v17, 0x4108, v111
	v_add_u32_e32 v18, 0x4110, v111
	v_add_u32_e32 v19, 0x4118, v111
	v_add_u32_e32 v20, 0x480, v112
	v_add_u32_e32 v21, 0x488, v112
	v_add_u32_e32 v22, 0x490, v112
	v_add_u32_e32 v23, 0x498, v112
	v_add_u32_e32 v26, 0x900, v112
	v_add_u32_e32 v27, 0x908, v112
	v_add_u32_e32 v30, 0x910, v112
	v_add_u32_e32 v31, 0x918, v112
	v_add_u32_e32 v36, 0xd80, v112
	v_add_u32_e32 v38, 0xd88, v112
	v_add_u32_e32 v39, 0xd90, v112
	v_add_u32_e32 v52, 0xd98, v112
	v_add_u32_e32 v53, 0x1200, v112
	v_add_u32_e32 v58, 0x1208, v112
	v_add_u32_e32 v59, 0x1210, v112
	v_add_u32_e32 v60, 0x1218, v112
	ds_read2_b32 v[54:55], v16 offset1:1
	ds_read2_b32 v[44:45], v17 offset1:1
	ds_read2_b32 v[28:29], v18 offset1:1
	ds_read2_b32 v[16:17], v19 offset1:1
	ds_read2_b32 v[32:33], v20 offset1:1
	ds_read2_b32 v[46:47], v21 offset1:1
	ds_read2_b32 v[42:43], v22 offset1:1
	ds_read2_b32 v[18:19], v23 offset1:1
	ds_read2_b32 v[34:35], v26 offset1:1
	ds_read2_b32 v[48:49], v27 offset1:1
	ds_read2_b32 v[40:41], v30 offset1:1
	ds_read2_b32 v[20:21], v31 offset1:1
	ds_read2_b32 v[36:37], v36 offset1:1
	ds_read2_b32 v[50:51], v38 offset1:1
	ds_read2_b32 v[38:39], v39 offset1:1
	ds_read2_b32 v[22:23], v52 offset1:1
	ds_read2_b32 v[56:57], v53 offset1:1
	ds_read2_b32 v[52:53], v58 offset1:1
	ds_read2_b32 v[30:31], v59 offset1:1
	ds_read2_b32 v[26:27], v60 offset1:1
	s_waitcnt lgkmcnt(14)
	v_dot4_i32_i8 v58, v14, v24, 0
	v_dot4_i32_i8 v24, v54, v24, 0
	v_dot4_i32_i8 v113, v15, v25, v58
	v_dot4_i32_i8 v114, v55, v25, v24
	v_dot4_i32_i8 v24, v14, v32, 0
	v_dot4_i32_i8 v25, v54, v32, 0
	v_dot4_i32_i8 v115, v15, v33, v24
	v_dot4_i32_i8 v116, v55, v33, v25
	s_waitcnt lgkmcnt(11)
	v_dot4_i32_i8 v24, v14, v34, 0
	v_dot4_i32_i8 v25, v54, v34, 0
	v_dot4_i32_i8 v117, v15, v35, v24
	v_dot4_i32_i8 v118, v55, v35, v25
	s_waitcnt lgkmcnt(7)
	v_dot4_i32_i8 v24, v14, v36, 0
	v_dot4_i32_i8 v25, v54, v36, 0
	v_dot4_i32_i8 v119, v15, v37, v24
	v_dot4_i32_i8 v120, v55, v37, v25
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v24, v14, v56, 0
	v_dot4_i32_i8 v25, v54, v56, 0
	v_add_u32_e32 v61, 0x1680, v112
	v_dot4_i32_i8 v121, v15, v57, v24
	v_dot4_i32_i8 v122, v55, v57, v25
	v_add_u32_e32 v24, 0x1688, v112
	v_add_u32_e32 v25, 0x1690, v112
	v_add_u32_e32 v36, 0x1698, v112
	ds_read2_b32 v[34:35], v61 offset1:1
	ds_read2_b32 v[60:61], v24 offset1:1
	ds_read2_b32 v[32:33], v25 offset1:1
	ds_read2_b32 v[24:25], v36 offset1:1
	v_add_u32_e32 v58, 0x1b18, v112
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v36, v14, v34, 0
	v_dot4_i32_i8 v34, v54, v34, 0
	v_dot4_i32_i8 v123, v15, v35, v36
	v_dot4_i32_i8 v124, v55, v35, v34
	v_add_u32_e32 v34, 0x1b00, v112
	v_add_u32_e32 v35, 0x1b08, v112
	v_add_u32_e32 v36, 0x1b10, v112
	ds_read2_b32 v[56:57], v34 offset1:1
	ds_read2_b32 v[76:77], v35 offset1:1
	ds_read2_b32 v[36:37], v36 offset1:1
	ds_read2_b32 v[34:35], v58 offset1:1
	v_add_u32_e32 v75, 0x1f98, v112
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v58, v14, v56, 0
	v_dot4_i32_i8 v56, v54, v56, 0
	v_dot4_i32_i8 v66, v15, v57, v58
	v_dot4_i32_i8 v74, v55, v57, v56
	v_add_u32_e32 v56, 0x1f80, v112
	v_add_u32_e32 v57, 0x1f88, v112
	v_add_u32_e32 v58, 0x1f90, v112
	ds_read2_b32 v[78:79], v56 offset1:1
	ds_read2_b32 v[125:126], v57 offset1:1
	ds_read2_b32 v[58:59], v58 offset1:1
	ds_read2_b32 v[56:57], v75 offset1:1
	s_add_i32 s54, s54, 8
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v14, v14, v78, 0
	v_dot4_i32_i8 v14, v15, v79, v14
	v_dot4_i32_i8 v15, v54, v78, 0
	v_dot4_i32_i8 v54, v10, v12, v113
	v_dot4_i32_i8 v12, v44, v12, v114
	v_dot4_i32_i8 v54, v11, v13, v54
	v_dot4_i32_i8 v12, v45, v13, v12
	v_dot4_i32_i8 v13, v10, v46, v115
	v_dot4_i32_i8 v46, v44, v46, v116
	v_dot4_i32_i8 v13, v11, v47, v13
	v_dot4_i32_i8 v46, v45, v47, v46
	v_dot4_i32_i8 v47, v10, v48, v117
	v_dot4_i32_i8 v48, v44, v48, v118
	v_dot4_i32_i8 v47, v11, v49, v47
	v_dot4_i32_i8 v48, v45, v49, v48
	v_dot4_i32_i8 v49, v10, v50, v119
	v_dot4_i32_i8 v50, v44, v50, v120
	v_dot4_i32_i8 v49, v11, v51, v49
	v_dot4_i32_i8 v50, v45, v51, v50
	v_dot4_i32_i8 v51, v10, v52, v121
	v_dot4_i32_i8 v52, v44, v52, v122
	v_dot4_i32_i8 v15, v55, v79, v15
	v_dot4_i32_i8 v51, v11, v53, v51
	v_dot4_i32_i8 v52, v45, v53, v52
	v_dot4_i32_i8 v53, v10, v60, v123
	v_dot4_i32_i8 v55, v44, v60, v124
	v_dot4_i32_i8 v60, v10, v76, v66
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v10, v10, v125, v14
	v_dot4_i32_i8 v53, v11, v61, v53
	v_dot4_i32_i8 v55, v45, v61, v55
	v_dot4_i32_i8 v61, v44, v76, v74
	v_dot4_i32_i8 v60, v11, v77, v60
	v_dot4_i32_i8 v10, v11, v126, v10
	v_dot4_i32_i8 v11, v44, v125, v15
	v_dot4_i32_i8 v61, v45, v77, v61
	v_dot4_i32_i8 v11, v45, v126, v11
	v_dot4_i32_i8 v45, v6, v8, v54
	v_dot4_i32_i8 v45, v7, v9, v45
	v_dot4_i32_i8 v8, v28, v8, v12
	v_dot4_i32_i8 v13, v6, v42, v13
	v_dot4_i32_i8 v42, v28, v42, v46
	v_dot4_i32_i8 v47, v6, v40, v47
	v_dot4_i32_i8 v40, v28, v40, v48
	v_dot4_i32_i8 v49, v6, v38, v49
	v_dot4_i32_i8 v38, v28, v38, v50
	v_dot4_i32_i8 v51, v6, v30, v51
	v_dot4_i32_i8 v30, v28, v30, v52
	v_dot4_i32_i8 v53, v6, v32, v53
	v_dot4_i32_i8 v32, v28, v32, v55
	v_dot4_i32_i8 v55, v6, v36, v60
	v_dot4_i32_i8 v36, v28, v36, v61
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v6, v6, v58, v10
	v_dot4_i32_i8 v10, v28, v58, v11
	v_dot4_i32_i8 v8, v29, v9, v8
	v_dot4_i32_i8 v13, v7, v43, v13
	v_dot4_i32_i8 v42, v29, v43, v42
	v_dot4_i32_i8 v47, v7, v41, v47
	v_dot4_i32_i8 v40, v29, v41, v40
	v_dot4_i32_i8 v49, v7, v39, v49
	v_dot4_i32_i8 v38, v29, v39, v38
	v_dot4_i32_i8 v51, v7, v31, v51
	v_dot4_i32_i8 v30, v29, v31, v30
	v_dot4_i32_i8 v53, v7, v33, v53
	v_dot4_i32_i8 v32, v29, v33, v32
	v_dot4_i32_i8 v55, v7, v37, v55
	v_dot4_i32_i8 v36, v29, v37, v36
	v_dot4_i32_i8 v11, v4, v2, v45
	v_dot4_i32_i8 v6, v7, v59, v6
	v_dot4_i32_i8 v7, v29, v59, v10
	ds_read_b32 v14, v108
	ds_read_b32 v15, v109
	ds_read_b32 v44, v110
	ds_read_b32 v9, v108 offset:1152
	ds_read_b32 v12, v108 offset:2304
	ds_read_b32 v43, v108 offset:3456
	ds_read_b32 v46, v108 offset:4608
	ds_read_b32 v41, v108 offset:5760
	ds_read_b32 v48, v108 offset:6912
	ds_read_b32 v39, v108 offset:8064
	v_dot4_i32_i8 v10, v5, v3, v11
	v_dot4_i32_i8 v2, v16, v2, v8
	v_dot4_i32_i8 v8, v4, v18, v13
	v_dot4_i32_i8 v11, v16, v18, v42
	v_dot4_i32_i8 v13, v4, v20, v47
	v_dot4_i32_i8 v18, v16, v20, v40
	v_dot4_i32_i8 v20, v4, v22, v49
	v_dot4_i32_i8 v22, v16, v22, v38
	v_dot4_i32_i8 v28, v4, v26, v51
	v_dot4_i32_i8 v26, v16, v26, v30
	v_dot4_i32_i8 v29, v4, v24, v53
	v_dot4_i32_i8 v24, v16, v24, v32
	v_dot4_i32_i8 v30, v4, v34, v55
	v_dot4_i32_i8 v32, v16, v34, v36
	s_waitcnt lgkmcnt(10)
	v_dot4_i32_i8 v4, v4, v56, v6
	v_dot4_i32_i8 v6, v16, v56, v7
	v_cvt_f32_i32_e32 v7, v10
	v_dot4_i32_i8 v2, v17, v3, v2
	v_dot4_i32_i8 v3, v5, v19, v8
	v_dot4_i32_i8 v8, v17, v19, v11
	v_dot4_i32_i8 v10, v5, v21, v13
	v_dot4_i32_i8 v11, v17, v21, v18
	v_dot4_i32_i8 v13, v5, v23, v20
	v_dot4_i32_i8 v16, v17, v23, v22
	v_dot4_i32_i8 v18, v5, v27, v28
	v_dot4_i32_i8 v19, v17, v27, v26
	v_dot4_i32_i8 v20, v5, v25, v29
	v_dot4_i32_i8 v21, v17, v25, v24
	v_dot4_i32_i8 v22, v5, v35, v30
	v_dot4_i32_i8 v23, v17, v35, v32
	v_dot4_i32_i8 v4, v5, v57, v4
	v_dot4_i32_i8 v5, v17, v57, v6
	v_cvt_f32_i32_e32 v2, v2
	v_cvt_f32_i32_e32 v3, v3
	v_cvt_f32_i32_e32 v6, v8
	v_cvt_f32_i32_e32 v8, v10
	v_cvt_f32_i32_e32 v10, v11
	v_cvt_f32_i32_e32 v11, v13
	v_cvt_f32_i32_e32 v13, v16
	v_cvt_f32_i32_e32 v16, v18
	v_cvt_f32_i32_e32 v17, v19
	v_cvt_f32_i32_e32 v18, v20
	v_cvt_f32_i32_e32 v19, v21
	v_cvt_f32_i32_e32 v20, v22
	v_cvt_f32_i32_e32 v21, v23
	v_cvt_f32_i32_e32 v4, v4
	v_cvt_f32_i32_e32 v5, v5
	s_waitcnt lgkmcnt(8)
	v_mul_f32_e32 v50, v15, v14
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v14, v44, v14
	s_waitcnt lgkmcnt(6)
	v_mul_f32_e32 v31, v9, v15
	v_mul_f32_e32 v9, v9, v44
	s_waitcnt lgkmcnt(5)
	v_mul_f32_e32 v52, v12, v15
	v_mul_f32_e32 v12, v12, v44
	s_waitcnt lgkmcnt(4)
	v_mul_f32_e32 v33, v43, v15
	v_mul_f32_e32 v43, v43, v44
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v54, v46, v15
	v_mul_f32_e32 v46, v46, v44
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v37, v41, v15
	v_mul_f32_e32 v41, v41, v44
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v60, v48, v15
	v_mul_f32_e32 v48, v48, v44
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v15, v39, v15
	v_mul_f32_e32 v39, v39, v44
	v_add_u32_e32 v112, 32, v112
	v_add_u32_e32 v111, 32, v111
	v_add_u32_e32 v110, 4, v110
	v_add_u32_e32 v109, 4, v109
	v_add_u32_e32 v108, 4, v108
	s_cmp_lt_u32 s54, 24
	v_fmac_f32_e32 v106, v50, v7
	v_fmac_f32_e32 v107, v14, v2
	v_fmac_f32_e32 v105, v31, v3
	v_fmac_f32_e32 v104, v9, v6
	v_fmac_f32_e32 v103, v52, v8
	v_fmac_f32_e32 v102, v12, v10
	v_fmac_f32_e32 v101, v33, v11
	v_fmac_f32_e32 v100, v43, v13
	v_fmac_f32_e32 v98, v54, v16
	v_fmac_f32_e32 v97, v46, v17
	v_fmac_f32_e32 v96, v37, v18
	v_fmac_f32_e32 v95, v41, v19
	v_fmac_f32_e32 v94, v60, v20
	v_fmac_f32_e32 v93, v48, v21
	v_fmac_f32_e32 v92, v15, v4
	v_fmac_f32_e32 v91, v39, v5
	s_cbranch_scc1 .LBB28_59
; %bb.60:                               ; %_ZL22vec_dot_q8_0_q8_1_dp4aILi64ELi128ELi8EEvPKiS1_Pfi.exit104.i601
                                        ;   in Loop: Header=BB28_56 Depth=2
	s_add_i32 s7, s7, 8
	s_cmp_ge_i32 s7, s23
	s_barrier
	s_cbranch_scc0 .LBB28_56
; %bb.61:                               ; %Flow2614
                                        ;   in Loop: Header=BB28_47 Depth=1
	buffer_load_dword v11, off, s[68:71], 0 ; 4-byte Folded Reload
	v_mov_b32_e32 v75, v1
	v_mov_b32_e32 v1, 0
.LBB28_62:                              ; %._crit_edge.i477
                                        ;   in Loop: Header=BB28_47 Depth=1
	s_not_b32 s7, s58
	v_add_u32_e32 v2, s7, v99
	v_cmp_le_i32_e32 vcc, v75, v2
	s_and_saveexec_b64 s[54:55], vcc
	s_cbranch_execz .LBB28_45
; %bb.63:                               ; %.preheader.i.i482
                                        ;   in Loop: Header=BB28_47 Depth=1
	s_waitcnt vmcnt(0)
	ds_read_b32 v3, v11
	s_add_i32 s56, s59, s56
	s_ashr_i32 s57, s56, 31
	s_lshl_b64 s[56:57], s[56:57], 2
	s_add_u32 s7, s28, s56
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[3:4], s[58:59], v3, s27, v[0:1]
	s_addc_u32 s23, s29, s57
	v_mov_b32_e32 v5, s23
	v_ashrrev_i32_e32 v4, 31, v3
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_add_co_u32_e32 v3, vcc, s7, v3
	v_addc_co_u32_e32 v4, vcc, v5, v4, vcc
	global_store_dword v[3:4], v106, off
	global_store_dword v[3:4], v107, off offset:256
	buffer_load_dword v3, off, s[68:71], 0 offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_cmp_le_u32_e32 vcc, v3, v2
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB28_45
; %bb.64:                               ; %.preheader.1.i.i486
                                        ;   in Loop: Header=BB28_47 Depth=1
	ds_read_b32 v3, v11 offset:32
	v_mov_b32_e32 v5, s23
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[3:4], s[56:57], v3, s27, v[0:1]
	v_ashrrev_i32_e32 v4, 31, v3
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_add_co_u32_e32 v3, vcc, s7, v3
	v_addc_co_u32_e32 v4, vcc, v5, v4, vcc
	global_store_dword v[3:4], v105, off
	global_store_dword v[3:4], v104, off offset:256
	buffer_load_dword v3, off, s[68:71], 0 offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_cmp_le_u32_e32 vcc, v3, v2
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB28_45
; %bb.65:                               ; %.preheader.2.i.i490
                                        ;   in Loop: Header=BB28_47 Depth=1
	ds_read_b32 v3, v11 offset:64
	v_mov_b32_e32 v5, s23
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[3:4], s[56:57], v3, s27, v[0:1]
	v_ashrrev_i32_e32 v4, 31, v3
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_add_co_u32_e32 v3, vcc, s7, v3
	v_addc_co_u32_e32 v4, vcc, v5, v4, vcc
	global_store_dword v[3:4], v103, off
	global_store_dword v[3:4], v102, off offset:256
	buffer_load_dword v3, off, s[68:71], 0 offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_cmp_le_u32_e32 vcc, v3, v2
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB28_45
; %bb.66:                               ; %.preheader.3.i.i494
                                        ;   in Loop: Header=BB28_47 Depth=1
	ds_read_b32 v3, v11 offset:96
	v_mov_b32_e32 v5, s23
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[3:4], s[56:57], v3, s27, v[0:1]
	v_ashrrev_i32_e32 v4, 31, v3
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_add_co_u32_e32 v3, vcc, s7, v3
	v_addc_co_u32_e32 v4, vcc, v5, v4, vcc
	global_store_dword v[3:4], v101, off
	global_store_dword v[3:4], v100, off offset:256
	buffer_load_dword v3, off, s[68:71], 0 offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_cmp_le_u32_e32 vcc, v3, v2
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB28_45
; %bb.67:                               ; %.preheader.4.i.i498
                                        ;   in Loop: Header=BB28_47 Depth=1
	ds_read_b32 v3, v11 offset:128
	v_mov_b32_e32 v5, s23
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[3:4], s[56:57], v3, s27, v[0:1]
	v_ashrrev_i32_e32 v4, 31, v3
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_add_co_u32_e32 v3, vcc, s7, v3
	v_addc_co_u32_e32 v4, vcc, v5, v4, vcc
	global_store_dword v[3:4], v98, off
	global_store_dword v[3:4], v97, off offset:256
	buffer_load_dword v3, off, s[68:71], 0 offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_cmp_le_u32_e32 vcc, v3, v2
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB28_45
; %bb.68:                               ; %.preheader.5.i.i502
                                        ;   in Loop: Header=BB28_47 Depth=1
	ds_read_b32 v3, v11 offset:160
	v_mov_b32_e32 v5, s23
	v_add_u32_e32 v6, 48, v75
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[3:4], s[56:57], v3, s27, v[0:1]
	v_ashrrev_i32_e32 v4, 31, v3
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_add_co_u32_e32 v3, vcc, s7, v3
	v_addc_co_u32_e32 v4, vcc, v5, v4, vcc
	v_cmp_le_u32_e32 vcc, v6, v2
	global_store_dword v[3:4], v96, off
	global_store_dword v[3:4], v95, off offset:256
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB28_45
; %bb.69:                               ; %.preheader.6.i.i506
                                        ;   in Loop: Header=BB28_47 Depth=1
	ds_read_b32 v3, v11 offset:192
	v_mov_b32_e32 v5, s23
	v_add_u32_e32 v6, 56, v75
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[3:4], s[56:57], v3, s27, v[0:1]
	v_ashrrev_i32_e32 v4, 31, v3
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_add_co_u32_e32 v3, vcc, s7, v3
	v_addc_co_u32_e32 v4, vcc, v5, v4, vcc
	v_cmp_le_u32_e32 vcc, v6, v2
	global_store_dword v[3:4], v94, off
	global_store_dword v[3:4], v93, off offset:256
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB28_45
; %bb.70:                               ; %.preheader.7.i.i510
                                        ;   in Loop: Header=BB28_47 Depth=1
	ds_read_b32 v2, v11 offset:224
	v_mov_b32_e32 v4, s23
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[2:3], s[56:57], v2, s27, v[0:1]
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_add_co_u32_e32 v2, vcc, s7, v2
	v_addc_co_u32_e32 v3, vcc, v4, v3, vcc
	global_store_dword v[2:3], v92, off
	global_store_dword v[2:3], v91, off offset:256
	s_branch .LBB28_45
.LBB28_71:                              ; %._crit_edge.loopexit
	s_min_u32 s23, s22, s56
.LBB28_72:                              ; %._crit_edge
	s_andn2_b64 vcc, exec, s[54:55]
	s_cbranch_vccnz .LBB28_106
; %bb.73:
	s_mul_hi_u32 s4, s33, s20
	s_add_i32 s33, s33, s4
	s_lshr_b32 s4, s33, s21
	s_mul_hi_u32 s5, s4, s50
	s_add_i32 s5, s4, s5
	s_lshr_b32 s5, s5, s51
	s_mul_i32 s8, s5, s34
	s_sub_i32 s8, s4, s8
	s_mul_hi_u32 s4, s5, s36
	s_add_i32 s4, s5, s4
	s_lshr_b32 s9, s4, s37
	s_mul_i32 s4, s9, s38
	s_sub_i32 s4, s5, s4
	s_mul_hi_u32 s5, s9, s44
	s_add_i32 s5, s9, s5
	s_lshr_b32 s21, s5, s45
	s_mul_i32 s5, s21, s46
	s_sub_i32 s20, s9, s5
	s_lshl_b32 s22, s8, 6
	s_cmp_lg_u64 s[16:17], 0
	s_mov_b64 s[8:9], 0
	s_cbranch_scc0 .LBB28_105
; %bb.74:
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[16:17], s[4:5], 2
	s_add_u32 s16, s18, s16
	s_addc_u32 s17, s19, s17
	v_mov_b32_e32 v1, 0
	global_load_dwordx2 v[2:3], v1, s[16:17]
	s_mov_b32 s5, 0
	s_mov_b32 s16, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s17, v2
	v_subrev_u32_e32 v1, s17, v3
	v_cmp_lt_i32_e32 vcc, s22, v1
	s_cbranch_vccz .LBB28_78
; %bb.75:
	s_barrier
	s_and_saveexec_b64 s[8:9], s[0:1]
; %bb.76:                               ; %.lr.ph813
	v_lshl_add_u32 v1, v62, 2, 0
	ds_write_b32 v1, v62
; %bb.77:                               ; %.critedge467
	s_or_b64 exec, exec, s[8:9]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_mov_b64 s[8:9], -1
	s_mov_b32 s16, s17
.LBB28_78:                              ; %Flow2639
	s_and_b64 vcc, exec, s[8:9]
	s_cbranch_vccz .LBB28_106
.LBB28_79:
	v_mov_b32_e32 v18, 0
	s_cmp_ge_i32 s7, s23
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v28, 0
	s_cbranch_scc1 .LBB28_86
; %bb.80:                               ; %.lr.ph.i648
	s_add_i32 s0, s16, s22
	s_mul_i32 s0, s0, 36
	s_add_i32 s0, s0, s5
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s14, s14, s0
	s_mul_i32 s0, s25, s21
	s_addc_u32 s15, s15, s1
	s_lshl_b32 s1, s0, 7
	s_mul_hi_u32 s0, s4, s10
	s_add_i32 s0, s4, s0
	v_lshrrev_b32_e32 v2, 4, v0
	s_lshr_b32 s0, s0, s11
	v_lshl_add_u32 v2, v75, 2, v2
	s_mul_i32 s4, s0, s39
	s_mul_hi_u32 s0, s20, s42
	v_mul_lo_u32 v36, s25, v2
	s_movk_i32 s8, 0x104
	s_add_i32 s20, s20, s0
	v_and_b32_e32 v1, 15, v0
	v_mad_u32_u24 v2, v2, s8, 0
	s_lshr_b32 s0, s20, s43
	v_lshl_add_u32 v37, v1, 2, v2
	v_lshrrev_b32_e32 v1, 3, v0
	s_mul_i32 s5, s0, s47
	s_lshl_b32 s0, s25, 5
	v_lshl_add_u32 v1, v75, 3, v1
	v_and_b32_e32 v35, 7, v0
	v_add_u32_e32 v38, s0, v36
	v_mul_lo_u32 v41, s25, v1
	v_and_b32_e32 v2, 0x1ffc, v1
	v_lshlrev_b32_e32 v5, 5, v1
	v_add_u32_e32 v1, 64, v1
	v_mul_u32_u24_e32 v7, 36, v75
	v_add_u32_e32 v39, s0, v38
	v_lshlrev_b32_e32 v4, 2, v35
	v_and_b32_e32 v6, 0x3ffc, v1
	v_add_u32_e32 v8, 64, v0
	v_lshl_add_u32 v7, v7, 2, 0
	v_add_u32_e32 v40, s0, v39
	v_add3_u32 v2, 0, v2, v4
	v_add3_u32 v4, 0, v6, v4
	s_mul_i32 s0, s26, 36
	v_lshlrev_b32_e32 v6, 5, v0
	v_add_u32_e32 v44, 0x110, v7
	v_add_u32_e32 v46, 0x100, v7
	v_and_b32_e32 v7, 0x3fc, v8
	v_and_b32_e32 v8, 0x1fc, v0
	v_lshlrev_b32_e32 v3, 1, v35
	v_lshlrev_b32_e32 v1, 5, v1
	s_add_i32 s10, s4, s1
	s_ashr_i32 s1, s0, 31
	v_mad_u32_u24 v9, v0, s8, 0
	v_add3_u32 v7, v6, v7, 0
	v_add3_u32 v6, v6, v8, 0
	v_bfe_u32 v34, v0, 3, 1
	v_mov_b32_e32 v28, 0
	v_lshl_add_u32 v42, s25, 6, v41
	v_lshl_add_u32 v43, v62, 2, 0
	s_add_i32 s10, s10, s5
	s_lshl_b64 s[4:5], s[0:1], 2
	v_add_u32_e32 v45, 0x2900, v9
	v_add_u32_e32 v47, 0xb300, v7
	v_add_u32_e32 v48, 0xab00, v6
	v_add_u32_e32 v49, 0x2980, v9
	v_add_u32_e32 v50, 0xb310, v7
	v_add_u32_e32 v51, 0xab10, v6
	v_lshlrev_b32_e32 v52, 1, v3
	v_add_u32_e32 v53, v2, v5
	v_add_u32_e32 v54, v4, v1
	v_lshlrev_b32_e32 v55, 2, v62
	s_movk_i32 s1, 0x1000
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, 0
.LBB28_81:                              ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB28_82 Depth 2
                                        ;     Child Loop BB28_84 Depth 2
	s_add_i32 s8, s10, s7
	s_mul_hi_i32 s9, s8, 34
	s_mul_i32 s8, s8, 34
	s_add_u32 s8, s12, s8
	s_addc_u32 s9, s13, s9
	v_mad_u64_u32 v[2:3], s[16:17], v34, 34, s[8:9]
	v_add_u32_e32 v65, 0x8800, v37
	s_mov_b32 s11, -8
	v_mad_i64_i32 v[4:5], s[16:17], v36, 34, v[2:3]
	v_mad_i64_i32 v[6:7], s[16:17], v38, 34, v[2:3]
	v_add_co_u32_e32 v4, vcc, v4, v52
	v_addc_co_u32_e32 v5, vcc, 0, v5, vcc
	v_add_co_u32_e32 v6, vcc, v6, v52
	v_addc_co_u32_e32 v7, vcc, 0, v7, vcc
	v_mad_i64_i32 v[8:9], s[16:17], v39, 34, v[2:3]
	global_load_dword v1, v[4:5], off offset:2
	global_load_dword v10, v[4:5], off offset:70
	global_load_dword v11, v[4:5], off offset:138
	global_load_dword v12, v[4:5], off offset:206
	global_load_dword v13, v[6:7], off offset:2
	global_load_dword v14, v[6:7], off offset:70
	global_load_dword v15, v[6:7], off offset:138
	global_load_dword v16, v[6:7], off offset:206
	v_mad_u64_u32 v[6:7], s[8:9], v35, 34, s[8:9]
	v_add_co_u32_e32 v4, vcc, v8, v52
	v_addc_co_u32_e32 v5, vcc, 0, v9, vcc
	v_mad_i64_i32 v[8:9], s[8:9], v41, 34, v[6:7]
	v_mad_i64_i32 v[6:7], s[8:9], v42, 34, v[6:7]
	s_ashr_i32 s8, s7, 31
	s_lshr_b32 s8, s8, 30
	s_add_i32 s8, s7, s8
	v_mad_i64_i32 v[2:3], s[16:17], v40, 34, v[2:3]
	s_ashr_i32 s8, s8, 2
	s_mul_i32 s8, s0, s8
	s_ashr_i32 s9, s8, 31
	s_lshl_b64 s[8:9], s[8:9], 2
	v_add_co_u32_e32 v2, vcc, v2, v52
	s_add_u32 s8, s14, s8
	v_addc_co_u32_e32 v3, vcc, 0, v3, vcc
	s_addc_u32 s9, s15, s9
	global_load_dword v17, v[4:5], off offset:2
	global_load_dword v56, v[4:5], off offset:70
	global_load_dword v57, v[4:5], off offset:138
	global_load_dword v58, v[4:5], off offset:206
	global_load_dword v59, v[2:3], off offset:2
	global_load_dword v60, v[2:3], off offset:70
	global_load_dword v61, v[2:3], off offset:138
	global_load_dword v62, v[2:3], off offset:206
                                        ; kill: killed $vgpr4 killed $vgpr5
                                        ; kill: killed $vgpr2 killed $vgpr3
	global_load_ushort v63, v[8:9], off
	global_load_ushort v64, v[6:7], off
	v_mov_b32_e32 v2, s9
	v_add_co_u32_e32 v4, vcc, s8, v55
	v_addc_co_u32_e32 v5, vcc, 0, v2, vcc
	v_add_co_u32_e32 v2, vcc, s1, v4
	v_addc_co_u32_e32 v3, vcc, 0, v5, vcc
	v_add_co_u32_e32 v4, vcc, 0x2000, v4
	v_addc_co_u32_e32 v5, vcc, 0, v5, vcc
	global_load_dword v6, v55, s[8:9]
	global_load_dword v7, v55, s[8:9] offset:2048
	global_load_dword v8, v[2:3], off
	s_nop 0
	global_load_dword v2, v[2:3], off offset:2048
	s_nop 0
	global_load_dword v3, v[4:5], off
	v_add_u32_e32 v4, 0x2800, v37
	v_add_u32_e32 v5, 0x4800, v37
	v_add_u32_e32 v9, 0x6800, v37
	s_waitcnt vmcnt(21)
	ds_write2_b32 v4, v1, v10 offset0:64 offset1:80
	s_waitcnt vmcnt(19)
	ds_write2_b32 v4, v11, v12 offset0:96 offset1:112
	s_waitcnt vmcnt(17)
	ds_write2_b32 v5, v13, v14 offset0:96 offset1:112
	s_waitcnt vmcnt(15)
	ds_write2_b32 v5, v15, v16 offset0:128 offset1:144
	s_waitcnt vmcnt(13)
	ds_write2_b32 v9, v17, v56 offset0:128 offset1:144
	s_waitcnt vmcnt(11)
	ds_write2_b32 v9, v57, v58 offset0:160 offset1:176
	v_mov_b32_e32 v56, v48
	v_mov_b32_e32 v57, v47
	v_mov_b32_e32 v58, v46
	s_waitcnt vmcnt(6)
	v_cvt_f32_f16_e32 v1, v63
	s_waitcnt vmcnt(5)
	v_cvt_f32_f16_e32 v4, v64
	ds_write2_b32 v65, v59, v60 offset0:160 offset1:176
	ds_write2_b32 v65, v61, v62 offset0:192 offset1:208
	ds_write_b32 v53, v1 offset:43776
	ds_write_b32 v54, v4 offset:43776
	s_waitcnt vmcnt(3)
	ds_write2st64_b32 v43, v6, v7 offset0:1 offset1:9
	s_waitcnt vmcnt(1)
	ds_write2st64_b32 v43, v8, v2 offset0:17 offset1:25
	s_waitcnt vmcnt(0)
	ds_write_b32 v43, v3 offset:8448
	v_mov_b32_e32 v59, v45
	v_mov_b32_e32 v60, v44
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB28_82:                              ;   Parent Loop BB28_81 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ds_read_b32 v1, v58
	ds_read2_b32 v[12:13], v60 offset1:1
	ds_read2_b32 v[14:15], v60 offset0:2 offset1:3
	ds_read2_b32 v[16:17], v60 offset0:4 offset1:5
	ds_read2_b32 v[63:64], v60 offset0:6 offset1:7
	ds_read_b32 v61, v56
	ds_read2_b32 v[2:3], v59 offset1:1
	s_add_i32 s11, s11, 8
	v_add_u32_e32 v56, 4, v56
	s_cmp_lt_u32 s11, 24
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v11, v61, v1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v4, v2, v12, 0
	v_dot4_i32_i8 v6, v3, v13, v4
	ds_read2_b32 v[4:5], v59 offset0:2 offset1:3
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v6, v4, v14, v6
	v_dot4_i32_i8 v8, v5, v15, v6
	ds_read2_b32 v[6:7], v59 offset0:4 offset1:5
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v8, v6, v16, v8
	v_dot4_i32_i8 v10, v7, v17, v8
	ds_read2_b32 v[8:9], v59 offset0:6 offset1:7
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v10, v8, v63, v10
	v_dot4_i32_i8 v10, v9, v64, v10
	v_cvt_f32_i32_e32 v10, v10
	v_fmac_f32_e32 v28, v11, v10
	v_add_u32_e32 v10, 0x4100, v59
	ds_read_b32 v62, v57
	ds_read2_b32 v[10:11], v10 offset1:1
	v_add_u32_e32 v57, 4, v57
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v1, v62, v1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v12, v10, v12, 0
	v_dot4_i32_i8 v65, v11, v13, v12
	v_add_u32_e32 v12, 0x4108, v59
	ds_read2_b32 v[12:13], v12 offset1:1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v14, v12, v14, v65
	v_dot4_i32_i8 v65, v13, v15, v14
	v_add_u32_e32 v14, 0x4110, v59
	ds_read2_b32 v[14:15], v14 offset1:1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v16, v14, v16, v65
	v_dot4_i32_i8 v65, v15, v17, v16
	v_add_u32_e32 v16, 0x4118, v59
	ds_read2_b32 v[16:17], v16 offset1:1
	v_add_u32_e32 v59, 32, v59
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v63, v16, v63, v65
	v_dot4_i32_i8 v63, v17, v64, v63
	v_cvt_f32_i32_e32 v63, v63
	v_fmac_f32_e32 v33, v1, v63
	v_add_u32_e32 v1, 0x480, v60
	ds_read_b32 v71, v58 offset:1152
	ds_read2_b32 v[63:64], v1 offset1:1
	v_add_u32_e32 v1, 0x488, v60
	ds_read2_b32 v[65:66], v1 offset1:1
	v_add_u32_e32 v1, 0x490, v60
	ds_read2_b32 v[67:68], v1 offset1:1
	v_add_u32_e32 v1, 0x498, v60
	ds_read2_b32 v[69:70], v1 offset1:1
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v1, v2, v63, 0
	v_dot4_i32_i8 v1, v3, v64, v1
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v1, v4, v65, v1
	v_dot4_i32_i8 v1, v5, v66, v1
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v1, v6, v67, v1
	v_dot4_i32_i8 v1, v7, v68, v1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v1, v8, v69, v1
	v_dot4_i32_i8 v1, v9, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v72, v71, v61
	v_fmac_f32_e32 v32, v72, v1
	v_dot4_i32_i8 v1, v10, v63, 0
	v_dot4_i32_i8 v1, v11, v64, v1
	v_dot4_i32_i8 v1, v12, v65, v1
	v_dot4_i32_i8 v1, v13, v66, v1
	v_dot4_i32_i8 v1, v14, v67, v1
	v_dot4_i32_i8 v1, v15, v68, v1
	v_dot4_i32_i8 v1, v16, v69, v1
	v_dot4_i32_i8 v1, v17, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v63, v71, v62
	v_fmac_f32_e32 v31, v63, v1
	v_add_u32_e32 v1, 0x900, v60
	ds_read_b32 v71, v58 offset:2304
	ds_read2_b32 v[63:64], v1 offset1:1
	v_add_u32_e32 v1, 0x908, v60
	ds_read2_b32 v[65:66], v1 offset1:1
	v_add_u32_e32 v1, 0x910, v60
	ds_read2_b32 v[67:68], v1 offset1:1
	v_add_u32_e32 v1, 0x918, v60
	ds_read2_b32 v[69:70], v1 offset1:1
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v1, v2, v63, 0
	v_dot4_i32_i8 v1, v3, v64, v1
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v1, v4, v65, v1
	v_dot4_i32_i8 v1, v5, v66, v1
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v1, v6, v67, v1
	v_dot4_i32_i8 v1, v7, v68, v1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v1, v8, v69, v1
	v_dot4_i32_i8 v1, v9, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v72, v71, v61
	v_fmac_f32_e32 v30, v72, v1
	v_dot4_i32_i8 v1, v10, v63, 0
	v_dot4_i32_i8 v1, v11, v64, v1
	v_dot4_i32_i8 v1, v12, v65, v1
	v_dot4_i32_i8 v1, v13, v66, v1
	v_dot4_i32_i8 v1, v14, v67, v1
	v_dot4_i32_i8 v1, v15, v68, v1
	v_dot4_i32_i8 v1, v16, v69, v1
	v_dot4_i32_i8 v1, v17, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v63, v71, v62
	v_fmac_f32_e32 v29, v63, v1
	v_add_u32_e32 v1, 0xd80, v60
	ds_read_b32 v71, v58 offset:3456
	ds_read2_b32 v[63:64], v1 offset1:1
	v_add_u32_e32 v1, 0xd88, v60
	ds_read2_b32 v[65:66], v1 offset1:1
	v_add_u32_e32 v1, 0xd90, v60
	ds_read2_b32 v[67:68], v1 offset1:1
	v_add_u32_e32 v1, 0xd98, v60
	ds_read2_b32 v[69:70], v1 offset1:1
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v1, v2, v63, 0
	v_dot4_i32_i8 v1, v3, v64, v1
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v1, v4, v65, v1
	v_dot4_i32_i8 v1, v5, v66, v1
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v1, v6, v67, v1
	v_dot4_i32_i8 v1, v7, v68, v1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v1, v8, v69, v1
	v_dot4_i32_i8 v1, v9, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v72, v71, v61
	v_fmac_f32_e32 v27, v72, v1
	v_dot4_i32_i8 v1, v10, v63, 0
	v_dot4_i32_i8 v1, v11, v64, v1
	v_dot4_i32_i8 v1, v12, v65, v1
	v_dot4_i32_i8 v1, v13, v66, v1
	v_dot4_i32_i8 v1, v14, v67, v1
	v_dot4_i32_i8 v1, v15, v68, v1
	v_dot4_i32_i8 v1, v16, v69, v1
	v_dot4_i32_i8 v1, v17, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v63, v71, v62
	v_fmac_f32_e32 v26, v63, v1
	v_add_u32_e32 v1, 0x1200, v60
	ds_read_b32 v71, v58 offset:4608
	ds_read2_b32 v[63:64], v1 offset1:1
	v_add_u32_e32 v1, 0x1208, v60
	ds_read2_b32 v[65:66], v1 offset1:1
	v_add_u32_e32 v1, 0x1210, v60
	ds_read2_b32 v[67:68], v1 offset1:1
	v_add_u32_e32 v1, 0x1218, v60
	ds_read2_b32 v[69:70], v1 offset1:1
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v1, v2, v63, 0
	v_dot4_i32_i8 v1, v3, v64, v1
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v1, v4, v65, v1
	v_dot4_i32_i8 v1, v5, v66, v1
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v1, v6, v67, v1
	v_dot4_i32_i8 v1, v7, v68, v1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v1, v8, v69, v1
	v_dot4_i32_i8 v1, v9, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v72, v71, v61
	v_fmac_f32_e32 v25, v72, v1
	v_dot4_i32_i8 v1, v10, v63, 0
	v_dot4_i32_i8 v1, v11, v64, v1
	v_dot4_i32_i8 v1, v12, v65, v1
	v_dot4_i32_i8 v1, v13, v66, v1
	v_dot4_i32_i8 v1, v14, v67, v1
	v_dot4_i32_i8 v1, v15, v68, v1
	v_dot4_i32_i8 v1, v16, v69, v1
	v_dot4_i32_i8 v1, v17, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v63, v71, v62
	v_fmac_f32_e32 v24, v63, v1
	v_add_u32_e32 v1, 0x1680, v60
	ds_read_b32 v71, v58 offset:5760
	ds_read2_b32 v[63:64], v1 offset1:1
	v_add_u32_e32 v1, 0x1688, v60
	ds_read2_b32 v[65:66], v1 offset1:1
	v_add_u32_e32 v1, 0x1690, v60
	ds_read2_b32 v[67:68], v1 offset1:1
	v_add_u32_e32 v1, 0x1698, v60
	ds_read2_b32 v[69:70], v1 offset1:1
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v1, v2, v63, 0
	v_dot4_i32_i8 v1, v3, v64, v1
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v1, v4, v65, v1
	v_dot4_i32_i8 v1, v5, v66, v1
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v1, v6, v67, v1
	v_dot4_i32_i8 v1, v7, v68, v1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v1, v8, v69, v1
	v_dot4_i32_i8 v1, v9, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v72, v71, v61
	v_fmac_f32_e32 v23, v72, v1
	v_dot4_i32_i8 v1, v10, v63, 0
	v_dot4_i32_i8 v1, v11, v64, v1
	v_dot4_i32_i8 v1, v12, v65, v1
	v_dot4_i32_i8 v1, v13, v66, v1
	v_dot4_i32_i8 v1, v14, v67, v1
	v_dot4_i32_i8 v1, v15, v68, v1
	v_dot4_i32_i8 v1, v16, v69, v1
	v_dot4_i32_i8 v1, v17, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v63, v71, v62
	v_fmac_f32_e32 v22, v63, v1
	v_add_u32_e32 v1, 0x1b00, v60
	ds_read_b32 v71, v58 offset:6912
	ds_read2_b32 v[63:64], v1 offset1:1
	v_add_u32_e32 v1, 0x1b08, v60
	ds_read2_b32 v[65:66], v1 offset1:1
	v_add_u32_e32 v1, 0x1b10, v60
	ds_read2_b32 v[67:68], v1 offset1:1
	v_add_u32_e32 v1, 0x1b18, v60
	ds_read2_b32 v[69:70], v1 offset1:1
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v1, v2, v63, 0
	v_dot4_i32_i8 v1, v3, v64, v1
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v1, v4, v65, v1
	v_dot4_i32_i8 v1, v5, v66, v1
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v1, v6, v67, v1
	v_dot4_i32_i8 v1, v7, v68, v1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v1, v8, v69, v1
	v_dot4_i32_i8 v1, v9, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v72, v71, v61
	v_fmac_f32_e32 v21, v72, v1
	v_dot4_i32_i8 v1, v10, v63, 0
	v_dot4_i32_i8 v1, v11, v64, v1
	v_dot4_i32_i8 v1, v12, v65, v1
	v_dot4_i32_i8 v1, v13, v66, v1
	v_dot4_i32_i8 v1, v14, v67, v1
	v_dot4_i32_i8 v1, v15, v68, v1
	v_dot4_i32_i8 v1, v16, v69, v1
	v_dot4_i32_i8 v1, v17, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v63, v71, v62
	v_fmac_f32_e32 v20, v63, v1
	v_add_u32_e32 v1, 0x1f80, v60
	ds_read_b32 v71, v58 offset:8064
	ds_read2_b32 v[63:64], v1 offset1:1
	v_add_u32_e32 v1, 0x1f88, v60
	ds_read2_b32 v[65:66], v1 offset1:1
	v_add_u32_e32 v1, 0x1f90, v60
	ds_read2_b32 v[67:68], v1 offset1:1
	v_add_u32_e32 v1, 0x1f98, v60
	ds_read2_b32 v[69:70], v1 offset1:1
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v1, v2, v63, 0
	v_dot4_i32_i8 v1, v3, v64, v1
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v1, v4, v65, v1
	v_dot4_i32_i8 v1, v5, v66, v1
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v1, v6, v67, v1
	v_dot4_i32_i8 v1, v7, v68, v1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v1, v8, v69, v1
	v_dot4_i32_i8 v1, v9, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v2, v71, v61
	v_add_u32_e32 v60, 32, v60
	v_add_u32_e32 v58, 4, v58
	v_fmac_f32_e32 v19, v2, v1
	v_dot4_i32_i8 v1, v10, v63, 0
	v_dot4_i32_i8 v1, v11, v64, v1
	v_dot4_i32_i8 v1, v12, v65, v1
	v_dot4_i32_i8 v1, v13, v66, v1
	v_dot4_i32_i8 v1, v14, v67, v1
	v_dot4_i32_i8 v1, v15, v68, v1
	v_dot4_i32_i8 v1, v16, v69, v1
	v_dot4_i32_i8 v1, v17, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v2, v71, v62
	v_fmac_f32_e32 v18, v2, v1
	s_cbranch_scc1 .LBB28_82
; %bb.83:                               ; %_ZL22vec_dot_q8_0_q8_1_dp4aILi64ELi128ELi8EEvPKiS1_Pfi.exit.i700
                                        ;   in Loop: Header=BB28_81 Depth=1
	s_add_u32 s8, s8, s4
	s_addc_u32 s9, s9, s5
	v_mov_b32_e32 v1, s9
	v_add_co_u32_e32 v4, vcc, s8, v55
	v_addc_co_u32_e32 v1, vcc, 0, v1, vcc
	v_add_co_u32_e32 v2, vcc, s1, v4
	v_addc_co_u32_e32 v3, vcc, 0, v1, vcc
	v_add_co_u32_e32 v4, vcc, 0x2000, v4
	s_barrier
	v_addc_co_u32_e32 v5, vcc, 0, v1, vcc
	global_load_dword v1, v55, s[8:9]
	global_load_dword v6, v55, s[8:9] offset:2048
	global_load_dword v7, v[2:3], off
	global_load_dword v8, v[2:3], off offset:2048
	global_load_dword v9, v[4:5], off
	s_mov_b32 s8, -8
	v_mov_b32_e32 v56, v46
	v_mov_b32_e32 v57, v51
	v_mov_b32_e32 v58, v50
	v_mov_b32_e32 v59, v49
	v_mov_b32_e32 v60, v44
	s_waitcnt vmcnt(3)
	ds_write2st64_b32 v43, v1, v6 offset0:1 offset1:9
	s_waitcnt vmcnt(1)
	ds_write2st64_b32 v43, v7, v8 offset0:17 offset1:25
	s_waitcnt vmcnt(0)
	ds_write_b32 v43, v9 offset:8448
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB28_84:                              ;   Parent Loop BB28_81 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ds_read_b32 v1, v56
	ds_read2_b32 v[12:13], v60 offset1:1
	ds_read2_b32 v[14:15], v60 offset0:2 offset1:3
	ds_read2_b32 v[16:17], v60 offset0:4 offset1:5
	ds_read2_b32 v[63:64], v60 offset0:6 offset1:7
	ds_read_b32 v61, v57
	ds_read2_b32 v[2:3], v59 offset1:1
	s_add_i32 s8, s8, 8
	v_add_u32_e32 v57, 4, v57
	s_cmp_lt_u32 s8, 24
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v11, v61, v1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v4, v2, v12, 0
	v_dot4_i32_i8 v6, v3, v13, v4
	ds_read2_b32 v[4:5], v59 offset0:2 offset1:3
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v6, v4, v14, v6
	v_dot4_i32_i8 v8, v5, v15, v6
	ds_read2_b32 v[6:7], v59 offset0:4 offset1:5
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v8, v6, v16, v8
	v_dot4_i32_i8 v10, v7, v17, v8
	ds_read2_b32 v[8:9], v59 offset0:6 offset1:7
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v10, v8, v63, v10
	v_dot4_i32_i8 v10, v9, v64, v10
	v_cvt_f32_i32_e32 v10, v10
	v_fmac_f32_e32 v28, v11, v10
	v_add_u32_e32 v10, 0x4100, v59
	ds_read_b32 v62, v58
	ds_read2_b32 v[10:11], v10 offset1:1
	v_add_u32_e32 v58, 4, v58
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v1, v62, v1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v12, v10, v12, 0
	v_dot4_i32_i8 v65, v11, v13, v12
	v_add_u32_e32 v12, 0x4108, v59
	ds_read2_b32 v[12:13], v12 offset1:1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v14, v12, v14, v65
	v_dot4_i32_i8 v65, v13, v15, v14
	v_add_u32_e32 v14, 0x4110, v59
	ds_read2_b32 v[14:15], v14 offset1:1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v16, v14, v16, v65
	v_dot4_i32_i8 v65, v15, v17, v16
	v_add_u32_e32 v16, 0x4118, v59
	ds_read2_b32 v[16:17], v16 offset1:1
	v_add_u32_e32 v59, 32, v59
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v63, v16, v63, v65
	v_dot4_i32_i8 v63, v17, v64, v63
	v_cvt_f32_i32_e32 v63, v63
	v_fmac_f32_e32 v33, v1, v63
	v_add_u32_e32 v1, 0x480, v60
	ds_read_b32 v71, v56 offset:1152
	ds_read2_b32 v[63:64], v1 offset1:1
	v_add_u32_e32 v1, 0x488, v60
	ds_read2_b32 v[65:66], v1 offset1:1
	v_add_u32_e32 v1, 0x490, v60
	ds_read2_b32 v[67:68], v1 offset1:1
	v_add_u32_e32 v1, 0x498, v60
	ds_read2_b32 v[69:70], v1 offset1:1
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v1, v2, v63, 0
	v_dot4_i32_i8 v1, v3, v64, v1
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v1, v4, v65, v1
	v_dot4_i32_i8 v1, v5, v66, v1
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v1, v6, v67, v1
	v_dot4_i32_i8 v1, v7, v68, v1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v1, v8, v69, v1
	v_dot4_i32_i8 v1, v9, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v72, v71, v61
	v_fmac_f32_e32 v32, v72, v1
	v_dot4_i32_i8 v1, v10, v63, 0
	v_dot4_i32_i8 v1, v11, v64, v1
	v_dot4_i32_i8 v1, v12, v65, v1
	v_dot4_i32_i8 v1, v13, v66, v1
	v_dot4_i32_i8 v1, v14, v67, v1
	v_dot4_i32_i8 v1, v15, v68, v1
	v_dot4_i32_i8 v1, v16, v69, v1
	v_dot4_i32_i8 v1, v17, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v63, v71, v62
	v_fmac_f32_e32 v31, v63, v1
	v_add_u32_e32 v1, 0x900, v60
	ds_read_b32 v71, v56 offset:2304
	ds_read2_b32 v[63:64], v1 offset1:1
	v_add_u32_e32 v1, 0x908, v60
	ds_read2_b32 v[65:66], v1 offset1:1
	v_add_u32_e32 v1, 0x910, v60
	ds_read2_b32 v[67:68], v1 offset1:1
	v_add_u32_e32 v1, 0x918, v60
	ds_read2_b32 v[69:70], v1 offset1:1
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v1, v2, v63, 0
	v_dot4_i32_i8 v1, v3, v64, v1
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v1, v4, v65, v1
	v_dot4_i32_i8 v1, v5, v66, v1
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v1, v6, v67, v1
	v_dot4_i32_i8 v1, v7, v68, v1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v1, v8, v69, v1
	v_dot4_i32_i8 v1, v9, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v72, v71, v61
	v_fmac_f32_e32 v30, v72, v1
	v_dot4_i32_i8 v1, v10, v63, 0
	v_dot4_i32_i8 v1, v11, v64, v1
	v_dot4_i32_i8 v1, v12, v65, v1
	v_dot4_i32_i8 v1, v13, v66, v1
	v_dot4_i32_i8 v1, v14, v67, v1
	v_dot4_i32_i8 v1, v15, v68, v1
	v_dot4_i32_i8 v1, v16, v69, v1
	v_dot4_i32_i8 v1, v17, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v63, v71, v62
	v_fmac_f32_e32 v29, v63, v1
	v_add_u32_e32 v1, 0xd80, v60
	ds_read_b32 v71, v56 offset:3456
	ds_read2_b32 v[63:64], v1 offset1:1
	v_add_u32_e32 v1, 0xd88, v60
	ds_read2_b32 v[65:66], v1 offset1:1
	v_add_u32_e32 v1, 0xd90, v60
	ds_read2_b32 v[67:68], v1 offset1:1
	v_add_u32_e32 v1, 0xd98, v60
	ds_read2_b32 v[69:70], v1 offset1:1
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v1, v2, v63, 0
	v_dot4_i32_i8 v1, v3, v64, v1
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v1, v4, v65, v1
	v_dot4_i32_i8 v1, v5, v66, v1
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v1, v6, v67, v1
	v_dot4_i32_i8 v1, v7, v68, v1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v1, v8, v69, v1
	v_dot4_i32_i8 v1, v9, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v72, v71, v61
	v_fmac_f32_e32 v27, v72, v1
	v_dot4_i32_i8 v1, v10, v63, 0
	v_dot4_i32_i8 v1, v11, v64, v1
	v_dot4_i32_i8 v1, v12, v65, v1
	v_dot4_i32_i8 v1, v13, v66, v1
	v_dot4_i32_i8 v1, v14, v67, v1
	v_dot4_i32_i8 v1, v15, v68, v1
	v_dot4_i32_i8 v1, v16, v69, v1
	v_dot4_i32_i8 v1, v17, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v63, v71, v62
	v_fmac_f32_e32 v26, v63, v1
	v_add_u32_e32 v1, 0x1200, v60
	ds_read_b32 v71, v56 offset:4608
	ds_read2_b32 v[63:64], v1 offset1:1
	v_add_u32_e32 v1, 0x1208, v60
	ds_read2_b32 v[65:66], v1 offset1:1
	v_add_u32_e32 v1, 0x1210, v60
	ds_read2_b32 v[67:68], v1 offset1:1
	v_add_u32_e32 v1, 0x1218, v60
	ds_read2_b32 v[69:70], v1 offset1:1
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v1, v2, v63, 0
	v_dot4_i32_i8 v1, v3, v64, v1
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v1, v4, v65, v1
	v_dot4_i32_i8 v1, v5, v66, v1
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v1, v6, v67, v1
	v_dot4_i32_i8 v1, v7, v68, v1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v1, v8, v69, v1
	v_dot4_i32_i8 v1, v9, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v72, v71, v61
	v_fmac_f32_e32 v25, v72, v1
	v_dot4_i32_i8 v1, v10, v63, 0
	v_dot4_i32_i8 v1, v11, v64, v1
	v_dot4_i32_i8 v1, v12, v65, v1
	v_dot4_i32_i8 v1, v13, v66, v1
	v_dot4_i32_i8 v1, v14, v67, v1
	v_dot4_i32_i8 v1, v15, v68, v1
	v_dot4_i32_i8 v1, v16, v69, v1
	v_dot4_i32_i8 v1, v17, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v63, v71, v62
	v_fmac_f32_e32 v24, v63, v1
	v_add_u32_e32 v1, 0x1680, v60
	ds_read_b32 v71, v56 offset:5760
	ds_read2_b32 v[63:64], v1 offset1:1
	v_add_u32_e32 v1, 0x1688, v60
	ds_read2_b32 v[65:66], v1 offset1:1
	v_add_u32_e32 v1, 0x1690, v60
	ds_read2_b32 v[67:68], v1 offset1:1
	v_add_u32_e32 v1, 0x1698, v60
	ds_read2_b32 v[69:70], v1 offset1:1
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v1, v2, v63, 0
	v_dot4_i32_i8 v1, v3, v64, v1
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v1, v4, v65, v1
	v_dot4_i32_i8 v1, v5, v66, v1
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v1, v6, v67, v1
	v_dot4_i32_i8 v1, v7, v68, v1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v1, v8, v69, v1
	v_dot4_i32_i8 v1, v9, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v72, v71, v61
	v_fmac_f32_e32 v23, v72, v1
	v_dot4_i32_i8 v1, v10, v63, 0
	v_dot4_i32_i8 v1, v11, v64, v1
	v_dot4_i32_i8 v1, v12, v65, v1
	v_dot4_i32_i8 v1, v13, v66, v1
	v_dot4_i32_i8 v1, v14, v67, v1
	v_dot4_i32_i8 v1, v15, v68, v1
	v_dot4_i32_i8 v1, v16, v69, v1
	v_dot4_i32_i8 v1, v17, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v63, v71, v62
	v_fmac_f32_e32 v22, v63, v1
	v_add_u32_e32 v1, 0x1b00, v60
	ds_read_b32 v71, v56 offset:6912
	ds_read2_b32 v[63:64], v1 offset1:1
	v_add_u32_e32 v1, 0x1b08, v60
	ds_read2_b32 v[65:66], v1 offset1:1
	v_add_u32_e32 v1, 0x1b10, v60
	ds_read2_b32 v[67:68], v1 offset1:1
	v_add_u32_e32 v1, 0x1b18, v60
	ds_read2_b32 v[69:70], v1 offset1:1
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v1, v2, v63, 0
	v_dot4_i32_i8 v1, v3, v64, v1
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v1, v4, v65, v1
	v_dot4_i32_i8 v1, v5, v66, v1
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v1, v6, v67, v1
	v_dot4_i32_i8 v1, v7, v68, v1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v1, v8, v69, v1
	v_dot4_i32_i8 v1, v9, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v72, v71, v61
	v_fmac_f32_e32 v21, v72, v1
	v_dot4_i32_i8 v1, v10, v63, 0
	v_dot4_i32_i8 v1, v11, v64, v1
	v_dot4_i32_i8 v1, v12, v65, v1
	v_dot4_i32_i8 v1, v13, v66, v1
	v_dot4_i32_i8 v1, v14, v67, v1
	v_dot4_i32_i8 v1, v15, v68, v1
	v_dot4_i32_i8 v1, v16, v69, v1
	v_dot4_i32_i8 v1, v17, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v63, v71, v62
	v_fmac_f32_e32 v20, v63, v1
	v_add_u32_e32 v1, 0x1f80, v60
	ds_read_b32 v71, v56 offset:8064
	ds_read2_b32 v[63:64], v1 offset1:1
	v_add_u32_e32 v1, 0x1f88, v60
	ds_read2_b32 v[65:66], v1 offset1:1
	v_add_u32_e32 v1, 0x1f90, v60
	ds_read2_b32 v[67:68], v1 offset1:1
	v_add_u32_e32 v1, 0x1f98, v60
	ds_read2_b32 v[69:70], v1 offset1:1
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v1, v2, v63, 0
	v_dot4_i32_i8 v1, v3, v64, v1
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v1, v4, v65, v1
	v_dot4_i32_i8 v1, v5, v66, v1
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v1, v6, v67, v1
	v_dot4_i32_i8 v1, v7, v68, v1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v1, v8, v69, v1
	v_dot4_i32_i8 v1, v9, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v2, v71, v61
	v_add_u32_e32 v60, 32, v60
	v_add_u32_e32 v56, 4, v56
	v_fmac_f32_e32 v19, v2, v1
	v_dot4_i32_i8 v1, v10, v63, 0
	v_dot4_i32_i8 v1, v11, v64, v1
	v_dot4_i32_i8 v1, v12, v65, v1
	v_dot4_i32_i8 v1, v13, v66, v1
	v_dot4_i32_i8 v1, v14, v67, v1
	v_dot4_i32_i8 v1, v15, v68, v1
	v_dot4_i32_i8 v1, v16, v69, v1
	v_dot4_i32_i8 v1, v17, v70, v1
	v_cvt_f32_i32_e32 v1, v1
	v_mul_f32_e32 v2, v71, v62
	v_fmac_f32_e32 v18, v2, v1
	s_cbranch_scc1 .LBB28_84
; %bb.85:                               ; %_ZL22vec_dot_q8_0_q8_1_dp4aILi64ELi128ELi8EEvPKiS1_Pfi.exit101.i
                                        ;   in Loop: Header=BB28_81 Depth=1
	s_add_i32 s7, s7, 8
	s_cmp_ge_i32 s7, s23
	s_barrier
	s_cbranch_scc0 .LBB28_81
.LBB28_86:                              ; %._crit_edge.i612
	s_movk_i32 s0, 0x41
	v_cmp_gt_u32_e32 vcc, s0, v75
                                        ; implicit-def: $sgpr8_sgpr9
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB28_102
; %bb.87:                               ; %.preheader.i.i617
	v_lshl_add_u32 v3, v75, 2, 0
	ds_read_b32 v1, v3
	s_lshl_b32 s4, s6, 13
	s_mov_b32 s5, 0
	s_lshl_b64 s[4:5], s[4:5], 2
	s_add_u32 s8, s30, s4
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u32 v4, v1, 7, v0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	s_addc_u32 s9, s31, s5
	v_mov_b32_e32 v1, s9
	v_add_co_u32_e32 v4, vcc, s8, v4
	v_addc_co_u32_e32 v5, vcc, v1, v5, vcc
	v_cmp_gt_u32_e32 vcc, 57, v75
	s_mov_b64 s[6:7], s[2:3]
	global_store_dword v[4:5], v28, off
	global_store_dword v[4:5], v33, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB28_101
; %bb.88:                               ; %.preheader.1.i.i621
	ds_read_b32 v1, v3 offset:32
	v_mov_b32_e32 v2, s9
	s_mov_b64 s[10:11], s[2:3]
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u32 v4, v1, 7, v0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_add_co_u32_e32 v4, vcc, s8, v4
	v_addc_co_u32_e32 v5, vcc, v2, v5, vcc
	v_cmp_gt_u32_e32 vcc, 49, v75
	global_store_dword v[4:5], v32, off
	global_store_dword v[4:5], v31, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB28_100
; %bb.89:                               ; %.preheader.2.i.i625
	ds_read_b32 v1, v3 offset:64
	v_mov_b32_e32 v2, s9
	s_mov_b64 s[12:13], s[2:3]
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u32 v4, v1, 7, v0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_add_co_u32_e32 v4, vcc, s8, v4
	v_addc_co_u32_e32 v5, vcc, v2, v5, vcc
	v_cmp_gt_u32_e32 vcc, 41, v75
	global_store_dword v[4:5], v30, off
	global_store_dword v[4:5], v29, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB28_99
; %bb.90:                               ; %.preheader.3.i.i629
	ds_read_b32 v1, v3 offset:96
	v_mov_b32_e32 v2, s9
	s_mov_b64 s[14:15], s[2:3]
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u32 v4, v1, 7, v0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_add_co_u32_e32 v4, vcc, s8, v4
	v_addc_co_u32_e32 v5, vcc, v2, v5, vcc
	v_cmp_gt_u32_e32 vcc, 33, v75
	global_store_dword v[4:5], v27, off
	global_store_dword v[4:5], v26, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB28_98
; %bb.91:                               ; %.preheader.4.i.i633
	ds_read_b32 v1, v3 offset:128
	v_mov_b32_e32 v2, s9
	s_mov_b64 s[16:17], s[2:3]
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u32 v4, v1, 7, v0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_add_co_u32_e32 v4, vcc, s8, v4
	v_addc_co_u32_e32 v5, vcc, v2, v5, vcc
	v_cmp_gt_u32_e32 vcc, 25, v75
	global_store_dword v[4:5], v25, off
	global_store_dword v[4:5], v24, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB28_97
; %bb.92:                               ; %.preheader.5.i.i637
	ds_read_b32 v1, v3 offset:160
	v_mov_b32_e32 v2, s9
	s_mov_b64 s[18:19], s[2:3]
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u32 v4, v1, 7, v0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_add_co_u32_e32 v4, vcc, s8, v4
	v_addc_co_u32_e32 v5, vcc, v2, v5, vcc
	v_cmp_gt_u32_e32 vcc, 17, v75
	global_store_dword v[4:5], v23, off
	global_store_dword v[4:5], v22, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB28_96
; %bb.93:                               ; %.preheader.6.i.i641
	ds_read_b32 v1, v3 offset:192
	v_mov_b32_e32 v2, s9
	s_mov_b64 s[18:19], s[2:3]
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u32 v4, v1, 7, v0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_add_co_u32_e32 v4, vcc, s8, v4
	v_addc_co_u32_e32 v5, vcc, v2, v5, vcc
	v_cmp_gt_u32_e32 vcc, 9, v75
	global_store_dword v[4:5], v21, off
	global_store_dword v[4:5], v20, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB28_95
; %bb.94:                               ; %.preheader.7.i.i645
	ds_read_b32 v1, v3 offset:224
	s_or_b64 s[18:19], s[2:3], exec
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v2, 7, v1
.LBB28_95:                              ; %Flow2649
	s_or_b64 exec, exec, s[20:21]
	s_andn2_b64 s[20:21], s[2:3], exec
	s_and_b64 s[18:19], s[18:19], exec
	s_or_b64 s[18:19], s[20:21], s[18:19]
.LBB28_96:                              ; %Flow2648
	s_or_b64 exec, exec, s[16:17]
	s_andn2_b64 s[16:17], s[2:3], exec
	s_and_b64 s[18:19], s[18:19], exec
	s_or_b64 s[16:17], s[16:17], s[18:19]
.LBB28_97:                              ; %Flow2647
	s_or_b64 exec, exec, s[14:15]
	s_andn2_b64 s[14:15], s[2:3], exec
	s_and_b64 s[16:17], s[16:17], exec
	s_or_b64 s[14:15], s[14:15], s[16:17]
.LBB28_98:                              ; %Flow2646
	s_or_b64 exec, exec, s[12:13]
	s_andn2_b64 s[12:13], s[2:3], exec
	s_and_b64 s[14:15], s[14:15], exec
	s_or_b64 s[12:13], s[12:13], s[14:15]
.LBB28_99:                              ; %Flow2645
	s_or_b64 exec, exec, s[10:11]
	s_andn2_b64 s[10:11], s[2:3], exec
	s_and_b64 s[12:13], s[12:13], exec
	s_or_b64 s[10:11], s[10:11], s[12:13]
.LBB28_100:                             ; %Flow2644
	s_or_b64 exec, exec, s[6:7]
	s_andn2_b64 s[6:7], s[2:3], exec
	s_and_b64 s[10:11], s[10:11], exec
	s_or_b64 s[6:7], s[6:7], s[10:11]
.LBB28_101:                             ; %Flow2643
	s_or_b64 exec, exec, s[4:5]
	s_andn2_b64 s[2:3], s[2:3], exec
	s_and_b64 s[4:5], s[6:7], exec
	s_or_b64 s[2:3], s[2:3], s[4:5]
.LBB28_102:                             ; %Flow2642
	s_or_b64 exec, exec, s[0:1]
	s_branch .LBB28_107
.LBB28_103:
                                        ; implicit-def: $sgpr4_sgpr5
	v_cvt_f32_u32_e32 v3, s7
	s_branch .LBB28_39
.LBB28_104:
                                        ; implicit-def: $sgpr52_sgpr53
	s_branch .LBB28_42
.LBB28_105:
	s_mul_i32 s0, s20, s48
	s_mul_i32 s1, s4, s40
	s_add_i32 s5, s0, s1
	s_mov_b32 s16, 0
	s_cbranch_execnz .LBB28_79
.LBB28_106:
                                        ; implicit-def: $vgpr18
                                        ; implicit-def: $vgpr19
                                        ; implicit-def: $sgpr8_sgpr9
                                        ; implicit-def: $vgpr2
.LBB28_107:                             ; %Flow2637
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB28_109
; %bb.108:                              ; %_ZL22mul_mat_q_process_tileIL9ggml_type8ELi64ELb0ELb0EEvPKciPKiS4_PfS5_iiiiiii.exit
	s_endpgm
.LBB28_109:                             ; %_ZL22mul_mat_q_process_tileIL9ggml_type8ELi64ELb0ELb0EEvPKciPKiS4_PfS5_iiiiiii.exit.sink.split
	v_add_u32_e32 v0, v2, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	v_mov_b32_e32 v2, s9
	v_add_co_u32_e32 v0, vcc, s8, v0
	v_addc_co_u32_e32 v1, vcc, v2, v1, vcc
	global_store_dword v[0:1], v19, off
	global_store_dword v[0:1], v18, off offset:256
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZL9mul_mat_qIL9ggml_type8ELi64ELb0EEvPKcPKiS4_S4_PfS5_15HIP_vector_typeIjLj3EEiiiiiS7_S7_iiiS7_S7_iiiS7_b
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 28
		.amdhsa_kernarg_size 424
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 128
		.amdhsa_next_free_sgpr 72
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZL9mul_mat_qIL9ggml_type8ELi64ELb0EEvPKcPKiS4_S4_PfS5_15HIP_vector_typeIjLj3EEiiiiiS7_S7_iiiS7_S7_iiiS7_b,"axG",@progbits,_ZL9mul_mat_qIL9ggml_type8ELi64ELb0EEvPKcPKiS4_S4_PfS5_15HIP_vector_typeIjLj3EEiiiiiS7_S7_iiiS7_S7_iiiS7_b,comdat
```

---

### 2.4 Native `mul_mat_q<(ggml_type)8, 96, false>` ISA body

Source: `mmq-instance-q8_0-hip-amdgcn-amd-amdhsa-gfx906.s`  |  Mangled: `mul_mat_q<(ggml_type)8, 96, false>`  |  Body lines: 4834  |  Total instructions: 4580

```asm
_ZL9mul_mat_qIL9ggml_type8ELi96ELb0EEvPKcPKiS4_S4_PfS5_15HIP_vector_typeIjLj3EEiiiiiS7_S7_iiiS7_S7_iiiS7_b: ; @_ZL9mul_mat_qIL9ggml_type8ELi96ELb0EEvPKcPKiS4_S4_PfS5_15HIP_vector_typeIjLj3EEiiiiiS7_S7_iiiS7_S7_iiiS7_b
; %bb.0:
	s_load_dwordx8 s[20:27], s[4:5], 0x30
	s_load_dwordx2 s[10:11], s[4:5], 0x50
	s_load_dwordx8 s[36:43], s[4:5], 0x5c
	s_load_dwordx2 s[34:35], s[4:5], 0xa0
	s_load_dwordx8 s[44:51], s[4:5], 0x80
	v_lshl_add_u32 v24, v1, 6, v0
	s_movk_i32 s0, 0x60
	v_cmp_gt_u32_e64 s[0:1], s0, v24
	s_and_saveexec_b64 s[2:3], s[0:1]
; %bb.1:                                ; %.lr.ph
	v_lshl_add_u32 v2, v24, 2, 0
	ds_write_b32 v2, v24
; %bb.2:                                ; %.critedge
	s_or_b64 exec, exec, s[2:3]
	s_load_dwordx8 s[12:19], s[4:5], 0x0
	s_load_dwordx4 s[28:31], s[4:5], 0x20
	s_waitcnt lgkmcnt(0)
	s_bitcmp1_b32 s35, 0
	s_cselect_b64 s[2:3], -1, 0
	s_and_b64 vcc, exec, s[2:3]
	s_barrier
	s_cbranch_vccnz .LBB44_8
; %bb.3:
	s_mul_hi_u32 s2, s36, s8
	s_add_i32 s2, s8, s2
	s_lshr_b32 s57, s2, s37
	s_mul_i32 s2, s57, s38
	s_sub_i32 s8, s8, s2
	s_mulk_i32 s7, 0x60
	s_cmp_lg_u64 s[16:17], 0
	s_mov_b64 s[2:3], 0
	s_cbranch_scc0 .LBB44_9
; %bb.4:
	s_ashr_i32 s9, s8, 31
	s_lshl_b64 s[52:53], s[8:9], 2
	s_add_u32 s54, s18, s52
	s_addc_u32 s55, s19, s53
	s_load_dwordx2 s[52:53], s[54:55], 0x0
	s_mov_b64 s[54:55], 0
	s_mov_b32 s9, 0
	s_waitcnt lgkmcnt(0)
	s_sub_i32 s33, s53, s52
	s_cmp_lt_i32 s7, s33
	s_mov_b32 s53, 0
	s_cbranch_scc0 .LBB44_17
; %bb.5:                                ; %.preheader
	s_and_saveexec_b64 s[54:55], s[0:1]
	s_cbranch_execz .LBB44_7
; %bb.6:                                ; %.lr.ph1104
	s_add_i32 s35, s52, s7
	v_add_u32_e32 v2, s35, v24
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_mov_b32_e32 v4, s17
	v_add_co_u32_e32 v2, vcc, s16, v2
	v_addc_co_u32_e32 v3, vcc, v4, v3, vcc
	global_load_dword v2, v[2:3], off
	v_lshl_add_u32 v3, v24, 2, 0
	s_waitcnt vmcnt(0)
	ds_write_b32 v3, v2
.LBB44_7:                               ; %.critedge463
	s_or_b64 exec, exec, s[54:55]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_mov_b32 s53, s52
	s_mov_b32 s35, 0
	s_cbranch_execnz .LBB44_10
	s_branch .LBB44_18
.LBB44_8:
	s_mov_b64 s[2:3], 0
                                        ; implicit-def: $vgpr25
                                        ; implicit-def: $vgpr26
                                        ; implicit-def: $sgpr8_sgpr9
                                        ; implicit-def: $vgpr2
	s_cbranch_execnz .LBB44_45
	s_branch .LBB44_126
.LBB44_9:
	s_mul_i32 s9, s57, s48
	s_mul_i32 s33, s8, s40
	s_add_i32 s9, s33, s9
	s_mul_i32 s33, s57, s49
	s_mul_i32 s52, s27, s7
	s_mul_i32 s35, s8, s41
	s_add_i32 s33, s33, s52
	s_add_i32 s35, s33, s35
	s_mov_b32 s53, 0
	s_mov_b32 s33, s24
	s_cbranch_execz .LBB44_18
.LBB44_10:
	s_lshl_b32 s54, s6, 7
	s_cmp_lt_i32 s22, 1
	s_cbranch_scc1 .LBB44_19
; %bb.11:                               ; %.lr.ph.i
	s_add_i32 s2, s53, s7
	s_mul_i32 s2, s2, 36
	s_add_i32 s2, s2, s9
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 2
	s_add_u32 s55, s14, s2
	s_mul_hi_u32 s2, s8, s10
	s_addc_u32 s56, s15, s3
	s_add_i32 s2, s8, s2
	v_lshrrev_b32_e32 v3, 4, v0
	s_lshr_b32 s2, s2, s11
	v_lshl_add_u32 v3, v1, 2, v3
	s_mul_i32 s8, s2, s39
	s_mul_hi_u32 s2, s57, s42
	v_mul_lo_u32 v51, s25, v3
	s_movk_i32 s52, 0x104
	s_add_i32 s57, s57, s2
	v_and_b32_e32 v2, 15, v0
	v_mad_u32_u24 v3, v3, s52, 0
	s_lshr_b32 s2, s57, s43
	v_lshl_add_u32 v52, v2, 2, v3
	v_lshrrev_b32_e32 v2, 3, v0
	s_mul_i32 s9, s2, s47
	s_lshl_b32 s2, s25, 5
	v_lshl_add_u32 v2, v1, 3, v2
	v_and_b32_e32 v49, 7, v0
	v_add_u32_e32 v53, s2, v51
	v_mul_lo_u32 v56, s25, v2
	v_and_b32_e32 v3, 0x1ffc, v2
	v_lshlrev_b32_e32 v6, 5, v2
	v_add_u32_e32 v2, 64, v2
	v_add_u32_e32 v54, s2, v53
	v_lshlrev_b32_e32 v5, 2, v49
	v_and_b32_e32 v7, 0x3ffc, v2
	v_add_u32_e32 v9, 64, v0
	s_mul_i32 s3, s25, s54
	v_add_u32_e32 v55, s2, v54
	v_add3_u32 v3, 0, v3, v5
	v_add3_u32 v5, 0, v7, v5
	s_mul_i32 s2, s26, 36
	v_lshlrev_b32_e32 v7, 5, v0
	v_mul_u32_u24_e32 v8, 36, v1
	v_and_b32_e32 v9, 0x3fc, v9
	v_and_b32_e32 v11, 0x1fc, v0
	v_lshlrev_b32_e32 v4, 1, v49
	v_lshlrev_b32_e32 v2, 5, v2
	s_add_i32 s58, s9, s3
	s_ashr_i32 s3, s2, 31
	v_lshl_add_u32 v8, v8, 2, 0
	v_mad_u32_u24 v10, v0, s52, 0
	v_add3_u32 v9, v7, v9, 0
	v_add3_u32 v7, v7, v11, 0
	v_bfe_u32 v47, v0, 3, 1
	v_mov_b32_e32 v25, 0
	s_mov_b32 s57, 0
	s_add_i32 s58, s58, s8
	s_lshl_b64 s[8:9], s[2:3], 2
	v_add_u32_e32 v57, 0x190, v8
	v_add_u32_e32 v58, 0x3980, v10
	v_add_u32_e32 v59, 0x180, v8
	v_add_u32_e32 v60, v3, v6
	v_add_u32_e32 v61, 0xc380, v9
	v_add_u32_e32 v62, v5, v2
	v_add_u32_e32 v63, 0xbb80, v7
	v_add_u32_e32 v64, 0x3a00, v10
	v_add_u32_e32 v65, 0xc390, v9
	v_add_u32_e32 v66, 0xbb90, v7
	v_lshlrev_b32_e32 v67, 1, v4
	s_movk_i32 s3, 0x1000
	v_lshl_add_u32 v68, s25, 6, v56
	v_lshl_add_u32 v69, v24, 2, 0
	v_lshlrev_b32_e32 v70, 2, v24
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v50, 0
.LBB44_12:                              ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB44_13 Depth 2
                                        ;     Child Loop BB44_15 Depth 2
	s_add_i32 s52, s58, s57
	s_mul_hi_i32 s53, s52, 34
	s_mul_i32 s52, s52, 34
	s_add_u32 s52, s12, s52
	s_addc_u32 s53, s13, s53
	v_mad_u64_u32 v[2:3], s[60:61], v47, 34, s[52:53]
	v_add_u32_e32 v8, 0x3800, v52
	v_add_u32_e32 v71, 0x80, v69
	v_mad_i64_i32 v[4:5], s[60:61], v51, 34, v[2:3]
	s_mov_b32 s59, -8
	v_mov_b32_e32 v72, v63
	v_add_co_u32_e32 v4, vcc, v4, v67
	v_addc_co_u32_e32 v5, vcc, 0, v5, vcc
	global_load_dword v6, v[4:5], off offset:2
	global_load_dword v7, v[4:5], off offset:70
	v_mov_b32_e32 v73, v61
	v_mov_b32_e32 v74, v59
	v_mov_b32_e32 v75, v58
	v_mov_b32_e32 v76, v57
	s_waitcnt vmcnt(0)
	ds_write2_b32 v8, v6, v7 offset0:96 offset1:112
	global_load_dword v6, v[4:5], off offset:138
	s_nop 0
	global_load_dword v4, v[4:5], off offset:206
	s_waitcnt vmcnt(0)
	ds_write2_b32 v8, v6, v4 offset0:128 offset1:144
	v_mad_i64_i32 v[4:5], s[60:61], v53, 34, v[2:3]
	v_add_u32_e32 v8, 0x5800, v52
	v_add_co_u32_e32 v4, vcc, v4, v67
	v_addc_co_u32_e32 v5, vcc, 0, v5, vcc
	global_load_dword v6, v[4:5], off offset:2
	global_load_dword v7, v[4:5], off offset:70
	s_waitcnt vmcnt(0)
	ds_write2_b32 v8, v6, v7 offset0:128 offset1:144
	global_load_dword v6, v[4:5], off offset:138
	s_nop 0
	global_load_dword v4, v[4:5], off offset:206
	s_waitcnt vmcnt(0)
	ds_write2_b32 v8, v6, v4 offset0:160 offset1:176
	v_mad_i64_i32 v[4:5], s[60:61], v54, 34, v[2:3]
	v_add_u32_e32 v8, 0x7800, v52
	v_mad_i64_i32 v[2:3], s[60:61], v55, 34, v[2:3]
	v_add_co_u32_e32 v4, vcc, v4, v67
	v_addc_co_u32_e32 v5, vcc, 0, v5, vcc
	global_load_dword v6, v[4:5], off offset:2
	global_load_dword v7, v[4:5], off offset:70
	v_add_co_u32_e32 v2, vcc, v2, v67
	v_addc_co_u32_e32 v3, vcc, 0, v3, vcc
	s_waitcnt vmcnt(0)
	ds_write2_b32 v8, v6, v7 offset0:160 offset1:176
	global_load_dword v6, v[4:5], off offset:138
	s_nop 0
	global_load_dword v4, v[4:5], off offset:206
	s_waitcnt vmcnt(0)
	ds_write2_b32 v8, v6, v4 offset0:192 offset1:208
	global_load_dword v4, v[2:3], off offset:2
	global_load_dword v5, v[2:3], off offset:70
	v_add_u32_e32 v6, 0x9800, v52
	s_waitcnt vmcnt(0)
	ds_write2_b32 v6, v4, v5 offset0:192 offset1:208
	global_load_dword v4, v[2:3], off offset:138
	s_nop 0
	global_load_dword v2, v[2:3], off offset:206
	s_waitcnt vmcnt(0)
	ds_write2_b32 v6, v4, v2 offset0:224 offset1:240
	v_mad_u64_u32 v[2:3], s[52:53], v49, 34, s[52:53]
	v_mad_i64_i32 v[4:5], s[52:53], v56, 34, v[2:3]
	v_mad_i64_i32 v[2:3], s[52:53], v68, 34, v[2:3]
	global_load_ushort v4, v[4:5], off
	s_lshr_b32 s52, s57, 2
	global_load_ushort v2, v[2:3], off
	s_mul_i32 s52, s2, s52
	s_ashr_i32 s53, s52, 31
	s_lshl_b64 s[52:53], s[52:53], 2
	s_add_u32 s52, s55, s52
	s_addc_u32 s53, s56, s53
	s_waitcnt vmcnt(1)
	v_cvt_f32_f16_e32 v4, v4
	s_waitcnt vmcnt(0)
	v_cvt_f32_f16_e32 v2, v2
	ds_write_b32 v60, v4 offset:48000
	v_add_co_u32_e32 v4, vcc, s52, v70
	ds_write_b32 v62, v2 offset:48000
	v_mov_b32_e32 v2, s53
	v_addc_co_u32_e32 v5, vcc, 0, v2, vcc
	global_load_dword v2, v70, s[52:53]
	global_load_dword v3, v70, s[52:53] offset:2048
	s_waitcnt vmcnt(0)
	ds_write2st64_b32 v71, v2, v3 offset0:1 offset1:9
	v_add_co_u32_e32 v2, vcc, s3, v4
	v_addc_co_u32_e32 v3, vcc, 0, v5, vcc
	global_load_dword v6, v[2:3], off
	s_nop 0
	global_load_dword v2, v[2:3], off offset:2048
	s_waitcnt vmcnt(0)
	ds_write2st64_b32 v71, v6, v2 offset0:17 offset1:25
	v_add_co_u32_e32 v2, vcc, 0x2000, v4
	v_addc_co_u32_e32 v3, vcc, 0, v5, vcc
	global_load_dword v6, v[2:3], off
	s_nop 0
	global_load_dword v2, v[2:3], off offset:2048
	s_waitcnt vmcnt(0)
	ds_write2st64_b32 v71, v6, v2 offset0:33 offset1:41
	v_add_co_u32_e32 v2, vcc, 0x3000, v4
	v_addc_co_u32_e32 v3, vcc, 0, v5, vcc
	global_load_dword v2, v[2:3], off
	s_waitcnt vmcnt(0)
	ds_write_b32 v69, v2 offset:12672
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB44_13:                              ;   Parent Loop BB44_12 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ds_read_b32 v22, v74
	ds_read2_b32 v[2:3], v76 offset1:1
	ds_read2_b32 v[4:5], v76 offset0:2 offset1:3
	ds_read2_b32 v[18:19], v76 offset0:4 offset1:5
	ds_read2_b32 v[20:21], v76 offset0:6 offset1:7
	ds_read_b32 v78, v72
	ds_read2_b32 v[8:9], v75 offset1:1
	ds_read2_b32 v[10:11], v75 offset0:2 offset1:3
	ds_read2_b32 v[14:15], v75 offset0:4 offset1:5
	ds_read2_b32 v[16:17], v75 offset0:6 offset1:7
	s_waitcnt lgkmcnt(4)
	v_mul_f32_e32 v7, v78, v22
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v6, v8, v2, 0
	v_dot4_i32_i8 v6, v9, v3, v6
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v6, v10, v4, v6
	v_dot4_i32_i8 v6, v11, v5, v6
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v6, v14, v18, v6
	v_dot4_i32_i8 v6, v15, v19, v6
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v6, v16, v20, v6
	v_dot4_i32_i8 v6, v17, v21, v6
	v_cvt_f32_i32_e32 v6, v6
	v_add_u32_e32 v79, 0x498, v76
	s_add_i32 s59, s59, 8
	v_add_u32_e32 v72, 4, v72
	v_fmac_f32_e32 v50, v7, v6
	v_add_u32_e32 v6, 0x4100, v75
	ds_read_b32 v77, v73
	ds_read2_b32 v[12:13], v6 offset1:1
	v_add_u32_e32 v73, 4, v73
	s_cmp_lt_u32 s59, 24
	ds_read2_b32 v[79:80], v79 offset1:1
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v2, v12, v2, 0
	v_dot4_i32_i8 v2, v13, v3, v2
	v_add_u32_e32 v3, 0x4108, v75
	ds_read2_b32 v[6:7], v3 offset1:1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v2, v6, v4, v2
	v_dot4_i32_i8 v4, v7, v5, v2
	v_add_u32_e32 v2, 0x4110, v75
	ds_read2_b32 v[2:3], v2 offset1:1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v4, v2, v18, v4
	v_dot4_i32_i8 v18, v3, v19, v4
	v_add_u32_e32 v4, 0x4118, v75
	ds_read2_b32 v[4:5], v4 offset1:1
	v_mul_f32_e32 v19, v77, v22
	v_add_u32_e32 v22, 0x490, v76
	ds_read2_b32 v[22:23], v22 offset1:1
	v_add_u32_e32 v75, 32, v75
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v18, v4, v20, v18
	v_dot4_i32_i8 v18, v5, v21, v18
	v_cvt_f32_i32_e32 v18, v18
	v_add_u32_e32 v20, 0x488, v76
	ds_read2_b32 v[20:21], v20 offset1:1
	v_fmac_f32_e32 v48, v19, v18
	v_add_u32_e32 v18, 0x480, v76
	ds_read_b32 v81, v74 offset:1152
	ds_read2_b32 v[18:19], v18 offset1:1
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v83, v81, v78
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v82, v8, v18, 0
	v_dot4_i32_i8 v18, v12, v18, 0
	v_dot4_i32_i8 v18, v13, v19, v18
	v_dot4_i32_i8 v18, v6, v20, v18
	v_dot4_i32_i8 v18, v7, v21, v18
	v_dot4_i32_i8 v18, v2, v22, v18
	v_dot4_i32_i8 v18, v3, v23, v18
	v_dot4_i32_i8 v18, v4, v79, v18
	v_dot4_i32_i8 v82, v9, v19, v82
	v_dot4_i32_i8 v18, v5, v80, v18
	v_dot4_i32_i8 v82, v10, v20, v82
	v_cvt_f32_i32_e32 v18, v18
	v_dot4_i32_i8 v82, v11, v21, v82
	v_dot4_i32_i8 v82, v14, v22, v82
	v_dot4_i32_i8 v82, v15, v23, v82
	v_mul_f32_e32 v19, v81, v77
	v_dot4_i32_i8 v82, v16, v79, v82
	v_fmac_f32_e32 v45, v19, v18
	v_add_u32_e32 v18, 0x900, v76
	v_dot4_i32_i8 v82, v17, v80, v82
	ds_read_b32 v81, v74 offset:2304
	ds_read2_b32 v[18:19], v18 offset1:1
	v_cvt_f32_i32_e32 v82, v82
	v_add_u32_e32 v20, 0x908, v76
	ds_read2_b32 v[20:21], v20 offset1:1
	v_add_u32_e32 v22, 0x910, v76
	ds_read2_b32 v[22:23], v22 offset1:1
	v_fmac_f32_e32 v46, v83, v82
	v_add_u32_e32 v79, 0x918, v76
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v82, v8, v18, 0
	v_dot4_i32_i8 v18, v12, v18, 0
	ds_read2_b32 v[79:80], v79 offset1:1
	v_dot4_i32_i8 v18, v13, v19, v18
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v18, v6, v20, v18
	v_dot4_i32_i8 v18, v7, v21, v18
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v18, v2, v22, v18
	v_dot4_i32_i8 v18, v3, v23, v18
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v18, v4, v79, v18
	v_dot4_i32_i8 v82, v9, v19, v82
	v_dot4_i32_i8 v18, v5, v80, v18
	v_dot4_i32_i8 v82, v10, v20, v82
	v_cvt_f32_i32_e32 v18, v18
	v_dot4_i32_i8 v82, v11, v21, v82
	v_dot4_i32_i8 v82, v14, v22, v82
	v_dot4_i32_i8 v82, v15, v23, v82
	v_mul_f32_e32 v19, v81, v77
	v_dot4_i32_i8 v82, v16, v79, v82
	v_fmac_f32_e32 v43, v19, v18
	v_add_u32_e32 v18, 0xd80, v76
	v_dot4_i32_i8 v82, v17, v80, v82
	v_mul_f32_e32 v83, v81, v78
	ds_read_b32 v81, v74 offset:3456
	ds_read2_b32 v[18:19], v18 offset1:1
	v_cvt_f32_i32_e32 v82, v82
	v_add_u32_e32 v20, 0xd88, v76
	ds_read2_b32 v[20:21], v20 offset1:1
	v_add_u32_e32 v22, 0xd90, v76
	ds_read2_b32 v[22:23], v22 offset1:1
	v_fmac_f32_e32 v44, v83, v82
	v_add_u32_e32 v79, 0xd98, v76
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v82, v8, v18, 0
	v_dot4_i32_i8 v18, v12, v18, 0
	ds_read2_b32 v[79:80], v79 offset1:1
	v_dot4_i32_i8 v18, v13, v19, v18
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v18, v6, v20, v18
	v_dot4_i32_i8 v18, v7, v21, v18
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v18, v2, v22, v18
	v_dot4_i32_i8 v18, v3, v23, v18
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v18, v4, v79, v18
	v_dot4_i32_i8 v82, v9, v19, v82
	v_dot4_i32_i8 v18, v5, v80, v18
	v_dot4_i32_i8 v82, v10, v20, v82
	v_cvt_f32_i32_e32 v18, v18
	v_dot4_i32_i8 v82, v11, v21, v82
	v_dot4_i32_i8 v82, v14, v22, v82
	v_dot4_i32_i8 v82, v15, v23, v82
	v_mul_f32_e32 v19, v81, v77
	v_dot4_i32_i8 v82, v16, v79, v82
	v_fmac_f32_e32 v41, v19, v18
	v_add_u32_e32 v18, 0x1200, v76
	v_dot4_i32_i8 v82, v17, v80, v82
	v_mul_f32_e32 v83, v81, v78
	ds_read_b32 v81, v74 offset:4608
	ds_read2_b32 v[18:19], v18 offset1:1
	v_cvt_f32_i32_e32 v82, v82
	v_add_u32_e32 v20, 0x1208, v76
	ds_read2_b32 v[20:21], v20 offset1:1
	v_add_u32_e32 v22, 0x1210, v76
	ds_read2_b32 v[22:23], v22 offset1:1
	v_fmac_f32_e32 v42, v83, v82
	v_add_u32_e32 v79, 0x1218, v76
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v82, v8, v18, 0
	v_dot4_i32_i8 v18, v12, v18, 0
	ds_read2_b32 v[79:80], v79 offset1:1
	v_dot4_i32_i8 v18, v13, v19, v18
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v18, v6, v20, v18
	v_dot4_i32_i8 v18, v7, v21, v18
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v18, v2, v22, v18
	v_dot4_i32_i8 v18, v3, v23, v18
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v18, v4, v79, v18
	v_dot4_i32_i8 v82, v9, v19, v82
	v_dot4_i32_i8 v18, v5, v80, v18
	v_dot4_i32_i8 v82, v10, v20, v82
	v_cvt_f32_i32_e32 v18, v18
	v_dot4_i32_i8 v82, v11, v21, v82
	v_dot4_i32_i8 v82, v14, v22, v82
	v_dot4_i32_i8 v82, v15, v23, v82
	v_mul_f32_e32 v19, v81, v77
	v_dot4_i32_i8 v82, v16, v79, v82
	v_fmac_f32_e32 v39, v19, v18
	v_add_u32_e32 v18, 0x1680, v76
	v_dot4_i32_i8 v82, v17, v80, v82
	v_mul_f32_e32 v83, v81, v78
	ds_read_b32 v81, v74 offset:5760
	ds_read2_b32 v[18:19], v18 offset1:1
	v_cvt_f32_i32_e32 v82, v82
	v_add_u32_e32 v20, 0x1688, v76
	ds_read2_b32 v[20:21], v20 offset1:1
	v_add_u32_e32 v22, 0x1690, v76
	ds_read2_b32 v[22:23], v22 offset1:1
	v_fmac_f32_e32 v40, v83, v82
	v_add_u32_e32 v79, 0x1698, v76
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v82, v8, v18, 0
	v_dot4_i32_i8 v18, v12, v18, 0
	ds_read2_b32 v[79:80], v79 offset1:1
	v_dot4_i32_i8 v18, v13, v19, v18
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v18, v6, v20, v18
	v_dot4_i32_i8 v18, v7, v21, v18
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v18, v2, v22, v18
	v_dot4_i32_i8 v18, v3, v23, v18
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v18, v4, v79, v18
	v_dot4_i32_i8 v82, v9, v19, v82
	v_dot4_i32_i8 v18, v5, v80, v18
	v_dot4_i32_i8 v82, v10, v20, v82
	v_cvt_f32_i32_e32 v18, v18
	v_dot4_i32_i8 v82, v11, v21, v82
	v_dot4_i32_i8 v82, v14, v22, v82
	v_dot4_i32_i8 v82, v15, v23, v82
	v_mul_f32_e32 v19, v81, v77
	v_dot4_i32_i8 v82, v16, v79, v82
	v_fmac_f32_e32 v37, v19, v18
	v_add_u32_e32 v18, 0x1b00, v76
	v_dot4_i32_i8 v82, v17, v80, v82
	v_mul_f32_e32 v83, v81, v78
	ds_read_b32 v81, v74 offset:6912
	ds_read2_b32 v[18:19], v18 offset1:1
	v_cvt_f32_i32_e32 v82, v82
	v_add_u32_e32 v20, 0x1b08, v76
	ds_read2_b32 v[20:21], v20 offset1:1
	v_add_u32_e32 v22, 0x1b10, v76
	ds_read2_b32 v[22:23], v22 offset1:1
	v_fmac_f32_e32 v38, v83, v82
	v_add_u32_e32 v79, 0x1b18, v76
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v82, v8, v18, 0
	v_dot4_i32_i8 v18, v12, v18, 0
	ds_read2_b32 v[79:80], v79 offset1:1
	v_dot4_i32_i8 v18, v13, v19, v18
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v18, v6, v20, v18
	v_dot4_i32_i8 v18, v7, v21, v18
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v18, v2, v22, v18
	v_dot4_i32_i8 v18, v3, v23, v18
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v18, v4, v79, v18
	v_dot4_i32_i8 v82, v9, v19, v82
	v_dot4_i32_i8 v18, v5, v80, v18
	v_dot4_i32_i8 v82, v10, v20, v82
	v_cvt_f32_i32_e32 v18, v18
	v_dot4_i32_i8 v82, v11, v21, v82
	v_dot4_i32_i8 v82, v14, v22, v82
	v_dot4_i32_i8 v82, v15, v23, v82
	v_mul_f32_e32 v19, v81, v77
	v_dot4_i32_i8 v82, v16, v79, v82
	v_fmac_f32_e32 v35, v19, v18
	v_add_u32_e32 v18, 0x1f80, v76
	v_dot4_i32_i8 v82, v17, v80, v82
	v_mul_f32_e32 v83, v81, v78
	ds_read_b32 v81, v74 offset:8064
	ds_read2_b32 v[18:19], v18 offset1:1
	v_cvt_f32_i32_e32 v82, v82
	v_add_u32_e32 v20, 0x1f88, v76
	ds_read2_b32 v[20:21], v20 offset1:1
	v_add_u32_e32 v22, 0x1f90, v76
	ds_read2_b32 v[22:23], v22 offset1:1
	v_fmac_f32_e32 v36, v83, v82
	v_add_u32_e32 v79, 0x1f98, v76
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v82, v8, v18, 0
	v_dot4_i32_i8 v18, v12, v18, 0
	ds_read2_b32 v[79:80], v79 offset1:1
	v_dot4_i32_i8 v18, v13, v19, v18
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v18, v6, v20, v18
	v_dot4_i32_i8 v18, v7, v21, v18
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v18, v2, v22, v18
	v_dot4_i32_i8 v18, v3, v23, v18
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v18, v4, v79, v18
	v_dot4_i32_i8 v82, v9, v19, v82
	v_dot4_i32_i8 v18, v5, v80, v18
	v_dot4_i32_i8 v82, v10, v20, v82
	v_cvt_f32_i32_e32 v18, v18
	v_dot4_i32_i8 v82, v11, v21, v82
	v_dot4_i32_i8 v82, v14, v22, v82
	v_dot4_i32_i8 v82, v15, v23, v82
	v_mul_f32_e32 v19, v81, v77
	v_dot4_i32_i8 v82, v16, v79, v82
	v_fmac_f32_e32 v33, v19, v18
	v_add_u32_e32 v18, 0x2400, v76
	v_dot4_i32_i8 v82, v17, v80, v82
	v_mul_f32_e32 v83, v81, v78
	ds_read_b32 v81, v74 offset:9216
	ds_read2_b32 v[18:19], v18 offset1:1
	v_cvt_f32_i32_e32 v82, v82
	v_add_u32_e32 v20, 0x2408, v76
	ds_read2_b32 v[20:21], v20 offset1:1
	v_add_u32_e32 v22, 0x2410, v76
	ds_read2_b32 v[22:23], v22 offset1:1
	v_fmac_f32_e32 v34, v83, v82
	v_add_u32_e32 v79, 0x2418, v76
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v82, v8, v18, 0
	v_dot4_i32_i8 v18, v12, v18, 0
	ds_read2_b32 v[79:80], v79 offset1:1
	v_dot4_i32_i8 v18, v13, v19, v18
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v18, v6, v20, v18
	v_dot4_i32_i8 v18, v7, v21, v18
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v18, v2, v22, v18
	v_dot4_i32_i8 v18, v3, v23, v18
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v18, v4, v79, v18
	v_dot4_i32_i8 v82, v9, v19, v82
	v_dot4_i32_i8 v18, v5, v80, v18
	v_dot4_i32_i8 v82, v10, v20, v82
	v_cvt_f32_i32_e32 v18, v18
	v_dot4_i32_i8 v82, v11, v21, v82
	v_dot4_i32_i8 v82, v14, v22, v82
	v_dot4_i32_i8 v82, v15, v23, v82
	v_mul_f32_e32 v19, v81, v77
	v_dot4_i32_i8 v82, v16, v79, v82
	v_fmac_f32_e32 v31, v19, v18
	v_add_u32_e32 v18, 0x2880, v76
	v_dot4_i32_i8 v82, v17, v80, v82
	v_mul_f32_e32 v83, v81, v78
	ds_read_b32 v81, v74 offset:10368
	ds_read2_b32 v[18:19], v18 offset1:1
	v_cvt_f32_i32_e32 v82, v82
	v_add_u32_e32 v20, 0x2888, v76
	ds_read2_b32 v[20:21], v20 offset1:1
	v_add_u32_e32 v22, 0x2890, v76
	ds_read2_b32 v[22:23], v22 offset1:1
	v_fmac_f32_e32 v32, v83, v82
	v_add_u32_e32 v79, 0x2898, v76
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v82, v8, v18, 0
	v_dot4_i32_i8 v18, v12, v18, 0
	ds_read2_b32 v[79:80], v79 offset1:1
	v_dot4_i32_i8 v18, v13, v19, v18
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v18, v6, v20, v18
	v_dot4_i32_i8 v18, v7, v21, v18
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v18, v2, v22, v18
	v_dot4_i32_i8 v18, v3, v23, v18
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v18, v4, v79, v18
	v_dot4_i32_i8 v82, v9, v19, v82
	v_dot4_i32_i8 v18, v5, v80, v18
	v_dot4_i32_i8 v82, v10, v20, v82
	v_cvt_f32_i32_e32 v18, v18
	v_dot4_i32_i8 v82, v11, v21, v82
	v_dot4_i32_i8 v82, v14, v22, v82
	v_dot4_i32_i8 v82, v15, v23, v82
	v_mul_f32_e32 v19, v81, v77
	v_dot4_i32_i8 v82, v16, v79, v82
	v_fmac_f32_e32 v29, v19, v18
	v_add_u32_e32 v18, 0x2d00, v76
	v_dot4_i32_i8 v82, v17, v80, v82
	v_mul_f32_e32 v83, v81, v78
	ds_read_b32 v81, v74 offset:11520
	ds_read2_b32 v[18:19], v18 offset1:1
	v_cvt_f32_i32_e32 v82, v82
	v_add_u32_e32 v20, 0x2d08, v76
	ds_read2_b32 v[20:21], v20 offset1:1
	v_add_u32_e32 v22, 0x2d10, v76
	ds_read2_b32 v[22:23], v22 offset1:1
	v_fmac_f32_e32 v30, v83, v82
	v_add_u32_e32 v79, 0x2d18, v76
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v82, v8, v18, 0
	v_dot4_i32_i8 v18, v12, v18, 0
	ds_read2_b32 v[79:80], v79 offset1:1
	v_dot4_i32_i8 v18, v13, v19, v18
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v18, v6, v20, v18
	v_dot4_i32_i8 v18, v7, v21, v18
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v18, v2, v22, v18
	v_dot4_i32_i8 v18, v3, v23, v18
	v_dot4_i32_i8 v82, v9, v19, v82
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v18, v4, v79, v18
	v_dot4_i32_i8 v82, v10, v20, v82
	v_dot4_i32_i8 v18, v5, v80, v18
	v_dot4_i32_i8 v82, v11, v21, v82
	v_cvt_f32_i32_e32 v18, v18
	v_dot4_i32_i8 v82, v14, v22, v82
	v_dot4_i32_i8 v82, v15, v23, v82
	v_dot4_i32_i8 v82, v16, v79, v82
	v_mul_f32_e32 v19, v81, v77
	v_dot4_i32_i8 v82, v17, v80, v82
	v_mul_f32_e32 v83, v81, v78
	v_fmac_f32_e32 v27, v19, v18
	v_add_u32_e32 v18, 0x3180, v76
	ds_read_b32 v79, v74 offset:12672
	ds_read2_b32 v[80:81], v18 offset1:1
	v_add_u32_e32 v18, 0x3188, v76
	ds_read2_b32 v[22:23], v18 offset1:1
	v_add_u32_e32 v20, 0x3198, v76
	ds_read2_b32 v[20:21], v20 offset1:1
	v_add_u32_e32 v18, 0x3190, v76
	ds_read2_b32 v[18:19], v18 offset1:1
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v8, v8, v80, 0
	v_dot4_i32_i8 v8, v9, v81, v8
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v8, v10, v22, v8
	v_dot4_i32_i8 v8, v11, v23, v8
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v8, v14, v18, v8
	v_dot4_i32_i8 v8, v15, v19, v8
	v_dot4_i32_i8 v8, v16, v20, v8
	v_dot4_i32_i8 v8, v17, v21, v8
	v_cvt_f32_i32_e32 v8, v8
	v_mul_f32_e32 v9, v79, v78
	v_cvt_f32_i32_e32 v82, v82
	v_add_u32_e32 v76, 32, v76
	v_fmac_f32_e32 v26, v9, v8
	v_dot4_i32_i8 v8, v12, v80, 0
	v_dot4_i32_i8 v8, v13, v81, v8
	v_dot4_i32_i8 v6, v6, v22, v8
	v_dot4_i32_i8 v6, v7, v23, v6
	v_dot4_i32_i8 v2, v2, v18, v6
	v_dot4_i32_i8 v2, v3, v19, v2
	v_dot4_i32_i8 v2, v4, v20, v2
	v_dot4_i32_i8 v2, v5, v21, v2
	v_cvt_f32_i32_e32 v2, v2
	v_mul_f32_e32 v3, v79, v77
	v_fmac_f32_e32 v28, v83, v82
	v_add_u32_e32 v74, 4, v74
	v_fmac_f32_e32 v25, v3, v2
	s_cbranch_scc1 .LBB44_13
; %bb.14:                               ; %_ZL22vec_dot_q8_0_q8_1_dp4aILi96ELi128ELi8EEvPKiS1_Pfi.exit.i
                                        ;   in Loop: Header=BB44_12 Depth=1
	s_add_u32 s52, s52, s8
	s_addc_u32 s53, s53, s9
	v_mov_b32_e32 v2, s53
	v_add_co_u32_e32 v6, vcc, s52, v70
	v_addc_co_u32_e32 v7, vcc, 0, v2, vcc
	v_add_co_u32_e32 v2, vcc, s3, v6
	v_addc_co_u32_e32 v3, vcc, 0, v7, vcc
	v_add_co_u32_e32 v4, vcc, 0x2000, v6
	v_addc_co_u32_e32 v5, vcc, 0, v7, vcc
	v_add_co_u32_e32 v6, vcc, 0x3000, v6
	s_barrier
	v_addc_co_u32_e32 v7, vcc, 0, v7, vcc
	global_load_dword v8, v70, s[52:53]
	global_load_dword v9, v70, s[52:53] offset:2048
	global_load_dword v10, v[2:3], off
	s_nop 0
	global_load_dword v2, v[2:3], off offset:2048
	s_nop 0
	global_load_dword v3, v[4:5], off
	s_nop 0
	global_load_dword v4, v[4:5], off offset:2048
	s_nop 0
	global_load_dword v5, v[6:7], off
	s_mov_b32 s52, -8
	v_mov_b32_e32 v18, v59
	v_mov_b32_e32 v19, v66
	v_mov_b32_e32 v20, v65
	v_mov_b32_e32 v21, v64
	v_mov_b32_e32 v22, v57
	s_waitcnt vmcnt(5)
	ds_write2st64_b32 v71, v8, v9 offset0:1 offset1:9
	s_waitcnt vmcnt(3)
	ds_write2st64_b32 v71, v10, v2 offset0:17 offset1:25
	s_waitcnt vmcnt(1)
	ds_write2st64_b32 v71, v3, v4 offset0:33 offset1:41
	s_waitcnt vmcnt(0)
	ds_write_b32 v69, v5 offset:12672
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB44_15:                              ;   Parent Loop BB44_12 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ds_read_b32 v76, v18
	ds_read2_b32 v[2:3], v22 offset1:1
	ds_read2_b32 v[4:5], v22 offset0:2 offset1:3
	ds_read2_b32 v[72:73], v22 offset0:4 offset1:5
	ds_read2_b32 v[74:75], v22 offset0:6 offset1:7
	ds_read_b32 v71, v19
	ds_read2_b32 v[8:9], v21 offset1:1
	ds_read2_b32 v[10:11], v21 offset0:2 offset1:3
	ds_read2_b32 v[14:15], v21 offset0:4 offset1:5
	ds_read2_b32 v[16:17], v21 offset0:6 offset1:7
	s_waitcnt lgkmcnt(4)
	v_mul_f32_e32 v7, v71, v76
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v6, v8, v2, 0
	v_dot4_i32_i8 v6, v9, v3, v6
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v6, v10, v4, v6
	v_dot4_i32_i8 v6, v11, v5, v6
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v6, v14, v72, v6
	v_dot4_i32_i8 v6, v15, v73, v6
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v6, v16, v74, v6
	v_dot4_i32_i8 v6, v17, v75, v6
	v_cvt_f32_i32_e32 v6, v6
	v_add_u32_e32 v78, 0x498, v22
	s_add_i32 s52, s52, 8
	v_add_u32_e32 v19, 4, v19
	v_fmac_f32_e32 v50, v7, v6
	v_add_u32_e32 v6, 0x4100, v21
	ds_read_b32 v23, v20
	ds_read2_b32 v[12:13], v6 offset1:1
	v_add_u32_e32 v20, 4, v20
	s_cmp_lt_u32 s52, 24
	ds_read2_b32 v[78:79], v78 offset1:1
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v2, v12, v2, 0
	v_dot4_i32_i8 v2, v13, v3, v2
	v_add_u32_e32 v3, 0x4108, v21
	ds_read2_b32 v[6:7], v3 offset1:1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v2, v6, v4, v2
	v_dot4_i32_i8 v4, v7, v5, v2
	v_add_u32_e32 v2, 0x4110, v21
	ds_read2_b32 v[2:3], v2 offset1:1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v4, v2, v72, v4
	v_dot4_i32_i8 v72, v3, v73, v4
	v_add_u32_e32 v4, 0x4118, v21
	ds_read2_b32 v[4:5], v4 offset1:1
	v_mul_f32_e32 v73, v23, v76
	v_add_u32_e32 v76, 0x490, v22
	ds_read2_b32 v[76:77], v76 offset1:1
	v_add_u32_e32 v21, 32, v21
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v72, v4, v74, v72
	v_dot4_i32_i8 v72, v5, v75, v72
	v_cvt_f32_i32_e32 v72, v72
	v_add_u32_e32 v74, 0x488, v22
	ds_read2_b32 v[74:75], v74 offset1:1
	v_fmac_f32_e32 v48, v73, v72
	v_add_u32_e32 v72, 0x480, v22
	ds_read_b32 v80, v18 offset:1152
	ds_read2_b32 v[72:73], v72 offset1:1
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v82, v80, v71
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v81, v8, v72, 0
	v_dot4_i32_i8 v72, v12, v72, 0
	v_dot4_i32_i8 v72, v13, v73, v72
	v_dot4_i32_i8 v72, v6, v74, v72
	v_dot4_i32_i8 v72, v7, v75, v72
	v_dot4_i32_i8 v72, v2, v76, v72
	v_dot4_i32_i8 v72, v3, v77, v72
	v_dot4_i32_i8 v72, v4, v78, v72
	v_dot4_i32_i8 v81, v9, v73, v81
	v_dot4_i32_i8 v72, v5, v79, v72
	v_dot4_i32_i8 v81, v10, v74, v81
	v_cvt_f32_i32_e32 v72, v72
	v_dot4_i32_i8 v81, v11, v75, v81
	v_dot4_i32_i8 v81, v14, v76, v81
	v_dot4_i32_i8 v81, v15, v77, v81
	v_mul_f32_e32 v73, v80, v23
	v_dot4_i32_i8 v81, v16, v78, v81
	v_fmac_f32_e32 v45, v73, v72
	v_add_u32_e32 v72, 0x900, v22
	v_dot4_i32_i8 v81, v17, v79, v81
	ds_read_b32 v80, v18 offset:2304
	ds_read2_b32 v[72:73], v72 offset1:1
	v_cvt_f32_i32_e32 v81, v81
	v_add_u32_e32 v74, 0x908, v22
	ds_read2_b32 v[74:75], v74 offset1:1
	v_add_u32_e32 v76, 0x910, v22
	ds_read2_b32 v[76:77], v76 offset1:1
	v_fmac_f32_e32 v46, v82, v81
	v_add_u32_e32 v78, 0x918, v22
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v8, v72, 0
	v_dot4_i32_i8 v72, v12, v72, 0
	ds_read2_b32 v[78:79], v78 offset1:1
	v_dot4_i32_i8 v72, v13, v73, v72
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v72, v6, v74, v72
	v_dot4_i32_i8 v72, v7, v75, v72
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v72, v2, v76, v72
	v_dot4_i32_i8 v72, v3, v77, v72
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v72, v4, v78, v72
	v_dot4_i32_i8 v81, v9, v73, v81
	v_dot4_i32_i8 v72, v5, v79, v72
	v_dot4_i32_i8 v81, v10, v74, v81
	v_cvt_f32_i32_e32 v72, v72
	v_dot4_i32_i8 v81, v11, v75, v81
	v_dot4_i32_i8 v81, v14, v76, v81
	v_dot4_i32_i8 v81, v15, v77, v81
	v_mul_f32_e32 v73, v80, v23
	v_dot4_i32_i8 v81, v16, v78, v81
	v_fmac_f32_e32 v43, v73, v72
	v_add_u32_e32 v72, 0xd80, v22
	v_dot4_i32_i8 v81, v17, v79, v81
	v_mul_f32_e32 v82, v80, v71
	ds_read_b32 v80, v18 offset:3456
	ds_read2_b32 v[72:73], v72 offset1:1
	v_cvt_f32_i32_e32 v81, v81
	v_add_u32_e32 v74, 0xd88, v22
	ds_read2_b32 v[74:75], v74 offset1:1
	v_add_u32_e32 v76, 0xd90, v22
	ds_read2_b32 v[76:77], v76 offset1:1
	v_fmac_f32_e32 v44, v82, v81
	v_add_u32_e32 v78, 0xd98, v22
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v8, v72, 0
	v_dot4_i32_i8 v72, v12, v72, 0
	ds_read2_b32 v[78:79], v78 offset1:1
	v_dot4_i32_i8 v72, v13, v73, v72
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v72, v6, v74, v72
	v_dot4_i32_i8 v72, v7, v75, v72
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v72, v2, v76, v72
	v_dot4_i32_i8 v72, v3, v77, v72
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v72, v4, v78, v72
	v_dot4_i32_i8 v81, v9, v73, v81
	v_dot4_i32_i8 v72, v5, v79, v72
	v_dot4_i32_i8 v81, v10, v74, v81
	v_cvt_f32_i32_e32 v72, v72
	v_dot4_i32_i8 v81, v11, v75, v81
	v_dot4_i32_i8 v81, v14, v76, v81
	v_dot4_i32_i8 v81, v15, v77, v81
	v_mul_f32_e32 v73, v80, v23
	v_dot4_i32_i8 v81, v16, v78, v81
	v_fmac_f32_e32 v41, v73, v72
	v_add_u32_e32 v72, 0x1200, v22
	v_dot4_i32_i8 v81, v17, v79, v81
	v_mul_f32_e32 v82, v80, v71
	ds_read_b32 v80, v18 offset:4608
	ds_read2_b32 v[72:73], v72 offset1:1
	v_cvt_f32_i32_e32 v81, v81
	v_add_u32_e32 v74, 0x1208, v22
	ds_read2_b32 v[74:75], v74 offset1:1
	v_add_u32_e32 v76, 0x1210, v22
	ds_read2_b32 v[76:77], v76 offset1:1
	v_fmac_f32_e32 v42, v82, v81
	v_add_u32_e32 v78, 0x1218, v22
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v8, v72, 0
	v_dot4_i32_i8 v72, v12, v72, 0
	ds_read2_b32 v[78:79], v78 offset1:1
	v_dot4_i32_i8 v72, v13, v73, v72
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v72, v6, v74, v72
	v_dot4_i32_i8 v72, v7, v75, v72
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v72, v2, v76, v72
	v_dot4_i32_i8 v72, v3, v77, v72
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v72, v4, v78, v72
	v_dot4_i32_i8 v81, v9, v73, v81
	v_dot4_i32_i8 v72, v5, v79, v72
	v_dot4_i32_i8 v81, v10, v74, v81
	v_cvt_f32_i32_e32 v72, v72
	v_dot4_i32_i8 v81, v11, v75, v81
	v_dot4_i32_i8 v81, v14, v76, v81
	v_dot4_i32_i8 v81, v15, v77, v81
	v_mul_f32_e32 v73, v80, v23
	v_dot4_i32_i8 v81, v16, v78, v81
	v_fmac_f32_e32 v39, v73, v72
	v_add_u32_e32 v72, 0x1680, v22
	v_dot4_i32_i8 v81, v17, v79, v81
	v_mul_f32_e32 v82, v80, v71
	ds_read_b32 v80, v18 offset:5760
	ds_read2_b32 v[72:73], v72 offset1:1
	v_cvt_f32_i32_e32 v81, v81
	v_add_u32_e32 v74, 0x1688, v22
	ds_read2_b32 v[74:75], v74 offset1:1
	v_add_u32_e32 v76, 0x1690, v22
	ds_read2_b32 v[76:77], v76 offset1:1
	v_fmac_f32_e32 v40, v82, v81
	v_add_u32_e32 v78, 0x1698, v22
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v8, v72, 0
	v_dot4_i32_i8 v72, v12, v72, 0
	ds_read2_b32 v[78:79], v78 offset1:1
	v_dot4_i32_i8 v72, v13, v73, v72
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v72, v6, v74, v72
	v_dot4_i32_i8 v72, v7, v75, v72
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v72, v2, v76, v72
	v_dot4_i32_i8 v72, v3, v77, v72
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v72, v4, v78, v72
	v_dot4_i32_i8 v81, v9, v73, v81
	v_dot4_i32_i8 v72, v5, v79, v72
	v_dot4_i32_i8 v81, v10, v74, v81
	v_cvt_f32_i32_e32 v72, v72
	v_dot4_i32_i8 v81, v11, v75, v81
	v_dot4_i32_i8 v81, v14, v76, v81
	v_dot4_i32_i8 v81, v15, v77, v81
	v_mul_f32_e32 v73, v80, v23
	v_dot4_i32_i8 v81, v16, v78, v81
	v_fmac_f32_e32 v37, v73, v72
	v_add_u32_e32 v72, 0x1b00, v22
	v_dot4_i32_i8 v81, v17, v79, v81
	v_mul_f32_e32 v82, v80, v71
	ds_read_b32 v80, v18 offset:6912
	ds_read2_b32 v[72:73], v72 offset1:1
	v_cvt_f32_i32_e32 v81, v81
	v_add_u32_e32 v74, 0x1b08, v22
	ds_read2_b32 v[74:75], v74 offset1:1
	v_add_u32_e32 v76, 0x1b10, v22
	ds_read2_b32 v[76:77], v76 offset1:1
	v_fmac_f32_e32 v38, v82, v81
	v_add_u32_e32 v78, 0x1b18, v22
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v8, v72, 0
	v_dot4_i32_i8 v72, v12, v72, 0
	ds_read2_b32 v[78:79], v78 offset1:1
	v_dot4_i32_i8 v72, v13, v73, v72
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v72, v6, v74, v72
	v_dot4_i32_i8 v72, v7, v75, v72
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v72, v2, v76, v72
	v_dot4_i32_i8 v72, v3, v77, v72
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v72, v4, v78, v72
	v_dot4_i32_i8 v81, v9, v73, v81
	v_dot4_i32_i8 v72, v5, v79, v72
	v_dot4_i32_i8 v81, v10, v74, v81
	v_cvt_f32_i32_e32 v72, v72
	v_dot4_i32_i8 v81, v11, v75, v81
	v_dot4_i32_i8 v81, v14, v76, v81
	v_dot4_i32_i8 v81, v15, v77, v81
	v_mul_f32_e32 v73, v80, v23
	v_dot4_i32_i8 v81, v16, v78, v81
	v_fmac_f32_e32 v35, v73, v72
	v_add_u32_e32 v72, 0x1f80, v22
	v_dot4_i32_i8 v81, v17, v79, v81
	v_mul_f32_e32 v82, v80, v71
	ds_read_b32 v80, v18 offset:8064
	ds_read2_b32 v[72:73], v72 offset1:1
	v_cvt_f32_i32_e32 v81, v81
	v_add_u32_e32 v74, 0x1f88, v22
	ds_read2_b32 v[74:75], v74 offset1:1
	v_add_u32_e32 v76, 0x1f90, v22
	ds_read2_b32 v[76:77], v76 offset1:1
	v_fmac_f32_e32 v36, v82, v81
	v_add_u32_e32 v78, 0x1f98, v22
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v8, v72, 0
	v_dot4_i32_i8 v72, v12, v72, 0
	ds_read2_b32 v[78:79], v78 offset1:1
	v_dot4_i32_i8 v72, v13, v73, v72
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v72, v6, v74, v72
	v_dot4_i32_i8 v72, v7, v75, v72
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v72, v2, v76, v72
	v_dot4_i32_i8 v72, v3, v77, v72
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v72, v4, v78, v72
	v_dot4_i32_i8 v81, v9, v73, v81
	v_dot4_i32_i8 v72, v5, v79, v72
	v_dot4_i32_i8 v81, v10, v74, v81
	v_cvt_f32_i32_e32 v72, v72
	v_dot4_i32_i8 v81, v11, v75, v81
	v_dot4_i32_i8 v81, v14, v76, v81
	v_dot4_i32_i8 v81, v15, v77, v81
	v_mul_f32_e32 v73, v80, v23
	v_dot4_i32_i8 v81, v16, v78, v81
	v_fmac_f32_e32 v33, v73, v72
	v_add_u32_e32 v72, 0x2400, v22
	v_dot4_i32_i8 v81, v17, v79, v81
	v_mul_f32_e32 v82, v80, v71
	ds_read_b32 v80, v18 offset:9216
	ds_read2_b32 v[72:73], v72 offset1:1
	v_cvt_f32_i32_e32 v81, v81
	v_add_u32_e32 v74, 0x2408, v22
	ds_read2_b32 v[74:75], v74 offset1:1
	v_add_u32_e32 v76, 0x2410, v22
	ds_read2_b32 v[76:77], v76 offset1:1
	v_fmac_f32_e32 v34, v82, v81
	v_add_u32_e32 v78, 0x2418, v22
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v8, v72, 0
	v_dot4_i32_i8 v72, v12, v72, 0
	ds_read2_b32 v[78:79], v78 offset1:1
	v_dot4_i32_i8 v72, v13, v73, v72
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v72, v6, v74, v72
	v_dot4_i32_i8 v72, v7, v75, v72
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v72, v2, v76, v72
	v_dot4_i32_i8 v72, v3, v77, v72
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v72, v4, v78, v72
	v_dot4_i32_i8 v81, v9, v73, v81
	v_dot4_i32_i8 v72, v5, v79, v72
	v_dot4_i32_i8 v81, v10, v74, v81
	v_cvt_f32_i32_e32 v72, v72
	v_dot4_i32_i8 v81, v11, v75, v81
	v_dot4_i32_i8 v81, v14, v76, v81
	v_dot4_i32_i8 v81, v15, v77, v81
	v_mul_f32_e32 v73, v80, v23
	v_dot4_i32_i8 v81, v16, v78, v81
	v_fmac_f32_e32 v31, v73, v72
	v_add_u32_e32 v72, 0x2880, v22
	v_dot4_i32_i8 v81, v17, v79, v81
	v_mul_f32_e32 v82, v80, v71
	ds_read_b32 v80, v18 offset:10368
	ds_read2_b32 v[72:73], v72 offset1:1
	v_cvt_f32_i32_e32 v81, v81
	v_add_u32_e32 v74, 0x2888, v22
	ds_read2_b32 v[74:75], v74 offset1:1
	v_add_u32_e32 v76, 0x2890, v22
	ds_read2_b32 v[76:77], v76 offset1:1
	v_fmac_f32_e32 v32, v82, v81
	v_add_u32_e32 v78, 0x2898, v22
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v8, v72, 0
	v_dot4_i32_i8 v72, v12, v72, 0
	ds_read2_b32 v[78:79], v78 offset1:1
	v_dot4_i32_i8 v72, v13, v73, v72
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v72, v6, v74, v72
	v_dot4_i32_i8 v72, v7, v75, v72
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v72, v2, v76, v72
	v_dot4_i32_i8 v72, v3, v77, v72
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v72, v4, v78, v72
	v_dot4_i32_i8 v81, v9, v73, v81
	v_dot4_i32_i8 v72, v5, v79, v72
	v_dot4_i32_i8 v81, v10, v74, v81
	v_cvt_f32_i32_e32 v72, v72
	v_dot4_i32_i8 v81, v11, v75, v81
	v_dot4_i32_i8 v81, v14, v76, v81
	v_dot4_i32_i8 v81, v15, v77, v81
	v_mul_f32_e32 v73, v80, v23
	v_dot4_i32_i8 v81, v16, v78, v81
	v_fmac_f32_e32 v29, v73, v72
	v_add_u32_e32 v72, 0x2d00, v22
	v_dot4_i32_i8 v81, v17, v79, v81
	v_mul_f32_e32 v82, v80, v71
	ds_read_b32 v80, v18 offset:11520
	ds_read2_b32 v[72:73], v72 offset1:1
	v_cvt_f32_i32_e32 v81, v81
	v_add_u32_e32 v74, 0x2d08, v22
	ds_read2_b32 v[74:75], v74 offset1:1
	v_add_u32_e32 v76, 0x2d10, v22
	ds_read2_b32 v[76:77], v76 offset1:1
	v_fmac_f32_e32 v30, v82, v81
	v_add_u32_e32 v78, 0x2d18, v22
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v8, v72, 0
	v_dot4_i32_i8 v72, v12, v72, 0
	ds_read2_b32 v[78:79], v78 offset1:1
	v_dot4_i32_i8 v72, v13, v73, v72
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v72, v6, v74, v72
	v_dot4_i32_i8 v72, v7, v75, v72
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v72, v2, v76, v72
	v_dot4_i32_i8 v72, v3, v77, v72
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v72, v4, v78, v72
	v_dot4_i32_i8 v72, v5, v79, v72
	v_cvt_f32_i32_e32 v72, v72
	v_dot4_i32_i8 v81, v9, v73, v81
	v_mul_f32_e32 v73, v80, v23
	v_dot4_i32_i8 v81, v10, v74, v81
	v_fmac_f32_e32 v27, v73, v72
	v_add_u32_e32 v73, 0x3180, v22
	ds_read_b32 v72, v18 offset:12672
	ds_read2_b32 v[73:74], v73 offset1:1
	v_dot4_i32_i8 v81, v11, v75, v81
	v_add_u32_e32 v75, 0x3188, v22
	v_dot4_i32_i8 v81, v14, v76, v81
	ds_read2_b32 v[75:76], v75 offset1:1
	v_dot4_i32_i8 v81, v15, v77, v81
	v_add_u32_e32 v77, 0x3190, v22
	v_dot4_i32_i8 v81, v16, v78, v81
	ds_read2_b32 v[77:78], v77 offset1:1
	v_dot4_i32_i8 v81, v17, v79, v81
	v_add_u32_e32 v79, 0x3198, v22
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v8, v8, v73, 0
	v_mul_f32_e32 v82, v80, v71
	ds_read2_b32 v[79:80], v79 offset1:1
	v_dot4_i32_i8 v8, v9, v74, v8
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v8, v10, v75, v8
	v_dot4_i32_i8 v8, v11, v76, v8
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v8, v14, v77, v8
	v_dot4_i32_i8 v8, v15, v78, v8
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v8, v16, v79, v8
	v_dot4_i32_i8 v8, v17, v80, v8
	v_cvt_f32_i32_e32 v8, v8
	v_mul_f32_e32 v9, v72, v71
	v_cvt_f32_i32_e32 v81, v81
	v_add_u32_e32 v22, 32, v22
	v_fmac_f32_e32 v26, v9, v8
	v_dot4_i32_i8 v8, v12, v73, 0
	v_dot4_i32_i8 v8, v13, v74, v8
	v_dot4_i32_i8 v6, v6, v75, v8
	v_dot4_i32_i8 v6, v7, v76, v6
	v_dot4_i32_i8 v2, v2, v77, v6
	v_dot4_i32_i8 v2, v3, v78, v2
	v_dot4_i32_i8 v2, v4, v79, v2
	v_dot4_i32_i8 v2, v5, v80, v2
	v_cvt_f32_i32_e32 v2, v2
	v_mul_f32_e32 v3, v72, v23
	v_fmac_f32_e32 v28, v82, v81
	v_add_u32_e32 v18, 4, v18
	v_fmac_f32_e32 v25, v3, v2
	s_cbranch_scc1 .LBB44_15
; %bb.16:                               ; %_ZL22vec_dot_q8_0_q8_1_dp4aILi96ELi128ELi8EEvPKiS1_Pfi.exit104.i
                                        ;   in Loop: Header=BB44_12 Depth=1
	s_add_i32 s57, s57, 8
	s_cmp_ge_i32 s57, s22
	s_barrier
	s_cbranch_scc0 .LBB44_12
	s_branch .LBB44_20
.LBB44_17:                              ; %Flow3251
	s_mov_b32 s35, 0
	s_and_b64 vcc, exec, s[54:55]
	s_cbranch_vccnz .LBB44_10
.LBB44_18:
                                        ; implicit-def: $vgpr25
                                        ; implicit-def: $vgpr26
                                        ; implicit-def: $sgpr8_sgpr9
                                        ; implicit-def: $vgpr2
	s_branch .LBB44_126
.LBB44_19:
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v50, 0
.LBB44_20:                              ; %._crit_edge.i
	s_not_b32 s2, s7
	s_add_i32 s33, s33, s2
	v_cmp_ge_i32_e32 vcc, s33, v1
	s_mov_b64 s[2:3], 0
                                        ; implicit-def: $sgpr8_sgpr9
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[52:53], vcc
	s_cbranch_execz .LBB44_44
; %bb.21:                               ; %.preheader.i.i
	v_lshl_add_u32 v3, v1, 2, 0
	ds_read_b32 v2, v3
	s_add_i32 s2, s35, s54
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 2
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[4:5], s[8:9], v2, s27, v[0:1]
	s_add_u32 s8, s28, s2
	s_addc_u32 s9, s29, s3
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_mov_b32_e32 v2, s9
	v_add_co_u32_e32 v4, vcc, s8, v4
	v_addc_co_u32_e32 v5, vcc, v2, v5, vcc
	v_add_u32_e32 v2, 8, v1
	v_cmp_ge_u32_e32 vcc, s33, v2
	s_mov_b64 s[2:3], 0
	global_store_dword v[4:5], v50, off
	global_store_dword v[4:5], v48, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[54:55], vcc
	s_cbranch_execz .LBB44_43
; %bb.22:                               ; %.preheader.1.i.i
	ds_read_b32 v2, v3 offset:32
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[4:5], s[2:3], v2, s27, v[0:1]
	v_add_u32_e32 v5, 16, v1
	v_cmp_ge_u32_e32 vcc, s33, v5
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_mov_b32_e32 v2, s9
	v_add_co_u32_e64 v4, s[2:3], s8, v4
	v_addc_co_u32_e64 v5, s[2:3], v2, v5, s[2:3]
	s_mov_b64 s[2:3], 0
	global_store_dword v[4:5], v46, off
	global_store_dword v[4:5], v45, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[56:57], vcc
	s_cbranch_execz .LBB44_42
; %bb.23:                               ; %.preheader.2.i.i
	ds_read_b32 v2, v3 offset:64
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[4:5], s[2:3], v2, s27, v[0:1]
	v_add_u32_e32 v5, 24, v1
	v_cmp_ge_u32_e32 vcc, s33, v5
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_mov_b32_e32 v2, s9
	v_add_co_u32_e64 v4, s[2:3], s8, v4
	v_addc_co_u32_e64 v5, s[2:3], v2, v5, s[2:3]
	s_mov_b64 s[2:3], 0
	global_store_dword v[4:5], v44, off
	global_store_dword v[4:5], v43, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[58:59], vcc
	s_cbranch_execz .LBB44_41
; %bb.24:                               ; %.preheader.3.i.i
	ds_read_b32 v2, v3 offset:96
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[4:5], s[2:3], v2, s27, v[0:1]
	v_add_u32_e32 v5, 32, v1
	v_cmp_ge_u32_e32 vcc, s33, v5
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_mov_b32_e32 v2, s9
	v_add_co_u32_e64 v4, s[2:3], s8, v4
	v_addc_co_u32_e64 v5, s[2:3], v2, v5, s[2:3]
	s_mov_b64 s[2:3], 0
	global_store_dword v[4:5], v42, off
	global_store_dword v[4:5], v41, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[60:61], vcc
	s_cbranch_execz .LBB44_40
; %bb.25:                               ; %.preheader.4.i.i
	ds_read_b32 v2, v3 offset:128
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[4:5], s[2:3], v2, s27, v[0:1]
	v_add_u32_e32 v5, 40, v1
	v_cmp_ge_u32_e32 vcc, s33, v5
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_mov_b32_e32 v2, s9
	v_add_co_u32_e64 v4, s[2:3], s8, v4
	v_addc_co_u32_e64 v5, s[2:3], v2, v5, s[2:3]
	s_mov_b64 s[2:3], 0
	global_store_dword v[4:5], v40, off
	global_store_dword v[4:5], v39, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[62:63], vcc
	s_cbranch_execz .LBB44_39
; %bb.26:                               ; %.preheader.5.i.i
	ds_read_b32 v2, v3 offset:160
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[4:5], s[2:3], v2, s27, v[0:1]
	v_add_u32_e32 v5, 48, v1
	v_cmp_ge_u32_e32 vcc, s33, v5
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_mov_b32_e32 v2, s9
	v_add_co_u32_e64 v4, s[2:3], s8, v4
	v_addc_co_u32_e64 v5, s[2:3], v2, v5, s[2:3]
	s_mov_b64 s[2:3], 0
	global_store_dword v[4:5], v38, off
	global_store_dword v[4:5], v37, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[64:65], vcc
	s_cbranch_execz .LBB44_38
; %bb.27:                               ; %.preheader.6.i.i
	ds_read_b32 v2, v3 offset:192
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[4:5], s[2:3], v2, s27, v[0:1]
	v_add_u32_e32 v5, 56, v1
	v_cmp_ge_u32_e32 vcc, s33, v5
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_mov_b32_e32 v2, s9
	v_add_co_u32_e64 v4, s[2:3], s8, v4
	v_addc_co_u32_e64 v5, s[2:3], v2, v5, s[2:3]
	s_mov_b64 s[2:3], 0
	global_store_dword v[4:5], v36, off
	global_store_dword v[4:5], v35, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[66:67], vcc
	s_cbranch_execz .LBB44_37
; %bb.28:                               ; %.preheader.7.i.i
	ds_read_b32 v2, v3 offset:224
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[4:5], s[2:3], v2, s27, v[0:1]
	v_add_u32_e32 v5, 64, v1
	v_cmp_ge_u32_e32 vcc, s33, v5
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_mov_b32_e32 v2, s9
	v_add_co_u32_e64 v4, s[2:3], s8, v4
	v_addc_co_u32_e64 v5, s[2:3], v2, v5, s[2:3]
	s_mov_b64 s[2:3], 0
	global_store_dword v[4:5], v34, off
	global_store_dword v[4:5], v33, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[68:69], vcc
	s_cbranch_execz .LBB44_36
; %bb.29:                               ; %.preheader.8.i.i
	ds_read_b32 v2, v3 offset:256
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[4:5], s[2:3], v2, s27, v[0:1]
	v_add_u32_e32 v5, 0x48, v1
	v_cmp_ge_u32_e32 vcc, s33, v5
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_mov_b32_e32 v2, s9
	v_add_co_u32_e64 v4, s[2:3], s8, v4
	v_addc_co_u32_e64 v5, s[2:3], v2, v5, s[2:3]
	s_mov_b64 s[2:3], 0
	global_store_dword v[4:5], v32, off
	global_store_dword v[4:5], v31, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[70:71], vcc
	s_cbranch_execz .LBB44_35
; %bb.30:                               ; %.preheader.9.i.i
	ds_read_b32 v2, v3 offset:288
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[4:5], s[2:3], v2, s27, v[0:1]
	v_add_u32_e32 v5, 0x50, v1
	v_cmp_ge_u32_e32 vcc, s33, v5
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_mov_b32_e32 v2, s9
	v_add_co_u32_e64 v4, s[2:3], s8, v4
	v_addc_co_u32_e64 v5, s[2:3], v2, v5, s[2:3]
	s_mov_b64 s[2:3], 0
	global_store_dword v[4:5], v30, off
	global_store_dword v[4:5], v29, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[72:73], vcc
	s_cbranch_execz .LBB44_34
; %bb.31:                               ; %.preheader.10.i.i
	ds_read_b32 v2, v3 offset:320
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[4:5], s[2:3], v2, s27, v[0:1]
	v_add_u32_e32 v5, 0x58, v1
	v_cmp_ge_u32_e32 vcc, s33, v5
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_mov_b32_e32 v2, s9
	v_add_co_u32_e64 v4, s[2:3], s8, v4
	v_addc_co_u32_e64 v5, s[2:3], v2, v5, s[2:3]
	s_mov_b64 s[2:3], 0
	global_store_dword v[4:5], v28, off
	global_store_dword v[4:5], v27, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[74:75], vcc
	s_cbranch_execz .LBB44_33
; %bb.32:                               ; %.preheader.11.i.i
	ds_read_b32 v2, v3 offset:352
	s_mov_b64 s[2:3], exec
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v2, v2, s27
.LBB44_33:                              ; %Flow3264
	s_or_b64 exec, exec, s[74:75]
	s_and_b64 s[2:3], s[2:3], exec
.LBB44_34:                              ; %Flow3263
	s_or_b64 exec, exec, s[72:73]
	s_and_b64 s[2:3], s[2:3], exec
.LBB44_35:                              ; %Flow3262
	s_or_b64 exec, exec, s[70:71]
	s_and_b64 s[2:3], s[2:3], exec
.LBB44_36:                              ; %Flow3261
	s_or_b64 exec, exec, s[68:69]
	s_and_b64 s[2:3], s[2:3], exec
.LBB44_37:                              ; %Flow3260
	s_or_b64 exec, exec, s[66:67]
	s_and_b64 s[2:3], s[2:3], exec
.LBB44_38:                              ; %Flow3259
	s_or_b64 exec, exec, s[64:65]
	s_and_b64 s[2:3], s[2:3], exec
.LBB44_39:                              ; %Flow3258
	s_or_b64 exec, exec, s[62:63]
	s_and_b64 s[2:3], s[2:3], exec
.LBB44_40:                              ; %Flow3257
	s_or_b64 exec, exec, s[60:61]
	s_and_b64 s[2:3], s[2:3], exec
.LBB44_41:                              ; %Flow3256
	s_or_b64 exec, exec, s[58:59]
	s_and_b64 s[2:3], s[2:3], exec
.LBB44_42:                              ; %Flow3255
	s_or_b64 exec, exec, s[56:57]
	s_and_b64 s[2:3], s[2:3], exec
.LBB44_43:                              ; %Flow3254
	s_or_b64 exec, exec, s[54:55]
	s_and_b64 s[2:3], s[2:3], exec
.LBB44_44:                              ; %Flow3253
	s_or_b64 exec, exec, s[52:53]
	s_branch .LBB44_126
.LBB44_45:                              ; %.split
	s_add_i32 s7, s23, 0x7f
	s_ashr_i32 s8, s7, 31
	s_lshr_b32 s8, s8, 25
	s_add_i32 s7, s7, s8
	s_ashr_i32 s8, s7, 7
	s_load_dword s7, s[4:5], 0xa8
	s_mul_i32 s4, s22, s8
	s_mul_i32 s4, s4, s38
	s_mul_i32 s23, s4, s46
	s_mul_i32 s23, s23, s34
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_u32_e32 v2, s7
	s_mul_hi_u32 s4, s23, s6
	s_cmp_lg_u32 s4, 0
	s_mul_i32 s33, s23, s6
	s_cbranch_scc0 .LBB44_122
; %bb.46:
	v_madmk_f32 v3, 0, 0x4f800000, v2
	v_rcp_f32_e32 v3, v3
                                        ; implicit-def: $vgpr4
	s_sub_u32 s5, 0, s7
	s_subb_u32 s35, 0, 0
	v_mul_f32_e32 v3, 0x5f7ffffc, v3
	v_mul_f32_e32 v4, 0x2f800000, v3
	v_trunc_f32_e32 v4, v4
	v_madmk_f32 v3, v4, 0xcf800000, v3
	v_cvt_u32_f32_e32 v4, v4
	v_cvt_u32_f32_e32 v3, v3
	v_readfirstlane_b32 s52, v4
	v_readfirstlane_b32 s53, v3
	s_mul_hi_u32 s55, s5, s53
	s_mul_i32 s56, s5, s52
	s_mul_i32 s54, s35, s53
	s_add_i32 s55, s55, s56
	s_add_i32 s55, s55, s54
	s_mul_i32 s57, s5, s53
	s_mul_i32 s56, s53, s55
	s_mul_hi_u32 s58, s53, s57
	s_mul_hi_u32 s54, s53, s55
	s_add_u32 s56, s58, s56
	s_addc_u32 s54, 0, s54
	s_mul_hi_u32 s59, s52, s57
	s_mul_i32 s57, s52, s57
	s_add_u32 s56, s56, s57
	s_mul_hi_u32 s58, s52, s55
	s_addc_u32 s54, s54, s59
	s_addc_u32 s56, s58, 0
	s_mul_i32 s55, s52, s55
	s_add_u32 s54, s54, s55
	s_addc_u32 s55, 0, s56
	s_add_u32 s53, s53, s54
	s_addc_u32 s52, s52, s55
	s_mul_i32 s54, s5, s52
	s_mul_hi_u32 s55, s5, s53
	s_add_i32 s54, s55, s54
	s_mul_i32 s35, s35, s53
	s_add_i32 s54, s54, s35
	s_mul_i32 s5, s5, s53
	s_mul_hi_u32 s55, s52, s5
	s_mul_i32 s56, s52, s5
	s_mul_i32 s58, s53, s54
	s_mul_hi_u32 s5, s53, s5
	s_mul_hi_u32 s57, s53, s54
	s_add_u32 s5, s5, s58
	s_addc_u32 s57, 0, s57
	s_add_u32 s5, s5, s56
	s_mul_hi_u32 s35, s52, s54
	s_addc_u32 s5, s57, s55
	s_addc_u32 s35, s35, 0
	s_mul_i32 s54, s52, s54
	s_add_u32 s5, s5, s54
	s_addc_u32 s35, 0, s35
	s_add_u32 s5, s53, s5
	s_addc_u32 s35, s52, s35
	s_mul_i32 s53, s33, s35
	s_mul_hi_u32 s54, s33, s5
	s_mul_hi_u32 s52, s33, s35
	s_add_u32 s53, s54, s53
	s_addc_u32 s52, 0, s52
	s_mul_hi_u32 s55, s4, s5
	s_mul_i32 s5, s4, s5
	s_add_u32 s5, s53, s5
	s_mul_hi_u32 s54, s4, s35
	s_addc_u32 s5, s52, s55
	s_addc_u32 s52, s54, 0
	s_mul_i32 s35, s4, s35
	s_add_u32 s5, s5, s35
	s_addc_u32 s35, 0, s52
	s_mul_i32 s35, s7, s35
	s_mul_hi_u32 s54, s7, s5
	s_add_u32 s52, s5, 1
	s_add_u32 s53, s5, 2
	s_add_i32 s54, s54, s35
	s_mul_i32 s35, s7, s5
	s_sub_u32 s35, s33, s35
	s_subb_u32 s4, s4, s54
	s_sub_u32 s54, s35, s7
	s_subb_u32 s55, s4, 0
	s_cmp_ge_u32 s54, s7
	s_cselect_b32 s54, -1, 0
	s_cmp_eq_u32 s55, 0
	s_cselect_b32 s54, s54, -1
	s_cmp_lg_u32 s54, 0
	s_cselect_b32 s52, s53, s52
	s_cmp_ge_u32 s35, s7
	s_cselect_b32 s35, -1, 0
	s_cmp_eq_u32 s4, 0
	s_cselect_b32 s4, s35, -1
	s_cmp_lg_u32 s4, 0
	s_cselect_b32 s4, s52, s5
	v_cvt_f32_u32_e32 v3, s7
	s_cbranch_execnz .LBB44_48
.LBB44_47:
	v_rcp_iflag_f32_e32 v4, v3
	s_sub_i32 s4, 0, s7
	v_mul_f32_e32 v4, 0x4f7ffffe, v4
	v_cvt_u32_f32_e32 v4, v4
	v_readfirstlane_b32 s5, v4
	s_mul_i32 s4, s4, s5
	s_mul_hi_u32 s4, s5, s4
	s_add_i32 s5, s5, s4
	s_mul_hi_u32 s4, s33, s5
	s_mul_i32 s8, s4, s7
	s_sub_i32 s8, s33, s8
	s_add_i32 s5, s4, 1
	s_sub_i32 s9, s8, s7
	s_cmp_ge_u32 s8, s7
	s_cselect_b32 s4, s5, s4
	s_cselect_b32 s8, s9, s8
	s_add_i32 s5, s4, 1
	s_cmp_ge_u32 s8, s7
	s_cselect_b32 s4, s5, s4
.LBB44_48:
	s_add_i32 s5, s6, 1
	s_mul_hi_u32 s33, s23, s5
	s_cmp_lg_u32 s33, 0
	s_mul_i32 s5, s23, s5
	s_cbranch_scc0 .LBB44_123
; %bb.49:
	v_mov_b32_e32 v4, 0x4f800000
	v_mac_f32_e32 v2, 0, v4
	v_rcp_f32_e32 v2, v2
	s_sub_u32 s23, 0, s7
	s_subb_u32 s35, 0, 0
	v_mul_f32_e32 v2, 0x5f7ffffc, v2
	v_mul_f32_e32 v4, 0x2f800000, v2
	v_trunc_f32_e32 v4, v4
	v_madmk_f32 v2, v4, 0xcf800000, v2
	v_cvt_u32_f32_e32 v4, v4
	v_cvt_u32_f32_e32 v2, v2
	v_readfirstlane_b32 s52, v4
	v_readfirstlane_b32 s53, v2
	s_mul_hi_u32 s55, s23, s53
	s_mul_i32 s56, s23, s52
	s_mul_i32 s54, s35, s53
	s_add_i32 s55, s55, s56
	s_add_i32 s55, s55, s54
	s_mul_i32 s57, s23, s53
	s_mul_i32 s56, s53, s55
	s_mul_hi_u32 s58, s53, s57
	s_mul_hi_u32 s54, s53, s55
	s_add_u32 s56, s58, s56
	s_addc_u32 s54, 0, s54
	s_mul_hi_u32 s59, s52, s57
	s_mul_i32 s57, s52, s57
	s_add_u32 s56, s56, s57
	s_mul_hi_u32 s58, s52, s55
	s_addc_u32 s54, s54, s59
	s_addc_u32 s56, s58, 0
	s_mul_i32 s55, s52, s55
	s_add_u32 s54, s54, s55
	s_addc_u32 s55, 0, s56
	s_add_u32 s53, s53, s54
	s_addc_u32 s52, s52, s55
	s_mul_i32 s54, s23, s52
	s_mul_hi_u32 s55, s23, s53
	s_add_i32 s54, s55, s54
	s_mul_i32 s35, s35, s53
	s_add_i32 s54, s54, s35
	s_mul_i32 s23, s23, s53
	s_mul_hi_u32 s55, s52, s23
	s_mul_i32 s56, s52, s23
	s_mul_i32 s58, s53, s54
	s_mul_hi_u32 s23, s53, s23
	s_mul_hi_u32 s57, s53, s54
	s_add_u32 s23, s23, s58
	s_addc_u32 s57, 0, s57
	s_add_u32 s23, s23, s56
	s_mul_hi_u32 s35, s52, s54
	s_addc_u32 s23, s57, s55
	s_addc_u32 s35, s35, 0
	s_mul_i32 s54, s52, s54
	s_add_u32 s23, s23, s54
	s_addc_u32 s35, 0, s35
	s_add_u32 s23, s53, s23
	s_addc_u32 s35, s52, s35
	s_mul_i32 s53, s5, s35
	s_mul_hi_u32 s54, s5, s23
	s_mul_hi_u32 s52, s5, s35
	s_add_u32 s53, s54, s53
	s_addc_u32 s52, 0, s52
	s_mul_hi_u32 s55, s33, s23
	s_mul_i32 s23, s33, s23
	s_add_u32 s23, s53, s23
	s_mul_hi_u32 s54, s33, s35
	s_addc_u32 s23, s52, s55
	s_addc_u32 s52, s54, 0
	s_mul_i32 s35, s33, s35
	s_add_u32 s23, s23, s35
	s_addc_u32 s35, 0, s52
	s_mul_i32 s35, s7, s35
	s_mul_hi_u32 s54, s7, s23
	s_add_u32 s52, s23, 1
	s_add_u32 s53, s23, 2
	s_add_i32 s54, s54, s35
	s_mul_i32 s35, s7, s23
	s_sub_u32 s35, s5, s35
	s_subb_u32 s33, s33, s54
	s_sub_u32 s54, s35, s7
	s_subb_u32 s55, s33, 0
	s_cmp_ge_u32 s54, s7
	s_cselect_b32 s54, -1, 0
	s_cmp_eq_u32 s55, 0
	s_cselect_b32 s54, s54, -1
	s_cmp_lg_u32 s54, 0
	s_cselect_b32 s52, s53, s52
	s_cmp_ge_u32 s35, s7
	s_cselect_b32 s35, -1, 0
	s_cmp_eq_u32 s33, 0
	s_cselect_b32 s33, s35, -1
	s_cmp_lg_u32 s33, 0
	s_cselect_b32 s52, s52, s23
	s_cbranch_execnz .LBB44_51
.LBB44_50:
	v_rcp_iflag_f32_e32 v2, v3
	s_sub_i32 s8, 0, s7
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	v_readfirstlane_b32 s9, v2
	s_mul_i32 s8, s8, s9
	s_mul_hi_u32 s8, s9, s8
	s_add_i32 s9, s9, s8
	s_mul_hi_u32 s8, s5, s9
	s_mul_i32 s23, s8, s7
	s_sub_i32 s5, s5, s23
	s_add_i32 s9, s8, 1
	s_sub_i32 s23, s5, s7
	s_cmp_ge_u32 s5, s7
	s_cselect_b32 s8, s9, s8
	s_cselect_b32 s5, s23, s5
	s_add_i32 s9, s8, 1
	s_cmp_ge_u32 s5, s7
	s_cselect_b32 s52, s9, s8
.LBB44_51:
	s_mul_hi_u32 s5, s4, s20
	s_add_i32 s5, s5, s4
	s_lshr_b32 s5, s5, s21
	s_mul_i32 s5, s5, s22
	s_sub_i32 s5, s4, s5
	s_and_b32 s5, s5, 7
	s_sub_i32 s33, s4, s5
	s_mul_hi_u32 s4, s52, s20
	s_add_i32 s4, s4, s52
	s_lshr_b32 s4, s4, s21
	s_mul_i32 s4, s4, s22
	s_sub_i32 s4, s52, s4
	s_and_b32 s4, s4, 7
	s_sub_i32 s35, s52, s4
	s_mul_hi_u32 s4, s33, s20
	s_add_i32 s4, s33, s4
	s_lshr_b32 s4, s4, s21
	s_mul_i32 s4, s4, s22
	s_sub_i32 s7, s33, s4
	s_sub_i32 s4, s35, s4
	s_min_u32 s23, s22, s4
	s_cmp_lt_i32 s33, s35
	s_cselect_b64 s[54:55], -1, 0
	s_cmp_le_u32 s22, s4
	s_cselect_b64 s[4:5], -1, 0
	s_and_b64 s[4:5], s[54:55], s[4:5]
	s_andn2_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB44_83
; %bb.52:                               ; %.lr.ph1109
	v_lshlrev_b32_e32 v3, 2, v1
	v_lshrrev_b32_e32 v4, 4, v0
	v_add_u32_e32 v4, v3, v4
	v_mul_lo_u32 v22, s25, v4
	s_movk_i32 s54, 0x104
	v_and_b32_e32 v2, 15, v0
	v_mad_u32_u24 v4, v4, s54, 0
	s_cmp_lg_u64 s[16:17], 0
	v_lshl_add_u32 v23, v2, 2, v4
	v_lshrrev_b32_e32 v2, 3, v0
	s_cselect_b64 s[4:5], -1, 0
	s_lshl_b32 s8, s25, 5
	v_lshl_add_u32 v2, v1, 3, v2
	v_and_b32_e32 v20, 7, v0
	v_add_u32_e32 v25, s8, v22
	v_mul_lo_u32 v28, s25, v2
	v_and_b32_e32 v4, 0x1ffc, v2
	v_lshlrev_b32_e32 v7, 5, v2
	v_add_u32_e32 v2, 64, v2
	v_mul_u32_u24_e32 v9, 36, v1
	v_add_u32_e32 v26, s8, v25
	v_lshlrev_b32_e32 v6, 2, v20
	v_and_b32_e32 v8, 0x3ffc, v2
	v_add_u32_e32 v10, 64, v0
	v_add_u32_e32 v30, 0, v3
	v_lshl_add_u32 v3, v9, 2, 0
	v_add_u32_e32 v27, s8, v26
	v_add3_u32 v4, 0, v4, v6
	v_add3_u32 v6, 0, v8, v6
	s_mul_i32 s8, s26, 36
	v_lshlrev_b32_e32 v8, 5, v0
	v_add_u32_e32 v42, 0x190, v3
	v_add_u32_e32 v44, 0x180, v3
	v_and_b32_e32 v3, 0x3fc, v10
	v_and_b32_e32 v10, 0x1fc, v0
	v_lshlrev_b32_e32 v5, 1, v20
	v_lshlrev_b32_e32 v2, 5, v2
	s_ashr_i32 s9, s8, 31
	v_mad_u32_u24 v9, v0, s54, 0
	v_add3_u32 v3, v8, v3, 0
	v_add3_u32 v8, v8, v10, 0
	v_lshl_add_u32 v18, v24, 2, 0
	v_bfe_u32 v19, v0, 3, 1
	v_mov_b32_e32 v21, 0
	v_lshl_add_u32 v29, s25, 6, v28
	v_add_u32_e32 v31, 8, v1
	v_add_u32_e32 v32, 16, v1
	v_add_u32_e32 v33, 24, v1
	v_add_u32_e32 v34, 32, v1
	v_add_u32_e32 v35, 40, v1
	v_add_u32_e32 v36, 48, v1
	v_add_u32_e32 v37, 56, v1
	v_add_u32_e32 v38, 64, v1
	v_add_u32_e32 v39, 0x48, v1
	v_add_u32_e32 v40, 0x50, v1
	v_add_u32_e32 v41, 0x58, v1
	s_lshl_b64 s[52:53], s[8:9], 2
	v_add_u32_e32 v43, 0x3980, v9
	v_add_u32_e32 v45, 0xc380, v3
	v_add_u32_e32 v46, 0xbb80, v8
	v_add_u32_e32 v47, 0x3a00, v9
	v_add_u32_e32 v48, 0xc390, v3
	v_add_u32_e32 v49, 0xbb90, v8
	v_lshlrev_b32_e32 v50, 1, v5
	v_add_u32_e32 v51, v4, v7
	v_add_u32_e32 v52, v6, v2
	s_movk_i32 s9, 0x1000
	s_branch .LBB44_55
.LBB44_53:                              ; %Flow3237
                                        ;   in Loop: Header=BB44_55 Depth=1
	s_or_b64 exec, exec, s[54:55]
.LBB44_54:                              ; %_ZL22mul_mat_q_process_tileIL9ggml_type8ELi96ELb0ELb0EEvPKciPKiS4_PfS5_iiiiiii.exit736
                                        ;   in Loop: Header=BB44_55 Depth=1
	s_add_i32 s7, s33, s22
	s_mul_hi_u32 s23, s7, s20
	s_add_i32 s7, s7, s23
	s_lshr_b32 s7, s7, s21
	s_mul_i32 s33, s7, s22
	s_sub_i32 s56, s35, s33
	s_cmp_gt_i32 s35, s33
	s_cselect_b64 s[54:55], -1, 0
	s_cmp_le_u32 s22, s56
	s_cselect_b64 s[58:59], -1, 0
	s_and_b64 s[58:59], s[54:55], s[58:59]
	s_mov_b32 s7, 0
	s_and_b64 vcc, exec, s[58:59]
	s_mov_b32 s23, s22
	s_cbranch_vccz .LBB44_82
.LBB44_55:                              ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB44_64 Depth 2
                                        ;       Child Loop BB44_65 Depth 3
                                        ;       Child Loop BB44_67 Depth 3
	s_mul_hi_u32 s54, s33, s20
	s_add_i32 s54, s33, s54
	s_lshr_b32 s54, s54, s21
	s_mul_hi_u32 s55, s54, s50
	s_add_i32 s55, s54, s55
	s_lshr_b32 s55, s55, s51
	s_mul_i32 s56, s55, s34
	s_sub_i32 s58, s54, s56
	s_mul_hi_u32 s54, s55, s36
	s_add_i32 s54, s55, s54
	s_lshr_b32 s56, s54, s37
	s_mul_i32 s54, s56, s38
	s_sub_i32 s54, s55, s54
	s_mul_hi_u32 s55, s56, s44
	s_add_i32 s55, s56, s55
	s_lshr_b32 s60, s55, s45
	s_mul_i32 s55, s60, s46
	s_sub_i32 s61, s56, s55
	s_mulk_i32 s58, 0x60
	s_andn2_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB44_61
; %bb.56:                               ;   in Loop: Header=BB44_55 Depth=1
	s_ashr_i32 s55, s54, 31
	s_lshl_b64 s[56:57], s[54:55], 2
	s_add_u32 s56, s18, s56
	s_addc_u32 s57, s19, s57
	global_load_dwordx2 v[2:3], v21, s[56:57]
	s_mov_b64 s[56:57], 0
	s_mov_b32 s55, 0
	s_mov_b32 s62, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s59, v2
	v_subrev_u32_e32 v69, s59, v3
	v_cmp_lt_i32_e32 vcc, s58, v69
	s_cbranch_vccz .LBB44_60
; %bb.57:                               ;   in Loop: Header=BB44_55 Depth=1
	s_barrier
	s_and_saveexec_b64 s[56:57], s[0:1]
	s_cbranch_execz .LBB44_59
; %bb.58:                               ; %.lr.ph1105
                                        ;   in Loop: Header=BB44_55 Depth=1
	s_add_i32 s62, s59, s58
	v_add_u32_e32 v2, s62, v24
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_mov_b32_e32 v4, s17
	v_add_co_u32_e32 v2, vcc, s16, v2
	v_addc_co_u32_e32 v3, vcc, v4, v3, vcc
	global_load_dword v2, v[2:3], off
	s_waitcnt vmcnt(0)
	ds_write_b32 v18, v2
.LBB44_59:                              ; %.critedge465
                                        ;   in Loop: Header=BB44_55 Depth=1
	s_or_b64 exec, exec, s[56:57]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_mov_b64 s[56:57], -1
	s_mov_b32 s62, s59
.LBB44_60:                              ; %Flow3241
                                        ;   in Loop: Header=BB44_55 Depth=1
	s_mov_b32 s59, 0
	s_and_b64 vcc, exec, s[56:57]
	s_cbranch_vccz .LBB44_54
	s_branch .LBB44_62
.LBB44_61:                              ;   in Loop: Header=BB44_55 Depth=1
	s_mul_i32 s55, s61, s48
	s_mul_i32 s56, s54, s40
	s_mul_i32 s57, s54, s41
	s_mul_i32 s59, s58, s27
	s_add_i32 s55, s55, s56
	s_mul_i32 s56, s61, s49
	s_add_i32 s57, s57, s59
	v_mov_b32_e32 v69, s24
	s_add_i32 s59, s57, s56
	s_mov_b32 s62, 0
	s_cbranch_execz .LBB44_54
.LBB44_62:                              ;   in Loop: Header=BB44_55 Depth=1
	s_lshl_b32 s56, s60, 7
	v_mov_b32_e32 v53, 0
	s_cmp_ge_i32 s7, s23
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	s_cbranch_scc1 .LBB44_69
; %bb.63:                               ; %.lr.ph.i552
                                        ;   in Loop: Header=BB44_55 Depth=1
	s_add_i32 s57, s62, s58
	s_mul_i32 s57, s57, 36
	s_add_i32 s62, s57, s55
	s_ashr_i32 s63, s62, 31
	s_lshl_b64 s[62:63], s[62:63], 2
	s_add_u32 s57, s14, s62
	s_mul_hi_u32 s62, s54, s10
	s_addc_u32 s60, s15, s63
	s_add_i32 s54, s54, s62
	s_mul_hi_u32 s62, s61, s42
	s_lshr_b32 s54, s54, s11
	s_add_i32 s61, s61, s62
	s_mul_i32 s55, s56, s25
	s_mul_i32 s54, s54, s39
	s_lshr_b32 s61, s61, s43
	s_mul_i32 s61, s61, s47
	s_add_i32 s54, s54, s55
	s_add_i32 s61, s54, s61
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
.LBB44_64:                              ;   Parent Loop BB44_55 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB44_65 Depth 3
                                        ;       Child Loop BB44_67 Depth 3
	s_add_i32 s54, s61, s7
	s_mul_hi_i32 s55, s54, 34
	s_mul_i32 s54, s54, 34
	s_add_u32 s54, s12, s54
	s_addc_u32 s55, s13, s55
	v_mad_u64_u32 v[2:3], s[62:63], v19, 34, s[54:55]
	v_lshlrev_b32_e32 v78, 2, v24
	v_mad_i64_i32 v[4:5], s[62:63], v22, 34, v[2:3]
	v_mad_i64_i32 v[6:7], s[62:63], v25, 34, v[2:3]
	v_add_co_u32_e32 v4, vcc, v4, v50
	v_addc_co_u32_e32 v5, vcc, 0, v5, vcc
	v_add_co_u32_e32 v6, vcc, v6, v50
	v_addc_co_u32_e32 v7, vcc, 0, v7, vcc
	v_mad_i64_i32 v[8:9], s[62:63], v26, 34, v[2:3]
	global_load_dword v10, v[4:5], off offset:2
	global_load_dword v11, v[4:5], off offset:70
	global_load_dword v12, v[4:5], off offset:138
	global_load_dword v13, v[4:5], off offset:206
	global_load_dword v14, v[6:7], off offset:2
	global_load_dword v15, v[6:7], off offset:70
	global_load_dword v16, v[6:7], off offset:138
	global_load_dword v17, v[6:7], off offset:206
	v_mad_u64_u32 v[6:7], s[54:55], v20, 34, s[54:55]
	v_add_co_u32_e32 v4, vcc, v8, v50
	v_addc_co_u32_e32 v5, vcc, 0, v9, vcc
	v_mad_i64_i32 v[8:9], s[54:55], v28, 34, v[6:7]
	v_mad_i64_i32 v[6:7], s[54:55], v29, 34, v[6:7]
	s_ashr_i32 s54, s7, 31
	s_lshr_b32 s54, s54, 30
	s_add_i32 s54, s7, s54
	v_mad_i64_i32 v[2:3], s[62:63], v27, 34, v[2:3]
	s_ashr_i32 s54, s54, 2
	s_mul_i32 s54, s8, s54
	s_ashr_i32 s55, s54, 31
	s_lshl_b64 s[54:55], s[54:55], 2
	v_add_co_u32_e32 v2, vcc, v2, v50
	s_add_u32 s54, s57, s54
	v_addc_co_u32_e32 v3, vcc, 0, v3, vcc
	s_addc_u32 s55, s60, s55
	global_load_dword v79, v[4:5], off offset:2
	global_load_dword v80, v[4:5], off offset:70
	global_load_dword v81, v[4:5], off offset:138
	global_load_dword v82, v[4:5], off offset:206
	global_load_dword v83, v[2:3], off offset:2
	global_load_dword v84, v[2:3], off offset:70
	global_load_dword v85, v[2:3], off offset:138
	global_load_dword v86, v[2:3], off offset:206
	s_nop 0
	global_load_ushort v8, v[8:9], off
	s_nop 0
	global_load_ushort v9, v[6:7], off
	v_mov_b32_e32 v2, s55
	v_add_co_u32_e32 v6, vcc, s54, v78
	v_addc_co_u32_e32 v7, vcc, 0, v2, vcc
	v_add_co_u32_e32 v2, vcc, s9, v6
	v_addc_co_u32_e32 v3, vcc, 0, v7, vcc
	v_add_co_u32_e32 v4, vcc, 0x2000, v6
	v_addc_co_u32_e32 v5, vcc, 0, v7, vcc
	v_add_co_u32_e32 v6, vcc, 0x3000, v6
	v_addc_co_u32_e32 v7, vcc, 0, v7, vcc
	global_load_dword v87, v78, s[54:55]
	global_load_dword v88, v78, s[54:55] offset:2048
	global_load_dword v89, v[2:3], off
	s_nop 0
	global_load_dword v2, v[2:3], off offset:2048
	s_nop 0
	global_load_dword v3, v[4:5], off
	s_nop 0
	global_load_dword v4, v[4:5], off offset:2048
	s_nop 0
	global_load_dword v5, v[6:7], off
	v_add_u32_e32 v6, 0x3800, v23
	s_mov_b32 s62, -8
	s_waitcnt vmcnt(23)
	ds_write2_b32 v6, v10, v11 offset0:96 offset1:112
	s_waitcnt vmcnt(21)
	ds_write2_b32 v6, v12, v13 offset0:128 offset1:144
	v_add_u32_e32 v6, 0x5800, v23
	s_waitcnt vmcnt(19)
	ds_write2_b32 v6, v14, v15 offset0:128 offset1:144
	s_waitcnt vmcnt(17)
	ds_write2_b32 v6, v16, v17 offset0:160 offset1:176
	v_add_u32_e32 v6, 0x7800, v23
	s_waitcnt vmcnt(15)
	ds_write2_b32 v6, v79, v80 offset0:160 offset1:176
	s_waitcnt vmcnt(13)
	ds_write2_b32 v6, v81, v82 offset0:192 offset1:208
	v_add_u32_e32 v6, 0x9800, v23
	v_add_u32_e32 v79, 0x80, v18
	v_mov_b32_e32 v80, v46
	v_mov_b32_e32 v81, v45
	s_waitcnt vmcnt(8)
	v_cvt_f32_f16_e32 v7, v8
	s_waitcnt vmcnt(7)
	v_cvt_f32_f16_e32 v8, v9
	ds_write2_b32 v6, v83, v84 offset0:192 offset1:208
	ds_write2_b32 v6, v85, v86 offset0:224 offset1:240
	ds_write_b32 v51, v7 offset:48000
	ds_write_b32 v52, v8 offset:48000
	v_mov_b32_e32 v82, v44
	v_mov_b32_e32 v83, v43
	v_mov_b32_e32 v84, v42
	s_waitcnt vmcnt(5)
	ds_write2st64_b32 v79, v87, v88 offset0:1 offset1:9
	s_waitcnt vmcnt(3)
	ds_write2st64_b32 v79, v89, v2 offset0:17 offset1:25
	s_waitcnt vmcnt(1)
	ds_write2st64_b32 v79, v3, v4 offset0:33 offset1:41
	s_waitcnt vmcnt(0)
	ds_write_b32 v18, v5 offset:12672
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB44_65:                              ;   Parent Loop BB44_55 Depth=1
                                        ;     Parent Loop BB44_64 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ds_read_b32 v91, v82
	ds_read2_b32 v[2:3], v84 offset1:1
	ds_read2_b32 v[4:5], v84 offset0:2 offset1:3
	ds_read2_b32 v[87:88], v84 offset0:4 offset1:5
	ds_read2_b32 v[89:90], v84 offset0:6 offset1:7
	ds_read_b32 v86, v80
	ds_read2_b32 v[8:9], v83 offset1:1
	ds_read2_b32 v[10:11], v83 offset0:2 offset1:3
	ds_read2_b32 v[14:15], v83 offset0:4 offset1:5
	ds_read2_b32 v[16:17], v83 offset0:6 offset1:7
	s_waitcnt lgkmcnt(4)
	v_mul_f32_e32 v7, v86, v91
	s_waitcnt lgkmcnt(3)
	v_dot4_i32_i8 v6, v8, v2, 0
	v_dot4_i32_i8 v6, v9, v3, v6
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v6, v10, v4, v6
	v_dot4_i32_i8 v6, v11, v5, v6
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v6, v14, v87, v6
	v_dot4_i32_i8 v6, v15, v88, v6
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v6, v16, v89, v6
	v_dot4_i32_i8 v6, v17, v90, v6
	v_cvt_f32_i32_e32 v6, v6
	v_add_u32_e32 v93, 0x498, v84
	s_add_i32 s62, s62, 8
	v_add_u32_e32 v80, 4, v80
	v_fmac_f32_e32 v77, v7, v6
	v_add_u32_e32 v6, 0x4100, v83
	ds_read_b32 v85, v81
	ds_read2_b32 v[12:13], v6 offset1:1
	v_add_u32_e32 v81, 4, v81
	s_cmp_lt_u32 s62, 24
	ds_read2_b32 v[93:94], v93 offset1:1
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v2, v12, v2, 0
	v_dot4_i32_i8 v2, v13, v3, v2
	v_add_u32_e32 v3, 0x4108, v83
	ds_read2_b32 v[6:7], v3 offset1:1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v2, v6, v4, v2
	v_dot4_i32_i8 v4, v7, v5, v2
	v_add_u32_e32 v2, 0x4110, v83
	ds_read2_b32 v[2:3], v2 offset1:1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v4, v2, v87, v4
	v_dot4_i32_i8 v87, v3, v88, v4
	v_add_u32_e32 v4, 0x4118, v83
	ds_read2_b32 v[4:5], v4 offset1:1
	v_mul_f32_e32 v88, v85, v91
	v_add_u32_e32 v91, 0x490, v84
	ds_read2_b32 v[91:92], v91 offset1:1
	v_add_u32_e32 v83, 32, v83
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v87, v4, v89, v87
	v_dot4_i32_i8 v87, v5, v90, v87
	v_cvt_f32_i32_e32 v87, v87
	v_add_u32_e32 v89, 0x488, v84
	ds_read2_b32 v[89:90], v89 offset1:1
	v_fmac_f32_e32 v76, v88, v87
	v_add_u32_e32 v87, 0x480, v84
	ds_read_b32 v95, v82 offset:1152
	ds_read2_b32 v[87:88], v87 offset1:1
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v97, v95, v86
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v96, v8, v87, 0
	v_dot4_i32_i8 v87, v12, v87, 0
	v_dot4_i32_i8 v87, v13, v88, v87
	v_dot4_i32_i8 v87, v6, v89, v87
	v_dot4_i32_i8 v87, v7, v90, v87
	v_dot4_i32_i8 v87, v2, v91, v87
	v_dot4_i32_i8 v87, v3, v92, v87
	v_dot4_i32_i8 v87, v4, v93, v87
	v_dot4_i32_i8 v96, v9, v88, v96
	v_dot4_i32_i8 v87, v5, v94, v87
	v_dot4_i32_i8 v96, v10, v89, v96
	v_cvt_f32_i32_e32 v87, v87
	v_dot4_i32_i8 v96, v11, v90, v96
	v_dot4_i32_i8 v96, v14, v91, v96
	v_dot4_i32_i8 v96, v15, v92, v96
	v_mul_f32_e32 v88, v95, v85
	v_dot4_i32_i8 v96, v16, v93, v96
	v_fmac_f32_e32 v74, v88, v87
	v_add_u32_e32 v87, 0x900, v84
	v_dot4_i32_i8 v96, v17, v94, v96
	ds_read_b32 v95, v82 offset:2304
	ds_read2_b32 v[87:88], v87 offset1:1
	v_cvt_f32_i32_e32 v96, v96
	v_add_u32_e32 v89, 0x908, v84
	ds_read2_b32 v[89:90], v89 offset1:1
	v_add_u32_e32 v91, 0x910, v84
	ds_read2_b32 v[91:92], v91 offset1:1
	v_fmac_f32_e32 v75, v97, v96
	v_add_u32_e32 v93, 0x918, v84
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v96, v8, v87, 0
	v_dot4_i32_i8 v87, v12, v87, 0
	ds_read2_b32 v[93:94], v93 offset1:1
	v_dot4_i32_i8 v87, v13, v88, v87
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v87, v6, v89, v87
	v_dot4_i32_i8 v87, v7, v90, v87
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v87, v2, v91, v87
	v_dot4_i32_i8 v87, v3, v92, v87
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v87, v4, v93, v87
	v_dot4_i32_i8 v96, v9, v88, v96
	v_dot4_i32_i8 v87, v5, v94, v87
	v_dot4_i32_i8 v96, v10, v89, v96
	v_cvt_f32_i32_e32 v87, v87
	v_dot4_i32_i8 v96, v11, v90, v96
	v_dot4_i32_i8 v96, v14, v91, v96
	v_dot4_i32_i8 v96, v15, v92, v96
	v_mul_f32_e32 v88, v95, v85
	v_dot4_i32_i8 v96, v16, v93, v96
	v_fmac_f32_e32 v72, v88, v87
	v_add_u32_e32 v87, 0xd80, v84
	v_dot4_i32_i8 v96, v17, v94, v96
	v_mul_f32_e32 v97, v95, v86
	ds_read_b32 v95, v82 offset:3456
	ds_read2_b32 v[87:88], v87 offset1:1
	v_cvt_f32_i32_e32 v96, v96
	v_add_u32_e32 v89, 0xd88, v84
	ds_read2_b32 v[89:90], v89 offset1:1
	v_add_u32_e32 v91, 0xd90, v84
	ds_read2_b32 v[91:92], v91 offset1:1
	v_fmac_f32_e32 v73, v97, v96
	v_add_u32_e32 v93, 0xd98, v84
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v96, v8, v87, 0
	v_dot4_i32_i8 v87, v12, v87, 0
	ds_read2_b32 v[93:94], v93 offset1:1
	v_dot4_i32_i8 v87, v13, v88, v87
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v87, v6, v89, v87
	v_dot4_i32_i8 v87, v7, v90, v87
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v87, v2, v91, v87
	v_dot4_i32_i8 v87, v3, v92, v87
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v87, v4, v93, v87
	v_dot4_i32_i8 v96, v9, v88, v96
	v_dot4_i32_i8 v87, v5, v94, v87
	v_dot4_i32_i8 v96, v10, v89, v96
	v_cvt_f32_i32_e32 v87, v87
	v_dot4_i32_i8 v96, v11, v90, v96
	v_dot4_i32_i8 v96, v14, v91, v96
	v_dot4_i32_i8 v96, v15, v92, v96
	v_mul_f32_e32 v88, v95, v85
	v_dot4_i32_i8 v96, v16, v93, v96
	v_fmac_f32_e32 v70, v88, v87
	v_add_u32_e32 v87, 0x1200, v84
	v_dot4_i32_i8 v96, v17, v94, v96
	v_mul_f32_e32 v97, v95, v86
	ds_read_b32 v95, v82 offset:4608
	ds_read2_b32 v[87:88], v87 offset1:1
	v_cvt_f32_i32_e32 v96, v96
	v_add_u32_e32 v89, 0x1208, v84
	ds_read2_b32 v[89:90], v89 offset1:1
	v_add_u32_e32 v91, 0x1210, v84
	ds_read2_b32 v[91:92], v91 offset1:1
	v_fmac_f32_e32 v71, v97, v96
	v_add_u32_e32 v93, 0x1218, v84
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v96, v8, v87, 0
	v_dot4_i32_i8 v87, v12, v87, 0
	ds_read2_b32 v[93:94], v93 offset1:1
	v_dot4_i32_i8 v87, v13, v88, v87
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v87, v6, v89, v87
	v_dot4_i32_i8 v87, v7, v90, v87
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v87, v2, v91, v87
	v_dot4_i32_i8 v87, v3, v92, v87
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v87, v4, v93, v87
	v_dot4_i32_i8 v96, v9, v88, v96
	v_dot4_i32_i8 v87, v5, v94, v87
	v_dot4_i32_i8 v96, v10, v89, v96
	v_cvt_f32_i32_e32 v87, v87
	v_dot4_i32_i8 v96, v11, v90, v96
	v_dot4_i32_i8 v96, v14, v91, v96
	v_dot4_i32_i8 v96, v15, v92, v96
	v_mul_f32_e32 v88, v95, v85
	v_dot4_i32_i8 v96, v16, v93, v96
	v_fmac_f32_e32 v67, v88, v87
	v_add_u32_e32 v87, 0x1680, v84
	v_dot4_i32_i8 v96, v17, v94, v96
	v_mul_f32_e32 v97, v95, v86
	ds_read_b32 v95, v82 offset:5760
	ds_read2_b32 v[87:88], v87 offset1:1
	v_cvt_f32_i32_e32 v96, v96
	v_add_u32_e32 v89, 0x1688, v84
	ds_read2_b32 v[89:90], v89 offset1:1
	v_add_u32_e32 v91, 0x1690, v84
	ds_read2_b32 v[91:92], v91 offset1:1
	v_fmac_f32_e32 v68, v97, v96
	v_add_u32_e32 v93, 0x1698, v84
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v96, v8, v87, 0
	v_dot4_i32_i8 v87, v12, v87, 0
	ds_read2_b32 v[93:94], v93 offset1:1
	v_dot4_i32_i8 v87, v13, v88, v87
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v87, v6, v89, v87
	v_dot4_i32_i8 v87, v7, v90, v87
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v87, v2, v91, v87
	v_dot4_i32_i8 v87, v3, v92, v87
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v87, v4, v93, v87
	v_dot4_i32_i8 v96, v9, v88, v96
	v_dot4_i32_i8 v87, v5, v94, v87
	v_dot4_i32_i8 v96, v10, v89, v96
	v_cvt_f32_i32_e32 v87, v87
	v_dot4_i32_i8 v96, v11, v90, v96
	v_dot4_i32_i8 v96, v14, v91, v96
	v_dot4_i32_i8 v96, v15, v92, v96
	v_mul_f32_e32 v88, v95, v85
	v_dot4_i32_i8 v96, v16, v93, v96
	v_fmac_f32_e32 v65, v88, v87
	v_add_u32_e32 v87, 0x1b00, v84
	v_dot4_i32_i8 v96, v17, v94, v96
	v_mul_f32_e32 v97, v95, v86
	ds_read_b32 v95, v82 offset:6912
	ds_read2_b32 v[87:88], v87 offset1:1
	v_cvt_f32_i32_e32 v96, v96
	v_add_u32_e32 v89, 0x1b08, v84
	ds_read2_b32 v[89:90], v89 offset1:1
	v_add_u32_e32 v91, 0x1b10, v84
	ds_read2_b32 v[91:92], v91 offset1:1
	v_fmac_f32_e32 v66, v97, v96
	v_add_u32_e32 v93, 0x1b18, v84
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v96, v8, v87, 0
	v_dot4_i32_i8 v87, v12, v87, 0
	ds_read2_b32 v[93:94], v93 offset1:1
	v_dot4_i32_i8 v87, v13, v88, v87
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v87, v6, v89, v87
	v_dot4_i32_i8 v87, v7, v90, v87
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v87, v2, v91, v87
	v_dot4_i32_i8 v87, v3, v92, v87
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v87, v4, v93, v87
	v_dot4_i32_i8 v96, v9, v88, v96
	v_dot4_i32_i8 v87, v5, v94, v87
	v_dot4_i32_i8 v96, v10, v89, v96
	v_cvt_f32_i32_e32 v87, v87
	v_dot4_i32_i8 v96, v11, v90, v96
	v_dot4_i32_i8 v96, v14, v91, v96
	v_dot4_i32_i8 v96, v15, v92, v96
	v_mul_f32_e32 v88, v95, v85
	v_dot4_i32_i8 v96, v16, v93, v96
	v_fmac_f32_e32 v63, v88, v87
	v_add_u32_e32 v87, 0x1f80, v84
	v_dot4_i32_i8 v96, v17, v94, v96
	v_mul_f32_e32 v97, v95, v86
	ds_read_b32 v95, v82 offset:8064
	ds_read2_b32 v[87:88], v87 offset1:1
	v_cvt_f32_i32_e32 v96, v96
	v_add_u32_e32 v89, 0x1f88, v84
	ds_read2_b32 v[89:90], v89 offset1:1
	v_add_u32_e32 v91, 0x1f90, v84
	ds_read2_b32 v[91:92], v91 offset1:1
	v_fmac_f32_e32 v64, v97, v96
	v_add_u32_e32 v93, 0x1f98, v84
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v96, v8, v87, 0
	v_dot4_i32_i8 v87, v12, v87, 0
	ds_read2_b32 v[93:94], v93 offset1:1
	v_dot4_i32_i8 v87, v13, v88, v87
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v87, v6, v89, v87
	v_dot4_i32_i8 v87, v7, v90, v87
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v87, v2, v91, v87
	v_dot4_i32_i8 v87, v3, v92, v87
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v87, v4, v93, v87
	v_dot4_i32_i8 v96, v9, v88, v96
	v_dot4_i32_i8 v87, v5, v94, v87
	v_dot4_i32_i8 v96, v10, v89, v96
	v_cvt_f32_i32_e32 v87, v87
	v_dot4_i32_i8 v96, v11, v90, v96
	v_dot4_i32_i8 v96, v14, v91, v96
	v_dot4_i32_i8 v96, v15, v92, v96
	v_mul_f32_e32 v88, v95, v85
	v_dot4_i32_i8 v96, v16, v93, v96
	v_fmac_f32_e32 v61, v88, v87
	v_add_u32_e32 v87, 0x2400, v84
	v_dot4_i32_i8 v96, v17, v94, v96
	v_mul_f32_e32 v97, v95, v86
	ds_read_b32 v95, v82 offset:9216
	ds_read2_b32 v[87:88], v87 offset1:1
	v_cvt_f32_i32_e32 v96, v96
	v_add_u32_e32 v89, 0x2408, v84
	ds_read2_b32 v[89:90], v89 offset1:1
	v_add_u32_e32 v91, 0x2410, v84
	ds_read2_b32 v[91:92], v91 offset1:1
	v_fmac_f32_e32 v62, v97, v96
	v_add_u32_e32 v93, 0x2418, v84
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v96, v8, v87, 0
	v_dot4_i32_i8 v87, v12, v87, 0
	ds_read2_b32 v[93:94], v93 offset1:1
	v_dot4_i32_i8 v87, v13, v88, v87
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v87, v6, v89, v87
	v_dot4_i32_i8 v87, v7, v90, v87
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v87, v2, v91, v87
	v_dot4_i32_i8 v87, v3, v92, v87
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v87, v4, v93, v87
	v_dot4_i32_i8 v96, v9, v88, v96
	v_dot4_i32_i8 v87, v5, v94, v87
	v_dot4_i32_i8 v96, v10, v89, v96
	v_cvt_f32_i32_e32 v87, v87
	v_dot4_i32_i8 v96, v11, v90, v96
	v_dot4_i32_i8 v96, v14, v91, v96
	v_dot4_i32_i8 v96, v15, v92, v96
	v_mul_f32_e32 v88, v95, v85
	v_dot4_i32_i8 v96, v16, v93, v96
	v_fmac_f32_e32 v59, v88, v87
	v_add_u32_e32 v87, 0x2880, v84
	v_dot4_i32_i8 v96, v17, v94, v96
	v_mul_f32_e32 v97, v95, v86
	ds_read_b32 v95, v82 offset:10368
	ds_read2_b32 v[87:88], v87 offset1:1
	v_cvt_f32_i32_e32 v96, v96
	v_add_u32_e32 v89, 0x2888, v84
	ds_read2_b32 v[89:90], v89 offset1:1
	v_add_u32_e32 v91, 0x2890, v84
	ds_read2_b32 v[91:92], v91 offset1:1
	v_fmac_f32_e32 v60, v97, v96
	v_add_u32_e32 v93, 0x2898, v84
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v96, v8, v87, 0
	v_dot4_i32_i8 v87, v12, v87, 0
	ds_read2_b32 v[93:94], v93 offset1:1
	v_dot4_i32_i8 v87, v13, v88, v87
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v87, v6, v89, v87
	v_dot4_i32_i8 v87, v7, v90, v87
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v87, v2, v91, v87
	v_dot4_i32_i8 v87, v3, v92, v87
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v87, v4, v93, v87
	v_dot4_i32_i8 v96, v9, v88, v96
	v_dot4_i32_i8 v87, v5, v94, v87
	v_dot4_i32_i8 v96, v10, v89, v96
	v_cvt_f32_i32_e32 v87, v87
	v_dot4_i32_i8 v96, v11, v90, v96
	v_dot4_i32_i8 v96, v14, v91, v96
	v_dot4_i32_i8 v96, v15, v92, v96
	v_mul_f32_e32 v88, v95, v85
	v_dot4_i32_i8 v96, v16, v93, v96
	v_fmac_f32_e32 v57, v88, v87
	v_add_u32_e32 v87, 0x2d00, v84
	v_dot4_i32_i8 v96, v17, v94, v96
	v_mul_f32_e32 v97, v95, v86
	ds_read_b32 v95, v82 offset:11520
	ds_read2_b32 v[87:88], v87 offset1:1
	v_cvt_f32_i32_e32 v96, v96
	v_add_u32_e32 v89, 0x2d08, v84
	ds_read2_b32 v[89:90], v89 offset1:1
	v_add_u32_e32 v91, 0x2d10, v84
	ds_read2_b32 v[91:92], v91 offset1:1
	v_fmac_f32_e32 v58, v97, v96
	v_add_u32_e32 v93, 0x2d18, v84
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v96, v8, v87, 0
	v_dot4_i32_i8 v87, v12, v87, 0
	ds_read2_b32 v[93:94], v93 offset1:1
	v_dot4_i32_i8 v87, v13, v88, v87
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v87, v6, v89, v87
	v_dot4_i32_i8 v87, v7, v90, v87
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v87, v2, v91, v87
	v_dot4_i32_i8 v87, v3, v92, v87
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v87, v4, v93, v87
	v_dot4_i32_i8 v87, v5, v94, v87
	v_cvt_f32_i32_e32 v87, v87
	v_dot4_i32_i8 v96, v9, v88, v96
	v_mul_f32_e32 v88, v95, v85
	v_dot4_i32_i8 v96, v10, v89, v96
	v_fmac_f32_e32 v55, v88, v87
	v_add_u32_e32 v88, 0x3180, v84
	ds_read_b32 v87, v82 offset:12672
	ds_read2_b32 v[88:89], v88 offset1:1
	v_dot4_i32_i8 v96, v11, v90, v96
	v_add_u32_e32 v90, 0x3188, v84
	v_dot4_i32_i8 v96, v14, v91, v96
	ds_read2_b32 v[90:91], v90 offset1:1
	v_dot4_i32_i8 v96, v15, v92, v96
	v_add_u32_e32 v92, 0x3190, v84
	v_dot4_i32_i8 v96, v16, v93, v96
	ds_read2_b32 v[92:93], v92 offset1:1
	v_dot4_i32_i8 v96, v17, v94, v96
	v_add_u32_e32 v94, 0x3198, v84
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v8, v8, v88, 0
	v_mul_f32_e32 v97, v95, v86
	ds_read2_b32 v[94:95], v94 offset1:1
	v_dot4_i32_i8 v8, v9, v89, v8
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v8, v10, v90, v8
	v_dot4_i32_i8 v8, v11, v91, v8
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v8, v14, v92, v8
	v_dot4_i32_i8 v8, v15, v93, v8
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v8, v16, v94, v8
	v_dot4_i32_i8 v8, v17, v95, v8
	v_cvt_f32_i32_e32 v8, v8
	v_mul_f32_e32 v9, v87, v86
	v_cvt_f32_i32_e32 v96, v96
	v_add_u32_e32 v84, 32, v84
	v_fmac_f32_e32 v54, v9, v8
	v_dot4_i32_i8 v8, v12, v88, 0
	v_dot4_i32_i8 v8, v13, v89, v8
	v_dot4_i32_i8 v6, v6, v90, v8
	v_dot4_i32_i8 v6, v7, v91, v6
	v_dot4_i32_i8 v2, v2, v92, v6
	v_dot4_i32_i8 v2, v3, v93, v2
	v_dot4_i32_i8 v2, v4, v94, v2
	v_dot4_i32_i8 v2, v5, v95, v2
	v_cvt_f32_i32_e32 v2, v2
	v_mul_f32_e32 v3, v87, v85
	v_fmac_f32_e32 v56, v97, v96
	v_add_u32_e32 v82, 4, v82
	v_fmac_f32_e32 v53, v3, v2
	s_cbranch_scc1 .LBB44_65
; %bb.66:                               ; %_ZL22vec_dot_q8_0_q8_1_dp4aILi96ELi128ELi8EEvPKiS1_Pfi.exit.i666
                                        ;   in Loop: Header=BB44_64 Depth=2
	s_add_u32 s54, s54, s52
	s_addc_u32 s55, s55, s53
	v_mov_b32_e32 v2, s55
	v_add_co_u32_e32 v6, vcc, s54, v78
	v_addc_co_u32_e32 v7, vcc, 0, v2, vcc
	v_add_co_u32_e32 v2, vcc, s9, v6
	v_addc_co_u32_e32 v3, vcc, 0, v7, vcc
	v_add_co_u32_e32 v4, vcc, 0x2000, v6
	v_addc_co_u32_e32 v5, vcc, 0, v7, vcc
	v_add_co_u32_e32 v6, vcc, 0x3000, v6
	s_barrier
	v_addc_co_u32_e32 v7, vcc, 0, v7, vcc
	global_load_dword v8, v78, s[54:55]
	global_load_dword v9, v78, s[54:55] offset:2048
	global_load_dword v10, v[2:3], off
	s_nop 0
	global_load_dword v2, v[2:3], off offset:2048
	s_nop 0
	global_load_dword v3, v[4:5], off
	s_nop 0
	global_load_dword v4, v[4:5], off offset:2048
	s_nop 0
	global_load_dword v5, v[6:7], off
	s_mov_b32 s54, -8
	v_mov_b32_e32 v78, v44
	v_mov_b32_e32 v80, v49
	v_mov_b32_e32 v81, v48
	v_mov_b32_e32 v82, v47
	s_waitcnt vmcnt(5)
	ds_write2st64_b32 v79, v8, v9 offset0:1 offset1:9
	s_waitcnt vmcnt(3)
	ds_write2st64_b32 v79, v10, v2 offset0:17 offset1:25
	s_waitcnt vmcnt(1)
	ds_write2st64_b32 v79, v3, v4 offset0:33 offset1:41
	s_waitcnt vmcnt(0)
	ds_write_b32 v18, v5 offset:12672
	v_mov_b32_e32 v79, v42
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB44_67:                              ;   Parent Loop BB44_55 Depth=1
                                        ;     Parent Loop BB44_64 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ds_read_b32 v87, v78
	ds_read2_b32 v[2:3], v79 offset1:1
	ds_read2_b32 v[4:5], v79 offset0:2 offset1:3
	ds_read2_b32 v[6:7], v79 offset0:4 offset1:5
	ds_read2_b32 v[85:86], v79 offset0:6 offset1:7
	ds_read_b32 v84, v80
	ds_read2_b32 v[8:9], v82 offset1:1
	v_add_u32_e32 v89, 0x490, v79
	v_add_u32_e32 v91, 0x498, v79
	s_add_i32 s54, s54, 8
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v13, v84, v87
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v10, v8, v2, 0
	v_dot4_i32_i8 v12, v9, v3, v10
	ds_read2_b32 v[10:11], v82 offset0:2 offset1:3
	ds_read2_b32 v[14:15], v82 offset0:4 offset1:5
	ds_read2_b32 v[16:17], v82 offset0:6 offset1:7
	v_add_u32_e32 v80, 4, v80
	s_cmp_lt_u32 s54, 24
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v12, v10, v4, v12
	v_dot4_i32_i8 v12, v11, v5, v12
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v12, v14, v6, v12
	v_dot4_i32_i8 v12, v15, v7, v12
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v12, v16, v85, v12
	v_dot4_i32_i8 v12, v17, v86, v12
	v_cvt_f32_i32_e32 v12, v12
	ds_read2_b32 v[89:90], v89 offset1:1
	ds_read2_b32 v[91:92], v91 offset1:1
	v_fmac_f32_e32 v77, v13, v12
	v_add_u32_e32 v12, 0x4100, v82
	ds_read_b32 v83, v81
	ds_read2_b32 v[12:13], v12 offset1:1
	v_add_u32_e32 v81, 4, v81
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v2, v12, v2, 0
	v_dot4_i32_i8 v88, v13, v3, v2
	v_add_u32_e32 v2, 0x4108, v82
	ds_read2_b32 v[2:3], v2 offset1:1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v4, v2, v4, v88
	v_dot4_i32_i8 v88, v3, v5, v4
	v_add_u32_e32 v4, 0x4110, v82
	ds_read2_b32 v[4:5], v4 offset1:1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v6, v4, v6, v88
	v_dot4_i32_i8 v88, v5, v7, v6
	v_add_u32_e32 v6, 0x4118, v82
	ds_read2_b32 v[6:7], v6 offset1:1
	v_add_u32_e32 v82, 32, v82
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v85, v6, v85, v88
	v_dot4_i32_i8 v85, v7, v86, v85
	v_cvt_f32_i32_e32 v85, v85
	v_mul_f32_e32 v86, v83, v87
	v_add_u32_e32 v87, 0x488, v79
	ds_read2_b32 v[87:88], v87 offset1:1
	v_fmac_f32_e32 v76, v86, v85
	v_add_u32_e32 v85, 0x480, v79
	ds_read_b32 v93, v78 offset:1152
	ds_read2_b32 v[85:86], v85 offset1:1
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v95, v93, v84
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v94, v8, v85, 0
	v_dot4_i32_i8 v85, v12, v85, 0
	v_dot4_i32_i8 v85, v13, v86, v85
	v_dot4_i32_i8 v85, v2, v87, v85
	v_dot4_i32_i8 v85, v3, v88, v85
	v_dot4_i32_i8 v85, v4, v89, v85
	v_dot4_i32_i8 v85, v5, v90, v85
	v_dot4_i32_i8 v85, v6, v91, v85
	v_dot4_i32_i8 v94, v9, v86, v94
	v_dot4_i32_i8 v85, v7, v92, v85
	v_dot4_i32_i8 v94, v10, v87, v94
	v_cvt_f32_i32_e32 v85, v85
	v_dot4_i32_i8 v94, v11, v88, v94
	v_dot4_i32_i8 v94, v14, v89, v94
	v_dot4_i32_i8 v94, v15, v90, v94
	v_mul_f32_e32 v86, v93, v83
	v_dot4_i32_i8 v94, v16, v91, v94
	v_fmac_f32_e32 v74, v86, v85
	v_add_u32_e32 v85, 0x900, v79
	v_dot4_i32_i8 v94, v17, v92, v94
	ds_read_b32 v93, v78 offset:2304
	ds_read2_b32 v[85:86], v85 offset1:1
	v_cvt_f32_i32_e32 v94, v94
	v_add_u32_e32 v87, 0x908, v79
	ds_read2_b32 v[87:88], v87 offset1:1
	v_add_u32_e32 v89, 0x910, v79
	ds_read2_b32 v[89:90], v89 offset1:1
	v_fmac_f32_e32 v75, v95, v94
	v_add_u32_e32 v91, 0x918, v79
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v94, v8, v85, 0
	v_dot4_i32_i8 v85, v12, v85, 0
	ds_read2_b32 v[91:92], v91 offset1:1
	v_dot4_i32_i8 v85, v13, v86, v85
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v85, v2, v87, v85
	v_dot4_i32_i8 v85, v3, v88, v85
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v85, v4, v89, v85
	v_dot4_i32_i8 v85, v5, v90, v85
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v85, v6, v91, v85
	v_dot4_i32_i8 v94, v9, v86, v94
	v_dot4_i32_i8 v85, v7, v92, v85
	v_dot4_i32_i8 v94, v10, v87, v94
	v_cvt_f32_i32_e32 v85, v85
	v_dot4_i32_i8 v94, v11, v88, v94
	v_dot4_i32_i8 v94, v14, v89, v94
	v_dot4_i32_i8 v94, v15, v90, v94
	v_mul_f32_e32 v86, v93, v83
	v_dot4_i32_i8 v94, v16, v91, v94
	v_fmac_f32_e32 v72, v86, v85
	v_add_u32_e32 v85, 0xd80, v79
	v_dot4_i32_i8 v94, v17, v92, v94
	v_mul_f32_e32 v95, v93, v84
	ds_read_b32 v93, v78 offset:3456
	ds_read2_b32 v[85:86], v85 offset1:1
	v_cvt_f32_i32_e32 v94, v94
	v_add_u32_e32 v87, 0xd88, v79
	ds_read2_b32 v[87:88], v87 offset1:1
	v_add_u32_e32 v89, 0xd90, v79
	ds_read2_b32 v[89:90], v89 offset1:1
	v_fmac_f32_e32 v73, v95, v94
	v_add_u32_e32 v91, 0xd98, v79
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v94, v8, v85, 0
	v_dot4_i32_i8 v85, v12, v85, 0
	ds_read2_b32 v[91:92], v91 offset1:1
	v_dot4_i32_i8 v85, v13, v86, v85
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v85, v2, v87, v85
	v_dot4_i32_i8 v85, v3, v88, v85
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v85, v4, v89, v85
	v_dot4_i32_i8 v85, v5, v90, v85
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v85, v6, v91, v85
	v_dot4_i32_i8 v94, v9, v86, v94
	v_dot4_i32_i8 v85, v7, v92, v85
	v_dot4_i32_i8 v94, v10, v87, v94
	v_cvt_f32_i32_e32 v85, v85
	v_dot4_i32_i8 v94, v11, v88, v94
	v_dot4_i32_i8 v94, v14, v89, v94
	v_dot4_i32_i8 v94, v15, v90, v94
	v_mul_f32_e32 v86, v93, v83
	v_dot4_i32_i8 v94, v16, v91, v94
	v_fmac_f32_e32 v70, v86, v85
	v_add_u32_e32 v85, 0x1200, v79
	v_dot4_i32_i8 v94, v17, v92, v94
	v_mul_f32_e32 v95, v93, v84
	ds_read_b32 v93, v78 offset:4608
	ds_read2_b32 v[85:86], v85 offset1:1
	v_cvt_f32_i32_e32 v94, v94
	v_add_u32_e32 v87, 0x1208, v79
	ds_read2_b32 v[87:88], v87 offset1:1
	v_add_u32_e32 v89, 0x1210, v79
	ds_read2_b32 v[89:90], v89 offset1:1
	v_fmac_f32_e32 v71, v95, v94
	v_add_u32_e32 v91, 0x1218, v79
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v94, v8, v85, 0
	v_dot4_i32_i8 v85, v12, v85, 0
	ds_read2_b32 v[91:92], v91 offset1:1
	v_dot4_i32_i8 v85, v13, v86, v85
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v85, v2, v87, v85
	v_dot4_i32_i8 v85, v3, v88, v85
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v85, v4, v89, v85
	v_dot4_i32_i8 v85, v5, v90, v85
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v85, v6, v91, v85
	v_dot4_i32_i8 v94, v9, v86, v94
	v_dot4_i32_i8 v85, v7, v92, v85
	v_dot4_i32_i8 v94, v10, v87, v94
	v_cvt_f32_i32_e32 v85, v85
	v_dot4_i32_i8 v94, v11, v88, v94
	v_dot4_i32_i8 v94, v14, v89, v94
	v_dot4_i32_i8 v94, v15, v90, v94
	v_mul_f32_e32 v86, v93, v83
	v_dot4_i32_i8 v94, v16, v91, v94
	v_fmac_f32_e32 v67, v86, v85
	v_add_u32_e32 v85, 0x1680, v79
	v_dot4_i32_i8 v94, v17, v92, v94
	v_mul_f32_e32 v95, v93, v84
	ds_read_b32 v93, v78 offset:5760
	ds_read2_b32 v[85:86], v85 offset1:1
	v_cvt_f32_i32_e32 v94, v94
	v_add_u32_e32 v87, 0x1688, v79
	ds_read2_b32 v[87:88], v87 offset1:1
	v_add_u32_e32 v89, 0x1690, v79
	ds_read2_b32 v[89:90], v89 offset1:1
	v_fmac_f32_e32 v68, v95, v94
	v_add_u32_e32 v91, 0x1698, v79
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v94, v8, v85, 0
	v_dot4_i32_i8 v85, v12, v85, 0
	ds_read2_b32 v[91:92], v91 offset1:1
	v_dot4_i32_i8 v85, v13, v86, v85
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v85, v2, v87, v85
	v_dot4_i32_i8 v85, v3, v88, v85
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v85, v4, v89, v85
	v_dot4_i32_i8 v85, v5, v90, v85
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v85, v6, v91, v85
	v_dot4_i32_i8 v94, v9, v86, v94
	v_dot4_i32_i8 v85, v7, v92, v85
	v_dot4_i32_i8 v94, v10, v87, v94
	v_cvt_f32_i32_e32 v85, v85
	v_dot4_i32_i8 v94, v11, v88, v94
	v_dot4_i32_i8 v94, v14, v89, v94
	v_dot4_i32_i8 v94, v15, v90, v94
	v_mul_f32_e32 v86, v93, v83
	v_dot4_i32_i8 v94, v16, v91, v94
	v_fmac_f32_e32 v65, v86, v85
	v_add_u32_e32 v85, 0x1b00, v79
	v_dot4_i32_i8 v94, v17, v92, v94
	v_mul_f32_e32 v95, v93, v84
	ds_read_b32 v93, v78 offset:6912
	ds_read2_b32 v[85:86], v85 offset1:1
	v_cvt_f32_i32_e32 v94, v94
	v_add_u32_e32 v87, 0x1b08, v79
	ds_read2_b32 v[87:88], v87 offset1:1
	v_add_u32_e32 v89, 0x1b10, v79
	ds_read2_b32 v[89:90], v89 offset1:1
	v_fmac_f32_e32 v66, v95, v94
	v_add_u32_e32 v91, 0x1b18, v79
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v94, v8, v85, 0
	v_dot4_i32_i8 v85, v12, v85, 0
	ds_read2_b32 v[91:92], v91 offset1:1
	v_dot4_i32_i8 v85, v13, v86, v85
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v85, v2, v87, v85
	v_dot4_i32_i8 v85, v3, v88, v85
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v85, v4, v89, v85
	v_dot4_i32_i8 v85, v5, v90, v85
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v85, v6, v91, v85
	v_dot4_i32_i8 v94, v9, v86, v94
	v_dot4_i32_i8 v85, v7, v92, v85
	v_dot4_i32_i8 v94, v10, v87, v94
	v_cvt_f32_i32_e32 v85, v85
	v_dot4_i32_i8 v94, v11, v88, v94
	v_dot4_i32_i8 v94, v14, v89, v94
	v_dot4_i32_i8 v94, v15, v90, v94
	v_mul_f32_e32 v86, v93, v83
	v_dot4_i32_i8 v94, v16, v91, v94
	v_fmac_f32_e32 v63, v86, v85
	v_add_u32_e32 v85, 0x1f80, v79
	v_dot4_i32_i8 v94, v17, v92, v94
	v_mul_f32_e32 v95, v93, v84
	ds_read_b32 v93, v78 offset:8064
	ds_read2_b32 v[85:86], v85 offset1:1
	v_cvt_f32_i32_e32 v94, v94
	v_add_u32_e32 v87, 0x1f88, v79
	ds_read2_b32 v[87:88], v87 offset1:1
	v_add_u32_e32 v89, 0x1f90, v79
	ds_read2_b32 v[89:90], v89 offset1:1
	v_fmac_f32_e32 v64, v95, v94
	v_add_u32_e32 v91, 0x1f98, v79
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v94, v8, v85, 0
	v_dot4_i32_i8 v85, v12, v85, 0
	ds_read2_b32 v[91:92], v91 offset1:1
	v_dot4_i32_i8 v85, v13, v86, v85
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v85, v2, v87, v85
	v_dot4_i32_i8 v85, v3, v88, v85
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v85, v4, v89, v85
	v_dot4_i32_i8 v85, v5, v90, v85
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v85, v6, v91, v85
	v_dot4_i32_i8 v94, v9, v86, v94
	v_dot4_i32_i8 v85, v7, v92, v85
	v_dot4_i32_i8 v94, v10, v87, v94
	v_cvt_f32_i32_e32 v85, v85
	v_dot4_i32_i8 v94, v11, v88, v94
	v_dot4_i32_i8 v94, v14, v89, v94
	v_dot4_i32_i8 v94, v15, v90, v94
	v_mul_f32_e32 v86, v93, v83
	v_dot4_i32_i8 v94, v16, v91, v94
	v_fmac_f32_e32 v61, v86, v85
	v_add_u32_e32 v85, 0x2400, v79
	v_dot4_i32_i8 v94, v17, v92, v94
	v_mul_f32_e32 v95, v93, v84
	ds_read_b32 v93, v78 offset:9216
	ds_read2_b32 v[85:86], v85 offset1:1
	v_cvt_f32_i32_e32 v94, v94
	v_add_u32_e32 v87, 0x2408, v79
	ds_read2_b32 v[87:88], v87 offset1:1
	v_add_u32_e32 v89, 0x2410, v79
	ds_read2_b32 v[89:90], v89 offset1:1
	v_fmac_f32_e32 v62, v95, v94
	v_add_u32_e32 v91, 0x2418, v79
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v94, v8, v85, 0
	v_dot4_i32_i8 v85, v12, v85, 0
	ds_read2_b32 v[91:92], v91 offset1:1
	v_dot4_i32_i8 v85, v13, v86, v85
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v85, v2, v87, v85
	v_dot4_i32_i8 v85, v3, v88, v85
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v85, v4, v89, v85
	v_dot4_i32_i8 v85, v5, v90, v85
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v85, v6, v91, v85
	v_dot4_i32_i8 v94, v9, v86, v94
	v_dot4_i32_i8 v85, v7, v92, v85
	v_dot4_i32_i8 v94, v10, v87, v94
	v_cvt_f32_i32_e32 v85, v85
	v_dot4_i32_i8 v94, v11, v88, v94
	v_dot4_i32_i8 v94, v14, v89, v94
	v_dot4_i32_i8 v94, v15, v90, v94
	v_mul_f32_e32 v86, v93, v83
	v_dot4_i32_i8 v94, v16, v91, v94
	v_fmac_f32_e32 v59, v86, v85
	v_add_u32_e32 v85, 0x2880, v79
	v_dot4_i32_i8 v94, v17, v92, v94
	v_mul_f32_e32 v95, v93, v84
	ds_read_b32 v93, v78 offset:10368
	ds_read2_b32 v[85:86], v85 offset1:1
	v_cvt_f32_i32_e32 v94, v94
	v_add_u32_e32 v87, 0x2888, v79
	ds_read2_b32 v[87:88], v87 offset1:1
	v_add_u32_e32 v89, 0x2890, v79
	ds_read2_b32 v[89:90], v89 offset1:1
	v_fmac_f32_e32 v60, v95, v94
	v_add_u32_e32 v91, 0x2898, v79
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v94, v8, v85, 0
	v_dot4_i32_i8 v85, v12, v85, 0
	ds_read2_b32 v[91:92], v91 offset1:1
	v_dot4_i32_i8 v85, v13, v86, v85
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v85, v2, v87, v85
	v_dot4_i32_i8 v85, v3, v88, v85
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v85, v4, v89, v85
	v_dot4_i32_i8 v85, v5, v90, v85
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v85, v6, v91, v85
	v_dot4_i32_i8 v94, v9, v86, v94
	v_dot4_i32_i8 v85, v7, v92, v85
	v_dot4_i32_i8 v94, v10, v87, v94
	v_cvt_f32_i32_e32 v85, v85
	v_dot4_i32_i8 v94, v11, v88, v94
	v_dot4_i32_i8 v94, v14, v89, v94
	v_dot4_i32_i8 v94, v15, v90, v94
	v_mul_f32_e32 v86, v93, v83
	v_dot4_i32_i8 v94, v16, v91, v94
	v_fmac_f32_e32 v57, v86, v85
	v_add_u32_e32 v85, 0x2d00, v79
	v_dot4_i32_i8 v94, v17, v92, v94
	v_mul_f32_e32 v95, v93, v84
	ds_read_b32 v93, v78 offset:11520
	ds_read2_b32 v[85:86], v85 offset1:1
	v_cvt_f32_i32_e32 v94, v94
	v_add_u32_e32 v87, 0x2d08, v79
	ds_read2_b32 v[87:88], v87 offset1:1
	v_add_u32_e32 v89, 0x2d10, v79
	ds_read2_b32 v[89:90], v89 offset1:1
	v_fmac_f32_e32 v58, v95, v94
	v_add_u32_e32 v91, 0x2d18, v79
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v94, v8, v85, 0
	v_dot4_i32_i8 v85, v12, v85, 0
	ds_read2_b32 v[91:92], v91 offset1:1
	v_dot4_i32_i8 v85, v13, v86, v85
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v85, v2, v87, v85
	v_dot4_i32_i8 v85, v3, v88, v85
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v85, v4, v89, v85
	v_dot4_i32_i8 v85, v5, v90, v85
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v85, v6, v91, v85
	v_dot4_i32_i8 v85, v7, v92, v85
	v_cvt_f32_i32_e32 v85, v85
	v_dot4_i32_i8 v94, v9, v86, v94
	v_mul_f32_e32 v86, v93, v83
	v_mul_f32_e32 v95, v93, v84
	v_fmac_f32_e32 v55, v86, v85
	v_add_u32_e32 v85, 0x3180, v79
	ds_read_b32 v93, v78 offset:12672
	ds_read2_b32 v[85:86], v85 offset1:1
	v_dot4_i32_i8 v94, v10, v87, v94
	v_add_u32_e32 v87, 0x3188, v79
	v_dot4_i32_i8 v94, v11, v88, v94
	ds_read2_b32 v[87:88], v87 offset1:1
	v_dot4_i32_i8 v94, v14, v89, v94
	v_add_u32_e32 v89, 0x3190, v79
	v_dot4_i32_i8 v94, v15, v90, v94
	ds_read2_b32 v[89:90], v89 offset1:1
	v_dot4_i32_i8 v94, v16, v91, v94
	v_add_u32_e32 v91, 0x3198, v79
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v8, v8, v85, 0
	v_dot4_i32_i8 v94, v17, v92, v94
	ds_read2_b32 v[91:92], v91 offset1:1
	v_dot4_i32_i8 v8, v9, v86, v8
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v8, v10, v87, v8
	v_dot4_i32_i8 v8, v11, v88, v8
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v8, v14, v89, v8
	v_dot4_i32_i8 v8, v15, v90, v8
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v8, v16, v91, v8
	v_dot4_i32_i8 v8, v17, v92, v8
	v_cvt_f32_i32_e32 v8, v8
	v_mul_f32_e32 v9, v93, v84
	v_cvt_f32_i32_e32 v94, v94
	v_add_u32_e32 v79, 32, v79
	v_fmac_f32_e32 v54, v9, v8
	v_dot4_i32_i8 v8, v12, v85, 0
	v_dot4_i32_i8 v8, v13, v86, v8
	v_dot4_i32_i8 v2, v2, v87, v8
	v_dot4_i32_i8 v2, v3, v88, v2
	v_dot4_i32_i8 v2, v4, v89, v2
	v_dot4_i32_i8 v2, v5, v90, v2
	v_dot4_i32_i8 v2, v6, v91, v2
	v_dot4_i32_i8 v2, v7, v92, v2
	v_cvt_f32_i32_e32 v2, v2
	v_mul_f32_e32 v3, v93, v83
	v_fmac_f32_e32 v56, v95, v94
	v_add_u32_e32 v78, 4, v78
	v_fmac_f32_e32 v53, v3, v2
	s_cbranch_scc1 .LBB44_67
; %bb.68:                               ; %_ZL22vec_dot_q8_0_q8_1_dp4aILi96ELi128ELi8EEvPKiS1_Pfi.exit104.i735
                                        ;   in Loop: Header=BB44_64 Depth=2
	s_add_i32 s7, s7, 8
	s_cmp_ge_i32 s7, s23
	s_barrier
	s_cbranch_scc0 .LBB44_64
.LBB44_69:                              ; %._crit_edge.i477
                                        ;   in Loop: Header=BB44_55 Depth=1
	s_not_b32 s7, s58
	v_add_u32_e32 v2, s7, v69
	v_cmp_le_i32_e32 vcc, v1, v2
	s_and_saveexec_b64 s[54:55], vcc
	s_cbranch_execz .LBB44_53
; %bb.70:                               ; %.preheader.i.i505
                                        ;   in Loop: Header=BB44_55 Depth=1
	ds_read_b32 v3, v30
	s_add_i32 s56, s59, s56
	s_ashr_i32 s57, s56, 31
	s_lshl_b64 s[56:57], s[56:57], 2
	s_add_u32 s7, s28, s56
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[3:4], s[58:59], v3, s27, v[0:1]
	s_addc_u32 s23, s29, s57
	v_mov_b32_e32 v5, s23
	v_ashrrev_i32_e32 v4, 31, v3
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_add_co_u32_e32 v3, vcc, s7, v3
	v_addc_co_u32_e32 v4, vcc, v5, v4, vcc
	v_cmp_le_u32_e32 vcc, v31, v2
	global_store_dword v[3:4], v77, off
	global_store_dword v[3:4], v76, off offset:256
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB44_53
; %bb.71:                               ; %.preheader.1.i.i509
                                        ;   in Loop: Header=BB44_55 Depth=1
	ds_read_b32 v3, v30 offset:32
	v_mov_b32_e32 v5, s23
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[3:4], s[56:57], v3, s27, v[0:1]
	v_ashrrev_i32_e32 v4, 31, v3
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_add_co_u32_e32 v3, vcc, s7, v3
	v_addc_co_u32_e32 v4, vcc, v5, v4, vcc
	v_cmp_le_u32_e32 vcc, v32, v2
	global_store_dword v[3:4], v75, off
	global_store_dword v[3:4], v74, off offset:256
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB44_53
; %bb.72:                               ; %.preheader.2.i.i513
                                        ;   in Loop: Header=BB44_55 Depth=1
	ds_read_b32 v3, v30 offset:64
	v_mov_b32_e32 v5, s23
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[3:4], s[56:57], v3, s27, v[0:1]
	v_ashrrev_i32_e32 v4, 31, v3
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_add_co_u32_e32 v3, vcc, s7, v3
	v_addc_co_u32_e32 v4, vcc, v5, v4, vcc
	v_cmp_le_u32_e32 vcc, v33, v2
	global_store_dword v[3:4], v73, off
	global_store_dword v[3:4], v72, off offset:256
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB44_53
; %bb.73:                               ; %.preheader.3.i.i517
                                        ;   in Loop: Header=BB44_55 Depth=1
	ds_read_b32 v3, v30 offset:96
	v_mov_b32_e32 v5, s23
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[3:4], s[56:57], v3, s27, v[0:1]
	v_ashrrev_i32_e32 v4, 31, v3
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_add_co_u32_e32 v3, vcc, s7, v3
	v_addc_co_u32_e32 v4, vcc, v5, v4, vcc
	v_cmp_le_u32_e32 vcc, v34, v2
	global_store_dword v[3:4], v71, off
	global_store_dword v[3:4], v70, off offset:256
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB44_53
; %bb.74:                               ; %.preheader.4.i.i521
                                        ;   in Loop: Header=BB44_55 Depth=1
	ds_read_b32 v3, v30 offset:128
	v_mov_b32_e32 v5, s23
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[3:4], s[56:57], v3, s27, v[0:1]
	v_ashrrev_i32_e32 v4, 31, v3
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_add_co_u32_e32 v3, vcc, s7, v3
	v_addc_co_u32_e32 v4, vcc, v5, v4, vcc
	v_cmp_le_u32_e32 vcc, v35, v2
	global_store_dword v[3:4], v68, off
	global_store_dword v[3:4], v67, off offset:256
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB44_53
; %bb.75:                               ; %.preheader.5.i.i525
                                        ;   in Loop: Header=BB44_55 Depth=1
	ds_read_b32 v3, v30 offset:160
	v_mov_b32_e32 v5, s23
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[3:4], s[56:57], v3, s27, v[0:1]
	v_ashrrev_i32_e32 v4, 31, v3
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_add_co_u32_e32 v3, vcc, s7, v3
	v_addc_co_u32_e32 v4, vcc, v5, v4, vcc
	v_cmp_le_u32_e32 vcc, v36, v2
	global_store_dword v[3:4], v66, off
	global_store_dword v[3:4], v65, off offset:256
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB44_53
; %bb.76:                               ; %.preheader.6.i.i529
                                        ;   in Loop: Header=BB44_55 Depth=1
	ds_read_b32 v3, v30 offset:192
	v_mov_b32_e32 v5, s23
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[3:4], s[56:57], v3, s27, v[0:1]
	v_ashrrev_i32_e32 v4, 31, v3
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_add_co_u32_e32 v3, vcc, s7, v3
	v_addc_co_u32_e32 v4, vcc, v5, v4, vcc
	v_cmp_le_u32_e32 vcc, v37, v2
	global_store_dword v[3:4], v64, off
	global_store_dword v[3:4], v63, off offset:256
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB44_53
; %bb.77:                               ; %.preheader.7.i.i533
                                        ;   in Loop: Header=BB44_55 Depth=1
	ds_read_b32 v3, v30 offset:224
	v_mov_b32_e32 v5, s23
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[3:4], s[56:57], v3, s27, v[0:1]
	v_ashrrev_i32_e32 v4, 31, v3
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_add_co_u32_e32 v3, vcc, s7, v3
	v_addc_co_u32_e32 v4, vcc, v5, v4, vcc
	v_cmp_le_u32_e32 vcc, v38, v2
	global_store_dword v[3:4], v62, off
	global_store_dword v[3:4], v61, off offset:256
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB44_53
; %bb.78:                               ; %.preheader.8.i.i537
                                        ;   in Loop: Header=BB44_55 Depth=1
	ds_read_b32 v3, v30 offset:256
	v_mov_b32_e32 v5, s23
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[3:4], s[56:57], v3, s27, v[0:1]
	v_ashrrev_i32_e32 v4, 31, v3
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_add_co_u32_e32 v3, vcc, s7, v3
	v_addc_co_u32_e32 v4, vcc, v5, v4, vcc
	v_cmp_le_u32_e32 vcc, v39, v2
	global_store_dword v[3:4], v60, off
	global_store_dword v[3:4], v59, off offset:256
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB44_53
; %bb.79:                               ; %.preheader.9.i.i541
                                        ;   in Loop: Header=BB44_55 Depth=1
	ds_read_b32 v3, v30 offset:288
	v_mov_b32_e32 v5, s23
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[3:4], s[56:57], v3, s27, v[0:1]
	v_ashrrev_i32_e32 v4, 31, v3
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_add_co_u32_e32 v3, vcc, s7, v3
	v_addc_co_u32_e32 v4, vcc, v5, v4, vcc
	v_cmp_le_u32_e32 vcc, v40, v2
	global_store_dword v[3:4], v58, off
	global_store_dword v[3:4], v57, off offset:256
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB44_53
; %bb.80:                               ; %.preheader.10.i.i545
                                        ;   in Loop: Header=BB44_55 Depth=1
	ds_read_b32 v3, v30 offset:320
	v_mov_b32_e32 v5, s23
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[3:4], s[56:57], v3, s27, v[0:1]
	v_ashrrev_i32_e32 v4, 31, v3
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_add_co_u32_e32 v3, vcc, s7, v3
	v_addc_co_u32_e32 v4, vcc, v5, v4, vcc
	v_cmp_le_u32_e32 vcc, v41, v2
	global_store_dword v[3:4], v56, off
	global_store_dword v[3:4], v55, off offset:256
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB44_53
; %bb.81:                               ; %.preheader.11.i.i549
                                        ;   in Loop: Header=BB44_55 Depth=1
	ds_read_b32 v2, v30 offset:352
	v_mov_b32_e32 v4, s23
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[2:3], s[56:57], v2, s27, v[0:1]
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_add_co_u32_e32 v2, vcc, s7, v2
	v_addc_co_u32_e32 v3, vcc, v4, v3, vcc
	global_store_dword v[2:3], v54, off
	global_store_dword v[2:3], v53, off offset:256
	s_branch .LBB44_53
.LBB44_82:                              ; %._crit_edge.loopexit
	s_min_u32 s23, s22, s56
.LBB44_83:                              ; %._crit_edge
	s_andn2_b64 vcc, exec, s[54:55]
	s_cbranch_vccnz .LBB44_125
; %bb.84:
	s_mul_hi_u32 s4, s33, s20
	s_add_i32 s33, s33, s4
	s_lshr_b32 s4, s33, s21
	s_mul_hi_u32 s5, s4, s50
	s_add_i32 s5, s4, s5
	s_lshr_b32 s5, s5, s51
	s_mul_i32 s8, s5, s34
	s_sub_i32 s22, s4, s8
	s_mul_hi_u32 s4, s5, s36
	s_add_i32 s4, s5, s4
	s_lshr_b32 s8, s4, s37
	s_mul_i32 s4, s8, s38
	s_sub_i32 s4, s5, s4
	s_mul_hi_u32 s5, s8, s44
	s_add_i32 s5, s8, s5
	s_lshr_b32 s21, s5, s45
	s_mul_i32 s5, s21, s46
	s_sub_i32 s20, s8, s5
	s_mulk_i32 s22, 0x60
	s_cmp_lg_u64 s[16:17], 0
	s_mov_b64 s[8:9], 0
	s_cbranch_scc0 .LBB44_124
; %bb.85:
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[16:17], s[4:5], 2
	s_add_u32 s16, s18, s16
	s_addc_u32 s17, s19, s17
	v_mov_b32_e32 v2, 0
	global_load_dwordx2 v[2:3], v2, s[16:17]
	s_mov_b32 s5, 0
	s_mov_b32 s16, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s17, v2
	v_subrev_u32_e32 v2, s17, v3
	v_cmp_lt_i32_e32 vcc, s22, v2
	s_cbranch_vccz .LBB44_89
; %bb.86:
	s_barrier
	s_and_saveexec_b64 s[8:9], s[0:1]
; %bb.87:                               ; %.lr.ph1114
	v_lshl_add_u32 v2, v24, 2, 0
	ds_write_b32 v2, v24
; %bb.88:                               ; %.critedge467
	s_or_b64 exec, exec, s[8:9]
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_mov_b64 s[8:9], -1
	s_mov_b32 s16, s17
.LBB44_89:                              ; %Flow3267
	s_and_b64 vcc, exec, s[8:9]
	s_cbranch_vccz .LBB44_125
.LBB44_90:
	v_mov_b32_e32 v25, 0
	s_cmp_ge_i32 s7, s23
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	s_cbranch_scc1 .LBB44_97
; %bb.91:                               ; %.lr.ph.i821
	s_add_i32 s0, s16, s22
	s_mul_i32 s0, s0, 36
	s_add_i32 s0, s0, s5
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 2
	s_add_u32 s14, s14, s0
	s_mul_i32 s0, s25, s21
	s_addc_u32 s15, s15, s1
	s_lshl_b32 s1, s0, 7
	s_mul_hi_u32 s0, s4, s10
	s_add_i32 s0, s4, s0
	v_lshrrev_b32_e32 v3, 4, v0
	s_lshr_b32 s0, s0, s11
	v_lshl_add_u32 v3, v1, 2, v3
	s_mul_i32 s4, s0, s39
	s_mul_hi_u32 s0, s20, s42
	v_mul_lo_u32 v45, s25, v3
	s_movk_i32 s8, 0x104
	s_add_i32 s20, s20, s0
	v_and_b32_e32 v2, 15, v0
	v_mad_u32_u24 v3, v3, s8, 0
	s_lshr_b32 s0, s20, s43
	v_lshl_add_u32 v46, v2, 2, v3
	v_lshrrev_b32_e32 v2, 3, v0
	s_mul_i32 s5, s0, s47
	s_lshl_b32 s0, s25, 5
	v_lshl_add_u32 v2, v1, 3, v2
	v_and_b32_e32 v44, 7, v0
	v_add_u32_e32 v47, s0, v45
	v_mul_lo_u32 v50, s25, v2
	v_and_b32_e32 v3, 0x1ffc, v2
	v_lshlrev_b32_e32 v6, 5, v2
	v_add_u32_e32 v2, 64, v2
	v_mul_u32_u24_e32 v8, 36, v1
	v_add_u32_e32 v48, s0, v47
	v_lshlrev_b32_e32 v5, 2, v44
	v_and_b32_e32 v7, 0x3ffc, v2
	v_add_u32_e32 v9, 64, v0
	v_lshl_add_u32 v8, v8, 2, 0
	v_add_u32_e32 v49, s0, v48
	v_add3_u32 v3, 0, v3, v5
	v_add3_u32 v5, 0, v7, v5
	s_mul_i32 s0, s26, 36
	v_lshlrev_b32_e32 v7, 5, v0
	v_add_u32_e32 v53, 0x190, v8
	v_add_u32_e32 v55, 0x180, v8
	v_and_b32_e32 v8, 0x3fc, v9
	v_and_b32_e32 v9, 0x1fc, v0
	v_lshlrev_b32_e32 v4, 1, v44
	v_lshlrev_b32_e32 v2, 5, v2
	s_add_i32 s10, s4, s1
	s_ashr_i32 s1, s0, 31
	v_mad_u32_u24 v10, v0, s8, 0
	v_add3_u32 v8, v7, v8, 0
	v_add3_u32 v7, v7, v9, 0
	v_bfe_u32 v43, v0, 3, 1
	v_mov_b32_e32 v25, 0
	v_lshl_add_u32 v51, s25, 6, v50
	v_lshl_add_u32 v52, v24, 2, 0
	s_add_i32 s10, s10, s5
	s_lshl_b64 s[4:5], s[0:1], 2
	v_add_u32_e32 v54, 0x3980, v10
	v_add_u32_e32 v56, 0xc380, v8
	v_add_u32_e32 v57, 0xbb80, v7
	v_add_u32_e32 v58, 0x3a00, v10
	v_add_u32_e32 v59, 0xc390, v8
	v_add_u32_e32 v60, 0xbb90, v7
	v_lshlrev_b32_e32 v61, 1, v4
	v_add_u32_e32 v62, v3, v6
	v_add_u32_e32 v63, v5, v2
	v_lshlrev_b32_e32 v24, 2, v24
	s_movk_i32 s1, 0x1000
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
.LBB44_92:                              ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB44_93 Depth 2
                                        ;     Child Loop BB44_95 Depth 2
	s_add_i32 s8, s10, s7
	s_mul_hi_i32 s9, s8, 34
	s_mul_i32 s8, s8, 34
	s_add_u32 s8, s12, s8
	s_addc_u32 s9, s13, s9
	v_mad_u64_u32 v[2:3], s[16:17], v43, 34, s[8:9]
	s_mov_b32 s11, -8
	v_mad_i64_i32 v[4:5], s[16:17], v45, 34, v[2:3]
	v_mad_i64_i32 v[6:7], s[16:17], v47, 34, v[2:3]
	v_add_co_u32_e32 v4, vcc, v4, v61
	v_addc_co_u32_e32 v5, vcc, 0, v5, vcc
	v_add_co_u32_e32 v6, vcc, v6, v61
	v_addc_co_u32_e32 v7, vcc, 0, v7, vcc
	v_mad_i64_i32 v[8:9], s[16:17], v48, 34, v[2:3]
	global_load_dword v10, v[4:5], off offset:2
	global_load_dword v11, v[4:5], off offset:70
	global_load_dword v12, v[4:5], off offset:138
	global_load_dword v13, v[4:5], off offset:206
	global_load_dword v14, v[6:7], off offset:2
	global_load_dword v15, v[6:7], off offset:70
	global_load_dword v16, v[6:7], off offset:138
	global_load_dword v17, v[6:7], off offset:206
	v_mad_u64_u32 v[6:7], s[8:9], v44, 34, s[8:9]
	v_add_co_u32_e32 v4, vcc, v8, v61
	v_addc_co_u32_e32 v5, vcc, 0, v9, vcc
	v_mad_i64_i32 v[8:9], s[8:9], v50, 34, v[6:7]
	v_mad_i64_i32 v[6:7], s[8:9], v51, 34, v[6:7]
	s_ashr_i32 s8, s7, 31
	s_lshr_b32 s8, s8, 30
	s_add_i32 s8, s7, s8
	v_mad_i64_i32 v[2:3], s[16:17], v49, 34, v[2:3]
	s_ashr_i32 s8, s8, 2
	s_mul_i32 s8, s0, s8
	s_ashr_i32 s9, s8, 31
	s_lshl_b64 s[8:9], s[8:9], 2
	v_add_co_u32_e32 v2, vcc, v2, v61
	s_add_u32 s8, s14, s8
	v_addc_co_u32_e32 v3, vcc, 0, v3, vcc
	s_addc_u32 s9, s15, s9
	global_load_dword v64, v[4:5], off offset:2
	global_load_dword v65, v[4:5], off offset:70
	global_load_dword v66, v[4:5], off offset:138
	global_load_dword v67, v[4:5], off offset:206
	global_load_dword v68, v[2:3], off offset:2
	global_load_dword v69, v[2:3], off offset:70
	global_load_dword v70, v[2:3], off offset:138
	global_load_dword v71, v[2:3], off offset:206
	s_nop 0
	global_load_ushort v8, v[8:9], off
	s_nop 0
	global_load_ushort v9, v[6:7], off
	v_mov_b32_e32 v2, s9
	v_add_co_u32_e32 v6, vcc, s8, v24
	v_addc_co_u32_e32 v7, vcc, 0, v2, vcc
	v_add_co_u32_e32 v2, vcc, s1, v6
	v_addc_co_u32_e32 v3, vcc, 0, v7, vcc
	v_add_co_u32_e32 v4, vcc, 0x2000, v6
	v_addc_co_u32_e32 v5, vcc, 0, v7, vcc
	v_add_co_u32_e32 v6, vcc, 0x3000, v6
	v_addc_co_u32_e32 v7, vcc, 0, v7, vcc
	global_load_dword v72, v24, s[8:9]
	global_load_dword v73, v24, s[8:9] offset:2048
	global_load_dword v74, v[2:3], off
	s_nop 0
	global_load_dword v2, v[2:3], off offset:2048
	s_nop 0
	global_load_dword v3, v[4:5], off
	s_nop 0
	global_load_dword v4, v[4:5], off offset:2048
	s_nop 0
	global_load_dword v5, v[6:7], off
	v_add_u32_e32 v6, 0x3800, v46
	s_waitcnt vmcnt(23)
	ds_write2_b32 v6, v10, v11 offset0:96 offset1:112
	s_waitcnt vmcnt(21)
	ds_write2_b32 v6, v12, v13 offset0:128 offset1:144
	v_add_u32_e32 v6, 0x5800, v46
	s_waitcnt vmcnt(19)
	ds_write2_b32 v6, v14, v15 offset0:128 offset1:144
	s_waitcnt vmcnt(17)
	ds_write2_b32 v6, v16, v17 offset0:160 offset1:176
	v_add_u32_e32 v6, 0x7800, v46
	s_waitcnt vmcnt(15)
	ds_write2_b32 v6, v64, v65 offset0:160 offset1:176
	s_waitcnt vmcnt(13)
	ds_write2_b32 v6, v66, v67 offset0:192 offset1:208
	v_add_u32_e32 v6, 0x9800, v46
	v_add_u32_e32 v64, 0x80, v52
	v_mov_b32_e32 v65, v57
	v_mov_b32_e32 v66, v56
	s_waitcnt vmcnt(8)
	v_cvt_f32_f16_e32 v7, v8
	s_waitcnt vmcnt(7)
	v_cvt_f32_f16_e32 v8, v9
	ds_write2_b32 v6, v68, v69 offset0:192 offset1:208
	ds_write2_b32 v6, v70, v71 offset0:224 offset1:240
	ds_write_b32 v62, v7 offset:48000
	ds_write_b32 v63, v8 offset:48000
	v_mov_b32_e32 v67, v55
	v_mov_b32_e32 v68, v54
	v_mov_b32_e32 v69, v53
	s_waitcnt vmcnt(5)
	ds_write2st64_b32 v64, v72, v73 offset0:1 offset1:9
	s_waitcnt vmcnt(3)
	ds_write2st64_b32 v64, v74, v2 offset0:17 offset1:25
	s_waitcnt vmcnt(1)
	ds_write2st64_b32 v64, v3, v4 offset0:33 offset1:41
	s_waitcnt vmcnt(0)
	ds_write_b32 v52, v5 offset:12672
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB44_93:                              ;   Parent Loop BB44_92 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ds_read_b32 v74, v67
	ds_read2_b32 v[2:3], v69 offset1:1
	ds_read2_b32 v[4:5], v69 offset0:2 offset1:3
	ds_read2_b32 v[6:7], v69 offset0:4 offset1:5
	ds_read2_b32 v[72:73], v69 offset0:6 offset1:7
	ds_read_b32 v71, v65
	ds_read2_b32 v[8:9], v68 offset1:1
	v_add_u32_e32 v76, 0x490, v69
	v_add_u32_e32 v78, 0x498, v69
	s_add_i32 s11, s11, 8
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v13, v71, v74
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v10, v8, v2, 0
	v_dot4_i32_i8 v12, v9, v3, v10
	ds_read2_b32 v[10:11], v68 offset0:2 offset1:3
	ds_read2_b32 v[14:15], v68 offset0:4 offset1:5
	ds_read2_b32 v[16:17], v68 offset0:6 offset1:7
	v_add_u32_e32 v65, 4, v65
	s_cmp_lt_u32 s11, 24
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v12, v10, v4, v12
	v_dot4_i32_i8 v12, v11, v5, v12
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v12, v14, v6, v12
	v_dot4_i32_i8 v12, v15, v7, v12
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v12, v16, v72, v12
	v_dot4_i32_i8 v12, v17, v73, v12
	v_cvt_f32_i32_e32 v12, v12
	ds_read2_b32 v[76:77], v76 offset1:1
	ds_read2_b32 v[78:79], v78 offset1:1
	v_fmac_f32_e32 v42, v13, v12
	v_add_u32_e32 v12, 0x4100, v68
	ds_read_b32 v70, v66
	ds_read2_b32 v[12:13], v12 offset1:1
	v_add_u32_e32 v66, 4, v66
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v2, v12, v2, 0
	v_dot4_i32_i8 v75, v13, v3, v2
	v_add_u32_e32 v2, 0x4108, v68
	ds_read2_b32 v[2:3], v2 offset1:1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v4, v2, v4, v75
	v_dot4_i32_i8 v75, v3, v5, v4
	v_add_u32_e32 v4, 0x4110, v68
	ds_read2_b32 v[4:5], v4 offset1:1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v6, v4, v6, v75
	v_dot4_i32_i8 v75, v5, v7, v6
	v_add_u32_e32 v6, 0x4118, v68
	ds_read2_b32 v[6:7], v6 offset1:1
	v_add_u32_e32 v68, 32, v68
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v72, v6, v72, v75
	v_dot4_i32_i8 v72, v7, v73, v72
	v_cvt_f32_i32_e32 v72, v72
	v_mul_f32_e32 v73, v70, v74
	v_add_u32_e32 v74, 0x488, v69
	ds_read2_b32 v[74:75], v74 offset1:1
	v_fmac_f32_e32 v41, v73, v72
	v_add_u32_e32 v72, 0x480, v69
	ds_read_b32 v80, v67 offset:1152
	ds_read2_b32 v[72:73], v72 offset1:1
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v82, v80, v71
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v81, v8, v72, 0
	v_dot4_i32_i8 v72, v12, v72, 0
	v_dot4_i32_i8 v72, v13, v73, v72
	v_dot4_i32_i8 v72, v2, v74, v72
	v_dot4_i32_i8 v72, v3, v75, v72
	v_dot4_i32_i8 v72, v4, v76, v72
	v_dot4_i32_i8 v72, v5, v77, v72
	v_dot4_i32_i8 v72, v6, v78, v72
	v_dot4_i32_i8 v81, v9, v73, v81
	v_dot4_i32_i8 v72, v7, v79, v72
	v_dot4_i32_i8 v81, v10, v74, v81
	v_cvt_f32_i32_e32 v72, v72
	v_dot4_i32_i8 v81, v11, v75, v81
	v_dot4_i32_i8 v81, v14, v76, v81
	v_dot4_i32_i8 v81, v15, v77, v81
	v_mul_f32_e32 v73, v80, v70
	v_dot4_i32_i8 v81, v16, v78, v81
	v_fmac_f32_e32 v39, v73, v72
	v_add_u32_e32 v72, 0x900, v69
	v_dot4_i32_i8 v81, v17, v79, v81
	ds_read_b32 v80, v67 offset:2304
	ds_read2_b32 v[72:73], v72 offset1:1
	v_cvt_f32_i32_e32 v81, v81
	v_add_u32_e32 v74, 0x908, v69
	ds_read2_b32 v[74:75], v74 offset1:1
	v_add_u32_e32 v76, 0x910, v69
	ds_read2_b32 v[76:77], v76 offset1:1
	v_fmac_f32_e32 v40, v82, v81
	v_add_u32_e32 v78, 0x918, v69
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v8, v72, 0
	v_dot4_i32_i8 v72, v12, v72, 0
	ds_read2_b32 v[78:79], v78 offset1:1
	v_dot4_i32_i8 v72, v13, v73, v72
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v72, v2, v74, v72
	v_dot4_i32_i8 v72, v3, v75, v72
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v72, v4, v76, v72
	v_dot4_i32_i8 v72, v5, v77, v72
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v72, v6, v78, v72
	v_dot4_i32_i8 v81, v9, v73, v81
	v_dot4_i32_i8 v72, v7, v79, v72
	v_dot4_i32_i8 v81, v10, v74, v81
	v_cvt_f32_i32_e32 v72, v72
	v_dot4_i32_i8 v81, v11, v75, v81
	v_dot4_i32_i8 v81, v14, v76, v81
	v_dot4_i32_i8 v81, v15, v77, v81
	v_mul_f32_e32 v73, v80, v70
	v_dot4_i32_i8 v81, v16, v78, v81
	v_fmac_f32_e32 v37, v73, v72
	v_add_u32_e32 v72, 0xd80, v69
	v_dot4_i32_i8 v81, v17, v79, v81
	v_mul_f32_e32 v82, v80, v71
	ds_read_b32 v80, v67 offset:3456
	ds_read2_b32 v[72:73], v72 offset1:1
	v_cvt_f32_i32_e32 v81, v81
	v_add_u32_e32 v74, 0xd88, v69
	ds_read2_b32 v[74:75], v74 offset1:1
	v_add_u32_e32 v76, 0xd90, v69
	ds_read2_b32 v[76:77], v76 offset1:1
	v_fmac_f32_e32 v38, v82, v81
	v_add_u32_e32 v78, 0xd98, v69
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v8, v72, 0
	v_dot4_i32_i8 v72, v12, v72, 0
	ds_read2_b32 v[78:79], v78 offset1:1
	v_dot4_i32_i8 v72, v13, v73, v72
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v72, v2, v74, v72
	v_dot4_i32_i8 v72, v3, v75, v72
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v72, v4, v76, v72
	v_dot4_i32_i8 v72, v5, v77, v72
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v72, v6, v78, v72
	v_dot4_i32_i8 v81, v9, v73, v81
	v_dot4_i32_i8 v72, v7, v79, v72
	v_dot4_i32_i8 v81, v10, v74, v81
	v_cvt_f32_i32_e32 v72, v72
	v_dot4_i32_i8 v81, v11, v75, v81
	v_dot4_i32_i8 v81, v14, v76, v81
	v_dot4_i32_i8 v81, v15, v77, v81
	v_mul_f32_e32 v73, v80, v70
	v_dot4_i32_i8 v81, v16, v78, v81
	v_fmac_f32_e32 v35, v73, v72
	v_add_u32_e32 v72, 0x1200, v69
	v_dot4_i32_i8 v81, v17, v79, v81
	v_mul_f32_e32 v82, v80, v71
	ds_read_b32 v80, v67 offset:4608
	ds_read2_b32 v[72:73], v72 offset1:1
	v_cvt_f32_i32_e32 v81, v81
	v_add_u32_e32 v74, 0x1208, v69
	ds_read2_b32 v[74:75], v74 offset1:1
	v_add_u32_e32 v76, 0x1210, v69
	ds_read2_b32 v[76:77], v76 offset1:1
	v_fmac_f32_e32 v36, v82, v81
	v_add_u32_e32 v78, 0x1218, v69
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v8, v72, 0
	v_dot4_i32_i8 v72, v12, v72, 0
	ds_read2_b32 v[78:79], v78 offset1:1
	v_dot4_i32_i8 v72, v13, v73, v72
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v72, v2, v74, v72
	v_dot4_i32_i8 v72, v3, v75, v72
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v72, v4, v76, v72
	v_dot4_i32_i8 v72, v5, v77, v72
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v72, v6, v78, v72
	v_dot4_i32_i8 v81, v9, v73, v81
	v_dot4_i32_i8 v72, v7, v79, v72
	v_dot4_i32_i8 v81, v10, v74, v81
	v_cvt_f32_i32_e32 v72, v72
	v_dot4_i32_i8 v81, v11, v75, v81
	v_dot4_i32_i8 v81, v14, v76, v81
	v_dot4_i32_i8 v81, v15, v77, v81
	v_mul_f32_e32 v73, v80, v70
	v_dot4_i32_i8 v81, v16, v78, v81
	v_fmac_f32_e32 v33, v73, v72
	v_add_u32_e32 v72, 0x1680, v69
	v_dot4_i32_i8 v81, v17, v79, v81
	v_mul_f32_e32 v82, v80, v71
	ds_read_b32 v80, v67 offset:5760
	ds_read2_b32 v[72:73], v72 offset1:1
	v_cvt_f32_i32_e32 v81, v81
	v_add_u32_e32 v74, 0x1688, v69
	ds_read2_b32 v[74:75], v74 offset1:1
	v_add_u32_e32 v76, 0x1690, v69
	ds_read2_b32 v[76:77], v76 offset1:1
	v_fmac_f32_e32 v34, v82, v81
	v_add_u32_e32 v78, 0x1698, v69
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v8, v72, 0
	v_dot4_i32_i8 v72, v12, v72, 0
	ds_read2_b32 v[78:79], v78 offset1:1
	v_dot4_i32_i8 v72, v13, v73, v72
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v72, v2, v74, v72
	v_dot4_i32_i8 v72, v3, v75, v72
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v72, v4, v76, v72
	v_dot4_i32_i8 v72, v5, v77, v72
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v72, v6, v78, v72
	v_dot4_i32_i8 v81, v9, v73, v81
	v_dot4_i32_i8 v72, v7, v79, v72
	v_dot4_i32_i8 v81, v10, v74, v81
	v_cvt_f32_i32_e32 v72, v72
	v_dot4_i32_i8 v81, v11, v75, v81
	v_dot4_i32_i8 v81, v14, v76, v81
	v_dot4_i32_i8 v81, v15, v77, v81
	v_mul_f32_e32 v73, v80, v70
	v_dot4_i32_i8 v81, v16, v78, v81
	v_fmac_f32_e32 v31, v73, v72
	v_add_u32_e32 v72, 0x1b00, v69
	v_dot4_i32_i8 v81, v17, v79, v81
	v_mul_f32_e32 v82, v80, v71
	ds_read_b32 v80, v67 offset:6912
	ds_read2_b32 v[72:73], v72 offset1:1
	v_cvt_f32_i32_e32 v81, v81
	v_add_u32_e32 v74, 0x1b08, v69
	ds_read2_b32 v[74:75], v74 offset1:1
	v_add_u32_e32 v76, 0x1b10, v69
	ds_read2_b32 v[76:77], v76 offset1:1
	v_fmac_f32_e32 v32, v82, v81
	v_add_u32_e32 v78, 0x1b18, v69
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v8, v72, 0
	v_dot4_i32_i8 v72, v12, v72, 0
	ds_read2_b32 v[78:79], v78 offset1:1
	v_dot4_i32_i8 v72, v13, v73, v72
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v72, v2, v74, v72
	v_dot4_i32_i8 v72, v3, v75, v72
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v72, v4, v76, v72
	v_dot4_i32_i8 v72, v5, v77, v72
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v72, v6, v78, v72
	v_dot4_i32_i8 v81, v9, v73, v81
	v_dot4_i32_i8 v72, v7, v79, v72
	v_dot4_i32_i8 v81, v10, v74, v81
	v_cvt_f32_i32_e32 v72, v72
	v_dot4_i32_i8 v81, v11, v75, v81
	v_dot4_i32_i8 v81, v14, v76, v81
	v_dot4_i32_i8 v81, v15, v77, v81
	v_mul_f32_e32 v73, v80, v70
	v_dot4_i32_i8 v81, v16, v78, v81
	v_fmac_f32_e32 v29, v73, v72
	v_add_u32_e32 v72, 0x1f80, v69
	v_dot4_i32_i8 v81, v17, v79, v81
	v_mul_f32_e32 v82, v80, v71
	ds_read_b32 v80, v67 offset:8064
	ds_read2_b32 v[72:73], v72 offset1:1
	v_cvt_f32_i32_e32 v81, v81
	v_add_u32_e32 v74, 0x1f88, v69
	ds_read2_b32 v[74:75], v74 offset1:1
	v_add_u32_e32 v76, 0x1f90, v69
	ds_read2_b32 v[76:77], v76 offset1:1
	v_fmac_f32_e32 v30, v82, v81
	v_add_u32_e32 v78, 0x1f98, v69
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v8, v72, 0
	v_dot4_i32_i8 v72, v12, v72, 0
	ds_read2_b32 v[78:79], v78 offset1:1
	v_dot4_i32_i8 v72, v13, v73, v72
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v72, v2, v74, v72
	v_dot4_i32_i8 v72, v3, v75, v72
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v72, v4, v76, v72
	v_dot4_i32_i8 v72, v5, v77, v72
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v72, v6, v78, v72
	v_dot4_i32_i8 v81, v9, v73, v81
	v_dot4_i32_i8 v72, v7, v79, v72
	v_dot4_i32_i8 v81, v10, v74, v81
	v_cvt_f32_i32_e32 v72, v72
	v_dot4_i32_i8 v81, v11, v75, v81
	v_dot4_i32_i8 v81, v14, v76, v81
	v_dot4_i32_i8 v81, v15, v77, v81
	v_mul_f32_e32 v73, v80, v70
	v_dot4_i32_i8 v81, v16, v78, v81
	v_fmac_f32_e32 v27, v73, v72
	v_add_u32_e32 v72, 0x2400, v69
	v_dot4_i32_i8 v81, v17, v79, v81
	v_mul_f32_e32 v82, v80, v71
	ds_read_b32 v80, v67 offset:9216
	ds_read2_b32 v[72:73], v72 offset1:1
	v_cvt_f32_i32_e32 v81, v81
	v_add_u32_e32 v74, 0x2408, v69
	ds_read2_b32 v[74:75], v74 offset1:1
	v_add_u32_e32 v76, 0x2410, v69
	ds_read2_b32 v[76:77], v76 offset1:1
	v_fmac_f32_e32 v28, v82, v81
	v_add_u32_e32 v78, 0x2418, v69
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v8, v72, 0
	v_dot4_i32_i8 v72, v12, v72, 0
	ds_read2_b32 v[78:79], v78 offset1:1
	v_dot4_i32_i8 v72, v13, v73, v72
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v72, v2, v74, v72
	v_dot4_i32_i8 v72, v3, v75, v72
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v72, v4, v76, v72
	v_dot4_i32_i8 v72, v5, v77, v72
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v72, v6, v78, v72
	v_dot4_i32_i8 v81, v9, v73, v81
	v_dot4_i32_i8 v72, v7, v79, v72
	v_dot4_i32_i8 v81, v10, v74, v81
	v_cvt_f32_i32_e32 v72, v72
	v_dot4_i32_i8 v81, v11, v75, v81
	v_dot4_i32_i8 v81, v14, v76, v81
	v_dot4_i32_i8 v81, v15, v77, v81
	v_mul_f32_e32 v73, v80, v70
	v_dot4_i32_i8 v81, v16, v78, v81
	v_fmac_f32_e32 v22, v73, v72
	v_add_u32_e32 v72, 0x2880, v69
	v_dot4_i32_i8 v81, v17, v79, v81
	v_mul_f32_e32 v82, v80, v71
	ds_read_b32 v80, v67 offset:10368
	ds_read2_b32 v[72:73], v72 offset1:1
	v_cvt_f32_i32_e32 v81, v81
	v_add_u32_e32 v74, 0x2888, v69
	ds_read2_b32 v[74:75], v74 offset1:1
	v_add_u32_e32 v76, 0x2890, v69
	ds_read2_b32 v[76:77], v76 offset1:1
	v_fmac_f32_e32 v23, v82, v81
	v_add_u32_e32 v78, 0x2898, v69
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v8, v72, 0
	v_dot4_i32_i8 v72, v12, v72, 0
	ds_read2_b32 v[78:79], v78 offset1:1
	v_dot4_i32_i8 v72, v13, v73, v72
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v72, v2, v74, v72
	v_dot4_i32_i8 v72, v3, v75, v72
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v72, v4, v76, v72
	v_dot4_i32_i8 v72, v5, v77, v72
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v72, v6, v78, v72
	v_dot4_i32_i8 v81, v9, v73, v81
	v_dot4_i32_i8 v72, v7, v79, v72
	v_dot4_i32_i8 v81, v10, v74, v81
	v_cvt_f32_i32_e32 v72, v72
	v_dot4_i32_i8 v81, v11, v75, v81
	v_dot4_i32_i8 v81, v14, v76, v81
	v_dot4_i32_i8 v81, v15, v77, v81
	v_mul_f32_e32 v73, v80, v70
	v_dot4_i32_i8 v81, v16, v78, v81
	v_fmac_f32_e32 v20, v73, v72
	v_add_u32_e32 v72, 0x2d00, v69
	v_dot4_i32_i8 v81, v17, v79, v81
	v_mul_f32_e32 v82, v80, v71
	ds_read_b32 v80, v67 offset:11520
	ds_read2_b32 v[72:73], v72 offset1:1
	v_cvt_f32_i32_e32 v81, v81
	v_add_u32_e32 v74, 0x2d08, v69
	ds_read2_b32 v[74:75], v74 offset1:1
	v_add_u32_e32 v76, 0x2d10, v69
	ds_read2_b32 v[76:77], v76 offset1:1
	v_fmac_f32_e32 v21, v82, v81
	v_add_u32_e32 v78, 0x2d18, v69
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v8, v72, 0
	v_dot4_i32_i8 v72, v12, v72, 0
	ds_read2_b32 v[78:79], v78 offset1:1
	v_dot4_i32_i8 v72, v13, v73, v72
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v72, v2, v74, v72
	v_dot4_i32_i8 v72, v3, v75, v72
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v72, v4, v76, v72
	v_dot4_i32_i8 v72, v5, v77, v72
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v72, v6, v78, v72
	v_dot4_i32_i8 v72, v7, v79, v72
	v_cvt_f32_i32_e32 v72, v72
	v_dot4_i32_i8 v81, v9, v73, v81
	v_mul_f32_e32 v73, v80, v70
	v_mul_f32_e32 v82, v80, v71
	v_fmac_f32_e32 v18, v73, v72
	v_add_u32_e32 v72, 0x3180, v69
	ds_read_b32 v80, v67 offset:12672
	ds_read2_b32 v[72:73], v72 offset1:1
	v_dot4_i32_i8 v81, v10, v74, v81
	v_add_u32_e32 v74, 0x3188, v69
	v_dot4_i32_i8 v81, v11, v75, v81
	ds_read2_b32 v[74:75], v74 offset1:1
	v_dot4_i32_i8 v81, v14, v76, v81
	v_add_u32_e32 v76, 0x3190, v69
	v_dot4_i32_i8 v81, v15, v77, v81
	ds_read2_b32 v[76:77], v76 offset1:1
	v_dot4_i32_i8 v81, v16, v78, v81
	v_add_u32_e32 v78, 0x3198, v69
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v8, v8, v72, 0
	v_dot4_i32_i8 v81, v17, v79, v81
	ds_read2_b32 v[78:79], v78 offset1:1
	v_dot4_i32_i8 v8, v9, v73, v8
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v8, v10, v74, v8
	v_dot4_i32_i8 v8, v11, v75, v8
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v8, v14, v76, v8
	v_dot4_i32_i8 v8, v15, v77, v8
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v8, v16, v78, v8
	v_dot4_i32_i8 v8, v17, v79, v8
	v_cvt_f32_i32_e32 v8, v8
	v_mul_f32_e32 v9, v80, v71
	v_cvt_f32_i32_e32 v81, v81
	v_add_u32_e32 v69, 32, v69
	v_fmac_f32_e32 v26, v9, v8
	v_dot4_i32_i8 v8, v12, v72, 0
	v_dot4_i32_i8 v8, v13, v73, v8
	v_dot4_i32_i8 v2, v2, v74, v8
	v_dot4_i32_i8 v2, v3, v75, v2
	v_dot4_i32_i8 v2, v4, v76, v2
	v_dot4_i32_i8 v2, v5, v77, v2
	v_dot4_i32_i8 v2, v6, v78, v2
	v_dot4_i32_i8 v2, v7, v79, v2
	v_cvt_f32_i32_e32 v2, v2
	v_mul_f32_e32 v3, v80, v70
	v_fmac_f32_e32 v19, v82, v81
	v_add_u32_e32 v67, 4, v67
	v_fmac_f32_e32 v25, v3, v2
	s_cbranch_scc1 .LBB44_93
; %bb.94:                               ; %_ZL22vec_dot_q8_0_q8_1_dp4aILi96ELi128ELi8EEvPKiS1_Pfi.exit.i932
                                        ;   in Loop: Header=BB44_92 Depth=1
	s_add_u32 s8, s8, s4
	s_addc_u32 s9, s9, s5
	v_mov_b32_e32 v2, s9
	v_add_co_u32_e32 v6, vcc, s8, v24
	v_addc_co_u32_e32 v7, vcc, 0, v2, vcc
	v_add_co_u32_e32 v2, vcc, s1, v6
	v_addc_co_u32_e32 v3, vcc, 0, v7, vcc
	v_add_co_u32_e32 v4, vcc, 0x2000, v6
	v_addc_co_u32_e32 v5, vcc, 0, v7, vcc
	v_add_co_u32_e32 v6, vcc, 0x3000, v6
	s_barrier
	v_addc_co_u32_e32 v7, vcc, 0, v7, vcc
	global_load_dword v8, v24, s[8:9]
	global_load_dword v9, v24, s[8:9] offset:2048
	global_load_dword v10, v[2:3], off
	global_load_dword v11, v[2:3], off offset:2048
	global_load_dword v12, v[4:5], off
	global_load_dword v13, v[4:5], off offset:2048
	global_load_dword v14, v[6:7], off
	s_mov_b32 s8, -8
	v_mov_b32_e32 v65, v55
	v_mov_b32_e32 v66, v60
	v_mov_b32_e32 v67, v59
	v_mov_b32_e32 v68, v58
	s_waitcnt vmcnt(5)
	ds_write2st64_b32 v64, v8, v9 offset0:1 offset1:9
	s_waitcnt vmcnt(3)
	ds_write2st64_b32 v64, v10, v11 offset0:17 offset1:25
	s_waitcnt vmcnt(1)
	ds_write2st64_b32 v64, v12, v13 offset0:33 offset1:41
	s_waitcnt vmcnt(0)
	ds_write_b32 v52, v14 offset:12672
	v_mov_b32_e32 v64, v53
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB44_95:                              ;   Parent Loop BB44_92 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ds_read_b32 v73, v65
	ds_read2_b32 v[2:3], v64 offset1:1
	ds_read2_b32 v[4:5], v64 offset0:2 offset1:3
	ds_read2_b32 v[6:7], v64 offset0:4 offset1:5
	ds_read2_b32 v[71:72], v64 offset0:6 offset1:7
	ds_read_b32 v70, v66
	ds_read2_b32 v[8:9], v68 offset1:1
	v_add_u32_e32 v75, 0x490, v64
	v_add_u32_e32 v77, 0x498, v64
	s_add_i32 s8, s8, 8
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v13, v70, v73
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v10, v8, v2, 0
	v_dot4_i32_i8 v12, v9, v3, v10
	ds_read2_b32 v[10:11], v68 offset0:2 offset1:3
	ds_read2_b32 v[14:15], v68 offset0:4 offset1:5
	ds_read2_b32 v[16:17], v68 offset0:6 offset1:7
	v_add_u32_e32 v66, 4, v66
	s_cmp_lt_u32 s8, 24
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v12, v10, v4, v12
	v_dot4_i32_i8 v12, v11, v5, v12
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v12, v14, v6, v12
	v_dot4_i32_i8 v12, v15, v7, v12
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v12, v16, v71, v12
	v_dot4_i32_i8 v12, v17, v72, v12
	v_cvt_f32_i32_e32 v12, v12
	ds_read2_b32 v[75:76], v75 offset1:1
	ds_read2_b32 v[77:78], v77 offset1:1
	v_fmac_f32_e32 v42, v13, v12
	v_add_u32_e32 v12, 0x4100, v68
	ds_read_b32 v69, v67
	ds_read2_b32 v[12:13], v12 offset1:1
	v_add_u32_e32 v67, 4, v67
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v2, v12, v2, 0
	v_dot4_i32_i8 v74, v13, v3, v2
	v_add_u32_e32 v2, 0x4108, v68
	ds_read2_b32 v[2:3], v2 offset1:1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v4, v2, v4, v74
	v_dot4_i32_i8 v74, v3, v5, v4
	v_add_u32_e32 v4, 0x4110, v68
	ds_read2_b32 v[4:5], v4 offset1:1
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v6, v4, v6, v74
	v_dot4_i32_i8 v74, v5, v7, v6
	v_add_u32_e32 v6, 0x4118, v68
	ds_read2_b32 v[6:7], v6 offset1:1
	v_add_u32_e32 v68, 32, v68
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v71, v6, v71, v74
	v_dot4_i32_i8 v71, v7, v72, v71
	v_cvt_f32_i32_e32 v71, v71
	v_mul_f32_e32 v72, v69, v73
	v_add_u32_e32 v73, 0x488, v64
	ds_read2_b32 v[73:74], v73 offset1:1
	v_fmac_f32_e32 v41, v72, v71
	v_add_u32_e32 v71, 0x480, v64
	ds_read_b32 v79, v65 offset:1152
	ds_read2_b32 v[71:72], v71 offset1:1
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v81, v79, v70
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v80, v8, v71, 0
	v_dot4_i32_i8 v71, v12, v71, 0
	v_dot4_i32_i8 v71, v13, v72, v71
	v_dot4_i32_i8 v71, v2, v73, v71
	v_dot4_i32_i8 v71, v3, v74, v71
	v_dot4_i32_i8 v71, v4, v75, v71
	v_dot4_i32_i8 v71, v5, v76, v71
	v_dot4_i32_i8 v71, v6, v77, v71
	v_dot4_i32_i8 v80, v9, v72, v80
	v_dot4_i32_i8 v71, v7, v78, v71
	v_dot4_i32_i8 v80, v10, v73, v80
	v_cvt_f32_i32_e32 v71, v71
	v_dot4_i32_i8 v80, v11, v74, v80
	v_dot4_i32_i8 v80, v14, v75, v80
	v_dot4_i32_i8 v80, v15, v76, v80
	v_mul_f32_e32 v72, v79, v69
	v_dot4_i32_i8 v80, v16, v77, v80
	v_fmac_f32_e32 v39, v72, v71
	v_add_u32_e32 v71, 0x900, v64
	v_dot4_i32_i8 v80, v17, v78, v80
	ds_read_b32 v79, v65 offset:2304
	ds_read2_b32 v[71:72], v71 offset1:1
	v_cvt_f32_i32_e32 v80, v80
	v_add_u32_e32 v73, 0x908, v64
	ds_read2_b32 v[73:74], v73 offset1:1
	v_add_u32_e32 v75, 0x910, v64
	ds_read2_b32 v[75:76], v75 offset1:1
	v_fmac_f32_e32 v40, v81, v80
	v_add_u32_e32 v77, 0x918, v64
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v80, v8, v71, 0
	v_dot4_i32_i8 v71, v12, v71, 0
	ds_read2_b32 v[77:78], v77 offset1:1
	v_dot4_i32_i8 v71, v13, v72, v71
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v71, v2, v73, v71
	v_dot4_i32_i8 v71, v3, v74, v71
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v71, v4, v75, v71
	v_dot4_i32_i8 v71, v5, v76, v71
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v71, v6, v77, v71
	v_dot4_i32_i8 v80, v9, v72, v80
	v_dot4_i32_i8 v71, v7, v78, v71
	v_dot4_i32_i8 v80, v10, v73, v80
	v_cvt_f32_i32_e32 v71, v71
	v_dot4_i32_i8 v80, v11, v74, v80
	v_dot4_i32_i8 v80, v14, v75, v80
	v_dot4_i32_i8 v80, v15, v76, v80
	v_mul_f32_e32 v72, v79, v69
	v_dot4_i32_i8 v80, v16, v77, v80
	v_fmac_f32_e32 v37, v72, v71
	v_add_u32_e32 v71, 0xd80, v64
	v_dot4_i32_i8 v80, v17, v78, v80
	v_mul_f32_e32 v81, v79, v70
	ds_read_b32 v79, v65 offset:3456
	ds_read2_b32 v[71:72], v71 offset1:1
	v_cvt_f32_i32_e32 v80, v80
	v_add_u32_e32 v73, 0xd88, v64
	ds_read2_b32 v[73:74], v73 offset1:1
	v_add_u32_e32 v75, 0xd90, v64
	ds_read2_b32 v[75:76], v75 offset1:1
	v_fmac_f32_e32 v38, v81, v80
	v_add_u32_e32 v77, 0xd98, v64
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v80, v8, v71, 0
	v_dot4_i32_i8 v71, v12, v71, 0
	ds_read2_b32 v[77:78], v77 offset1:1
	v_dot4_i32_i8 v71, v13, v72, v71
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v71, v2, v73, v71
	v_dot4_i32_i8 v71, v3, v74, v71
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v71, v4, v75, v71
	v_dot4_i32_i8 v71, v5, v76, v71
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v71, v6, v77, v71
	v_dot4_i32_i8 v80, v9, v72, v80
	v_dot4_i32_i8 v71, v7, v78, v71
	v_dot4_i32_i8 v80, v10, v73, v80
	v_cvt_f32_i32_e32 v71, v71
	v_dot4_i32_i8 v80, v11, v74, v80
	v_dot4_i32_i8 v80, v14, v75, v80
	v_dot4_i32_i8 v80, v15, v76, v80
	v_mul_f32_e32 v72, v79, v69
	v_dot4_i32_i8 v80, v16, v77, v80
	v_fmac_f32_e32 v35, v72, v71
	v_add_u32_e32 v71, 0x1200, v64
	v_dot4_i32_i8 v80, v17, v78, v80
	v_mul_f32_e32 v81, v79, v70
	ds_read_b32 v79, v65 offset:4608
	ds_read2_b32 v[71:72], v71 offset1:1
	v_cvt_f32_i32_e32 v80, v80
	v_add_u32_e32 v73, 0x1208, v64
	ds_read2_b32 v[73:74], v73 offset1:1
	v_add_u32_e32 v75, 0x1210, v64
	ds_read2_b32 v[75:76], v75 offset1:1
	v_fmac_f32_e32 v36, v81, v80
	v_add_u32_e32 v77, 0x1218, v64
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v80, v8, v71, 0
	v_dot4_i32_i8 v71, v12, v71, 0
	ds_read2_b32 v[77:78], v77 offset1:1
	v_dot4_i32_i8 v71, v13, v72, v71
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v71, v2, v73, v71
	v_dot4_i32_i8 v71, v3, v74, v71
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v71, v4, v75, v71
	v_dot4_i32_i8 v71, v5, v76, v71
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v71, v6, v77, v71
	v_dot4_i32_i8 v80, v9, v72, v80
	v_dot4_i32_i8 v71, v7, v78, v71
	v_dot4_i32_i8 v80, v10, v73, v80
	v_cvt_f32_i32_e32 v71, v71
	v_dot4_i32_i8 v80, v11, v74, v80
	v_dot4_i32_i8 v80, v14, v75, v80
	v_dot4_i32_i8 v80, v15, v76, v80
	v_mul_f32_e32 v72, v79, v69
	v_dot4_i32_i8 v80, v16, v77, v80
	v_fmac_f32_e32 v33, v72, v71
	v_add_u32_e32 v71, 0x1680, v64
	v_dot4_i32_i8 v80, v17, v78, v80
	v_mul_f32_e32 v81, v79, v70
	ds_read_b32 v79, v65 offset:5760
	ds_read2_b32 v[71:72], v71 offset1:1
	v_cvt_f32_i32_e32 v80, v80
	v_add_u32_e32 v73, 0x1688, v64
	ds_read2_b32 v[73:74], v73 offset1:1
	v_add_u32_e32 v75, 0x1690, v64
	ds_read2_b32 v[75:76], v75 offset1:1
	v_fmac_f32_e32 v34, v81, v80
	v_add_u32_e32 v77, 0x1698, v64
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v80, v8, v71, 0
	v_dot4_i32_i8 v71, v12, v71, 0
	ds_read2_b32 v[77:78], v77 offset1:1
	v_dot4_i32_i8 v71, v13, v72, v71
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v71, v2, v73, v71
	v_dot4_i32_i8 v71, v3, v74, v71
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v71, v4, v75, v71
	v_dot4_i32_i8 v71, v5, v76, v71
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v71, v6, v77, v71
	v_dot4_i32_i8 v80, v9, v72, v80
	v_dot4_i32_i8 v71, v7, v78, v71
	v_dot4_i32_i8 v80, v10, v73, v80
	v_cvt_f32_i32_e32 v71, v71
	v_dot4_i32_i8 v80, v11, v74, v80
	v_dot4_i32_i8 v80, v14, v75, v80
	v_dot4_i32_i8 v80, v15, v76, v80
	v_mul_f32_e32 v72, v79, v69
	v_dot4_i32_i8 v80, v16, v77, v80
	v_fmac_f32_e32 v31, v72, v71
	v_add_u32_e32 v71, 0x1b00, v64
	v_dot4_i32_i8 v80, v17, v78, v80
	v_mul_f32_e32 v81, v79, v70
	ds_read_b32 v79, v65 offset:6912
	ds_read2_b32 v[71:72], v71 offset1:1
	v_cvt_f32_i32_e32 v80, v80
	v_add_u32_e32 v73, 0x1b08, v64
	ds_read2_b32 v[73:74], v73 offset1:1
	v_add_u32_e32 v75, 0x1b10, v64
	ds_read2_b32 v[75:76], v75 offset1:1
	v_fmac_f32_e32 v32, v81, v80
	v_add_u32_e32 v77, 0x1b18, v64
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v80, v8, v71, 0
	v_dot4_i32_i8 v71, v12, v71, 0
	ds_read2_b32 v[77:78], v77 offset1:1
	v_dot4_i32_i8 v71, v13, v72, v71
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v71, v2, v73, v71
	v_dot4_i32_i8 v71, v3, v74, v71
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v71, v4, v75, v71
	v_dot4_i32_i8 v71, v5, v76, v71
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v71, v6, v77, v71
	v_dot4_i32_i8 v80, v9, v72, v80
	v_dot4_i32_i8 v71, v7, v78, v71
	v_dot4_i32_i8 v80, v10, v73, v80
	v_cvt_f32_i32_e32 v71, v71
	v_dot4_i32_i8 v80, v11, v74, v80
	v_dot4_i32_i8 v80, v14, v75, v80
	v_dot4_i32_i8 v80, v15, v76, v80
	v_mul_f32_e32 v72, v79, v69
	v_dot4_i32_i8 v80, v16, v77, v80
	v_fmac_f32_e32 v29, v72, v71
	v_add_u32_e32 v71, 0x1f80, v64
	v_dot4_i32_i8 v80, v17, v78, v80
	v_mul_f32_e32 v81, v79, v70
	ds_read_b32 v79, v65 offset:8064
	ds_read2_b32 v[71:72], v71 offset1:1
	v_cvt_f32_i32_e32 v80, v80
	v_add_u32_e32 v73, 0x1f88, v64
	ds_read2_b32 v[73:74], v73 offset1:1
	v_add_u32_e32 v75, 0x1f90, v64
	ds_read2_b32 v[75:76], v75 offset1:1
	v_fmac_f32_e32 v30, v81, v80
	v_add_u32_e32 v77, 0x1f98, v64
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v80, v8, v71, 0
	v_dot4_i32_i8 v71, v12, v71, 0
	ds_read2_b32 v[77:78], v77 offset1:1
	v_dot4_i32_i8 v71, v13, v72, v71
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v71, v2, v73, v71
	v_dot4_i32_i8 v71, v3, v74, v71
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v71, v4, v75, v71
	v_dot4_i32_i8 v71, v5, v76, v71
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v71, v6, v77, v71
	v_dot4_i32_i8 v80, v9, v72, v80
	v_dot4_i32_i8 v71, v7, v78, v71
	v_dot4_i32_i8 v80, v10, v73, v80
	v_cvt_f32_i32_e32 v71, v71
	v_dot4_i32_i8 v80, v11, v74, v80
	v_dot4_i32_i8 v80, v14, v75, v80
	v_dot4_i32_i8 v80, v15, v76, v80
	v_mul_f32_e32 v72, v79, v69
	v_dot4_i32_i8 v80, v16, v77, v80
	v_fmac_f32_e32 v27, v72, v71
	v_add_u32_e32 v71, 0x2400, v64
	v_dot4_i32_i8 v80, v17, v78, v80
	v_mul_f32_e32 v81, v79, v70
	ds_read_b32 v79, v65 offset:9216
	ds_read2_b32 v[71:72], v71 offset1:1
	v_cvt_f32_i32_e32 v80, v80
	v_add_u32_e32 v73, 0x2408, v64
	ds_read2_b32 v[73:74], v73 offset1:1
	v_add_u32_e32 v75, 0x2410, v64
	ds_read2_b32 v[75:76], v75 offset1:1
	v_fmac_f32_e32 v28, v81, v80
	v_add_u32_e32 v77, 0x2418, v64
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v80, v8, v71, 0
	v_dot4_i32_i8 v71, v12, v71, 0
	ds_read2_b32 v[77:78], v77 offset1:1
	v_dot4_i32_i8 v71, v13, v72, v71
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v71, v2, v73, v71
	v_dot4_i32_i8 v71, v3, v74, v71
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v71, v4, v75, v71
	v_dot4_i32_i8 v71, v5, v76, v71
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v71, v6, v77, v71
	v_dot4_i32_i8 v80, v9, v72, v80
	v_dot4_i32_i8 v71, v7, v78, v71
	v_dot4_i32_i8 v80, v10, v73, v80
	v_cvt_f32_i32_e32 v71, v71
	v_dot4_i32_i8 v80, v11, v74, v80
	v_dot4_i32_i8 v80, v14, v75, v80
	v_dot4_i32_i8 v80, v15, v76, v80
	v_mul_f32_e32 v72, v79, v69
	v_dot4_i32_i8 v80, v16, v77, v80
	v_fmac_f32_e32 v22, v72, v71
	v_add_u32_e32 v71, 0x2880, v64
	v_dot4_i32_i8 v80, v17, v78, v80
	v_mul_f32_e32 v81, v79, v70
	ds_read_b32 v79, v65 offset:10368
	ds_read2_b32 v[71:72], v71 offset1:1
	v_cvt_f32_i32_e32 v80, v80
	v_add_u32_e32 v73, 0x2888, v64
	ds_read2_b32 v[73:74], v73 offset1:1
	v_add_u32_e32 v75, 0x2890, v64
	ds_read2_b32 v[75:76], v75 offset1:1
	v_fmac_f32_e32 v23, v81, v80
	v_add_u32_e32 v77, 0x2898, v64
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v80, v8, v71, 0
	v_dot4_i32_i8 v71, v12, v71, 0
	ds_read2_b32 v[77:78], v77 offset1:1
	v_dot4_i32_i8 v71, v13, v72, v71
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v71, v2, v73, v71
	v_dot4_i32_i8 v71, v3, v74, v71
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v71, v4, v75, v71
	v_dot4_i32_i8 v71, v5, v76, v71
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v71, v6, v77, v71
	v_dot4_i32_i8 v80, v9, v72, v80
	v_dot4_i32_i8 v71, v7, v78, v71
	v_dot4_i32_i8 v80, v10, v73, v80
	v_cvt_f32_i32_e32 v71, v71
	v_dot4_i32_i8 v80, v11, v74, v80
	v_dot4_i32_i8 v80, v14, v75, v80
	v_dot4_i32_i8 v80, v15, v76, v80
	v_mul_f32_e32 v72, v79, v69
	v_dot4_i32_i8 v80, v16, v77, v80
	v_fmac_f32_e32 v20, v72, v71
	v_add_u32_e32 v71, 0x2d00, v64
	v_dot4_i32_i8 v80, v17, v78, v80
	v_mul_f32_e32 v81, v79, v70
	ds_read_b32 v79, v65 offset:11520
	ds_read2_b32 v[71:72], v71 offset1:1
	v_cvt_f32_i32_e32 v80, v80
	v_add_u32_e32 v73, 0x2d08, v64
	ds_read2_b32 v[73:74], v73 offset1:1
	v_add_u32_e32 v75, 0x2d10, v64
	ds_read2_b32 v[75:76], v75 offset1:1
	v_fmac_f32_e32 v21, v81, v80
	v_add_u32_e32 v77, 0x2d18, v64
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v80, v8, v71, 0
	v_dot4_i32_i8 v71, v12, v71, 0
	ds_read2_b32 v[77:78], v77 offset1:1
	v_dot4_i32_i8 v71, v13, v72, v71
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v71, v2, v73, v71
	v_dot4_i32_i8 v71, v3, v74, v71
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v71, v4, v75, v71
	v_dot4_i32_i8 v71, v5, v76, v71
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v71, v6, v77, v71
	v_dot4_i32_i8 v71, v7, v78, v71
	v_cvt_f32_i32_e32 v71, v71
	v_dot4_i32_i8 v80, v9, v72, v80
	v_mul_f32_e32 v72, v79, v69
	v_mul_f32_e32 v81, v79, v70
	v_fmac_f32_e32 v18, v72, v71
	v_add_u32_e32 v71, 0x3180, v64
	ds_read_b32 v79, v65 offset:12672
	ds_read2_b32 v[71:72], v71 offset1:1
	v_dot4_i32_i8 v80, v10, v73, v80
	v_add_u32_e32 v73, 0x3188, v64
	v_dot4_i32_i8 v80, v11, v74, v80
	ds_read2_b32 v[73:74], v73 offset1:1
	v_dot4_i32_i8 v80, v14, v75, v80
	v_add_u32_e32 v75, 0x3190, v64
	v_dot4_i32_i8 v80, v15, v76, v80
	ds_read2_b32 v[75:76], v75 offset1:1
	v_dot4_i32_i8 v80, v16, v77, v80
	v_add_u32_e32 v77, 0x3198, v64
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v8, v8, v71, 0
	v_dot4_i32_i8 v80, v17, v78, v80
	ds_read2_b32 v[77:78], v77 offset1:1
	v_dot4_i32_i8 v8, v9, v72, v8
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v8, v10, v73, v8
	v_dot4_i32_i8 v8, v11, v74, v8
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v8, v14, v75, v8
	v_dot4_i32_i8 v8, v15, v76, v8
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v8, v16, v77, v8
	v_dot4_i32_i8 v8, v17, v78, v8
	v_cvt_f32_i32_e32 v8, v8
	v_mul_f32_e32 v9, v79, v70
	v_cvt_f32_i32_e32 v80, v80
	v_add_u32_e32 v64, 32, v64
	v_fmac_f32_e32 v26, v9, v8
	v_dot4_i32_i8 v8, v12, v71, 0
	v_dot4_i32_i8 v8, v13, v72, v8
	v_dot4_i32_i8 v2, v2, v73, v8
	v_dot4_i32_i8 v2, v3, v74, v2
	v_dot4_i32_i8 v2, v4, v75, v2
	v_dot4_i32_i8 v2, v5, v76, v2
	v_dot4_i32_i8 v2, v6, v77, v2
	v_dot4_i32_i8 v2, v7, v78, v2
	v_cvt_f32_i32_e32 v2, v2
	v_mul_f32_e32 v3, v79, v69
	v_fmac_f32_e32 v19, v81, v80
	v_add_u32_e32 v65, 4, v65
	v_fmac_f32_e32 v25, v3, v2
	s_cbranch_scc1 .LBB44_95
; %bb.96:                               ; %_ZL22vec_dot_q8_0_q8_1_dp4aILi96ELi128ELi8EEvPKiS1_Pfi.exit101.i
                                        ;   in Loop: Header=BB44_92 Depth=1
	s_add_i32 s7, s7, 8
	s_cmp_ge_i32 s7, s23
	s_barrier
	s_cbranch_scc0 .LBB44_92
.LBB44_97:                              ; %._crit_edge.i746
	s_movk_i32 s0, 0x61
	v_cmp_gt_u32_e32 vcc, s0, v1
                                        ; implicit-def: $sgpr8_sgpr9
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB44_121
; %bb.98:                               ; %.preheader.i.i774
	v_lshl_add_u32 v3, v1, 2, 0
	ds_read_b32 v2, v3
	s_mul_i32 s4, s6, 0x3000
	s_mov_b32 s5, 0
	s_lshl_b64 s[4:5], s[4:5], 2
	s_add_u32 s8, s30, s4
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u32 v4, v2, 7, v0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	s_addc_u32 s9, s31, s5
	v_mov_b32_e32 v2, s9
	v_add_co_u32_e32 v4, vcc, s8, v4
	v_addc_co_u32_e32 v5, vcc, v2, v5, vcc
	s_movk_i32 s4, 0x59
	v_cmp_gt_u32_e32 vcc, s4, v1
	s_mov_b64 s[6:7], s[2:3]
	global_store_dword v[4:5], v42, off
	global_store_dword v[4:5], v41, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB44_120
; %bb.99:                               ; %.preheader.1.i.i778
	ds_read_b32 v2, v3 offset:32
	v_mov_b32_e32 v6, s9
	s_movk_i32 s6, 0x51
	s_mov_b64 s[10:11], s[2:3]
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u32 v4, v2, 7, v0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
                                        ; implicit-def: $vgpr2
	v_add_co_u32_e32 v4, vcc, s8, v4
	v_addc_co_u32_e32 v5, vcc, v6, v5, vcc
	v_cmp_gt_u32_e32 vcc, s6, v1
	global_store_dword v[4:5], v40, off
	global_store_dword v[4:5], v39, off offset:256
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB44_119
; %bb.100:                              ; %.preheader.2.i.i782
	ds_read_b32 v2, v3 offset:64
	v_mov_b32_e32 v6, s9
	s_movk_i32 s10, 0x49
	s_mov_b64 s[12:13], s[2:3]
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u32 v4, v2, 7, v0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
                                        ; implicit-def: $vgpr2
	v_add_co_u32_e32 v4, vcc, s8, v4
	v_addc_co_u32_e32 v5, vcc, v6, v5, vcc
	v_cmp_gt_u32_e32 vcc, s10, v1
	global_store_dword v[4:5], v38, off
	global_store_dword v[4:5], v37, off offset:256
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB44_118
; %bb.101:                              ; %.preheader.3.i.i786
	ds_read_b32 v2, v3 offset:96
	v_mov_b32_e32 v6, s9
	s_movk_i32 s12, 0x41
	s_mov_b64 s[14:15], s[2:3]
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u32 v4, v2, 7, v0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
                                        ; implicit-def: $vgpr2
	v_add_co_u32_e32 v4, vcc, s8, v4
	v_addc_co_u32_e32 v5, vcc, v6, v5, vcc
	v_cmp_gt_u32_e32 vcc, s12, v1
	global_store_dword v[4:5], v36, off
	global_store_dword v[4:5], v35, off offset:256
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB44_117
; %bb.102:                              ; %.preheader.4.i.i790
	ds_read_b32 v2, v3 offset:128
	v_mov_b32_e32 v6, s9
	s_mov_b64 s[16:17], s[2:3]
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u32 v4, v2, 7, v0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
                                        ; implicit-def: $vgpr2
	v_add_co_u32_e32 v4, vcc, s8, v4
	v_addc_co_u32_e32 v5, vcc, v6, v5, vcc
	v_cmp_gt_u32_e32 vcc, 57, v1
	global_store_dword v[4:5], v34, off
	global_store_dword v[4:5], v33, off offset:256
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB44_116
; %bb.103:                              ; %.preheader.5.i.i794
	ds_read_b32 v2, v3 offset:160
	v_mov_b32_e32 v6, s9
	s_mov_b64 s[18:19], s[2:3]
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u32 v4, v2, 7, v0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
                                        ; implicit-def: $vgpr2
	v_add_co_u32_e32 v4, vcc, s8, v4
	v_addc_co_u32_e32 v5, vcc, v6, v5, vcc
	v_cmp_gt_u32_e32 vcc, 49, v1
	global_store_dword v[4:5], v32, off
	global_store_dword v[4:5], v31, off offset:256
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB44_115
; %bb.104:                              ; %.preheader.6.i.i798
	ds_read_b32 v2, v3 offset:192
	v_mov_b32_e32 v6, s9
	s_mov_b64 s[20:21], s[2:3]
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u32 v4, v2, 7, v0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
                                        ; implicit-def: $vgpr2
	v_add_co_u32_e32 v4, vcc, s8, v4
	v_addc_co_u32_e32 v5, vcc, v6, v5, vcc
	v_cmp_gt_u32_e32 vcc, 41, v1
	global_store_dword v[4:5], v30, off
	global_store_dword v[4:5], v29, off offset:256
	s_and_saveexec_b64 s[18:19], vcc
	s_cbranch_execz .LBB44_114
; %bb.105:                              ; %.preheader.7.i.i802
	ds_read_b32 v2, v3 offset:224
	v_mov_b32_e32 v6, s9
	s_movk_i32 s20, 0x61
	s_mov_b64 s[22:23], s[2:3]
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u32 v4, v2, 7, v0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
	v_or_b32_e32 v2, 64, v1
	v_add_co_u32_e32 v4, vcc, s8, v4
	v_addc_co_u32_e32 v5, vcc, v6, v5, vcc
	v_cmp_gt_u32_e32 vcc, s20, v2
	global_store_dword v[4:5], v28, off
	global_store_dword v[4:5], v27, off offset:256
                                        ; implicit-def: $vgpr2
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB44_113
; %bb.106:                              ; %.preheader.8.i.i806
	ds_read_b32 v2, v3 offset:256
	v_mov_b32_e32 v6, s9
	s_mov_b64 s[24:25], s[2:3]
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u32 v4, v2, 7, v0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
                                        ; implicit-def: $vgpr2
	v_add_co_u32_e32 v4, vcc, s8, v4
	v_addc_co_u32_e32 v5, vcc, v6, v5, vcc
	v_cmp_gt_u32_e32 vcc, 25, v1
	global_store_dword v[4:5], v23, off
	global_store_dword v[4:5], v22, off offset:256
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB44_112
; %bb.107:                              ; %.preheader.9.i.i810
	ds_read_b32 v2, v3 offset:288
	v_mov_b32_e32 v6, s9
	s_mov_b64 s[26:27], s[2:3]
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u32 v4, v2, 7, v0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
                                        ; implicit-def: $vgpr2
	v_add_co_u32_e32 v4, vcc, s8, v4
	v_addc_co_u32_e32 v5, vcc, v6, v5, vcc
	v_cmp_gt_u32_e32 vcc, 17, v1
	global_store_dword v[4:5], v21, off
	global_store_dword v[4:5], v20, off offset:256
	s_and_saveexec_b64 s[24:25], vcc
	s_cbranch_execz .LBB44_111
; %bb.108:                              ; %.preheader.10.i.i814
	ds_read_b32 v2, v3 offset:320
	v_mov_b32_e32 v6, s9
	s_mov_b64 s[26:27], s[2:3]
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u32 v4, v2, 7, v0
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[4:5], 2, v[4:5]
                                        ; implicit-def: $vgpr2
	v_add_co_u32_e32 v4, vcc, s8, v4
	v_addc_co_u32_e32 v5, vcc, v6, v5, vcc
	v_cmp_gt_u32_e32 vcc, 9, v1
	global_store_dword v[4:5], v19, off
	global_store_dword v[4:5], v18, off offset:256
	s_and_saveexec_b64 s[28:29], vcc
	s_cbranch_execz .LBB44_110
; %bb.109:                              ; %.preheader.11.i.i818
	ds_read_b32 v1, v3 offset:352
	s_or_b64 s[26:27], s[2:3], exec
	s_waitcnt lgkmcnt(0)
	v_lshlrev_b32_e32 v2, 7, v1
.LBB44_110:                             ; %Flow3281
	s_or_b64 exec, exec, s[28:29]
	s_andn2_b64 s[28:29], s[2:3], exec
	s_and_b64 s[26:27], s[26:27], exec
	s_or_b64 s[26:27], s[28:29], s[26:27]
.LBB44_111:                             ; %Flow3280
	s_or_b64 exec, exec, s[24:25]
	s_andn2_b64 s[24:25], s[2:3], exec
	s_and_b64 s[26:27], s[26:27], exec
	s_or_b64 s[24:25], s[24:25], s[26:27]
.LBB44_112:                             ; %Flow3279
	s_or_b64 exec, exec, s[22:23]
	s_andn2_b64 s[22:23], s[2:3], exec
	s_and_b64 s[24:25], s[24:25], exec
	s_or_b64 s[22:23], s[22:23], s[24:25]
.LBB44_113:                             ; %Flow3278
	s_or_b64 exec, exec, s[20:21]
	s_andn2_b64 s[20:21], s[2:3], exec
	s_and_b64 s[22:23], s[22:23], exec
	s_or_b64 s[20:21], s[20:21], s[22:23]
.LBB44_114:                             ; %Flow3277
	s_or_b64 exec, exec, s[18:19]
	s_andn2_b64 s[18:19], s[2:3], exec
	s_and_b64 s[20:21], s[20:21], exec
	s_or_b64 s[18:19], s[18:19], s[20:21]
.LBB44_115:                             ; %Flow3276
	s_or_b64 exec, exec, s[16:17]
	s_andn2_b64 s[16:17], s[2:3], exec
	s_and_b64 s[18:19], s[18:19], exec
	s_or_b64 s[16:17], s[16:17], s[18:19]
.LBB44_116:                             ; %Flow3275
	s_or_b64 exec, exec, s[14:15]
	s_andn2_b64 s[14:15], s[2:3], exec
	s_and_b64 s[16:17], s[16:17], exec
	s_or_b64 s[14:15], s[14:15], s[16:17]
.LBB44_117:                             ; %Flow3274
	s_or_b64 exec, exec, s[12:13]
	s_andn2_b64 s[12:13], s[2:3], exec
	s_and_b64 s[14:15], s[14:15], exec
	s_or_b64 s[12:13], s[12:13], s[14:15]
.LBB44_118:                             ; %Flow3273
	s_or_b64 exec, exec, s[10:11]
	s_andn2_b64 s[10:11], s[2:3], exec
	s_and_b64 s[12:13], s[12:13], exec
	s_or_b64 s[10:11], s[10:11], s[12:13]
.LBB44_119:                             ; %Flow3272
	s_or_b64 exec, exec, s[6:7]
	s_andn2_b64 s[6:7], s[2:3], exec
	s_and_b64 s[10:11], s[10:11], exec
	s_or_b64 s[6:7], s[6:7], s[10:11]
.LBB44_120:                             ; %Flow3271
	s_or_b64 exec, exec, s[4:5]
	s_andn2_b64 s[2:3], s[2:3], exec
	s_and_b64 s[4:5], s[6:7], exec
	s_or_b64 s[2:3], s[2:3], s[4:5]
.LBB44_121:                             ; %Flow3270
	s_or_b64 exec, exec, s[0:1]
	s_branch .LBB44_126
.LBB44_122:
                                        ; implicit-def: $sgpr4_sgpr5
	v_cvt_f32_u32_e32 v3, s7
	s_branch .LBB44_47
.LBB44_123:
                                        ; implicit-def: $sgpr52_sgpr53
	s_branch .LBB44_50
.LBB44_124:
	s_mul_i32 s0, s20, s48
	s_mul_i32 s1, s4, s40
	s_add_i32 s5, s0, s1
	s_mov_b32 s16, 0
	s_cbranch_execnz .LBB44_90
.LBB44_125:
                                        ; implicit-def: $vgpr25
                                        ; implicit-def: $vgpr26
                                        ; implicit-def: $sgpr8_sgpr9
                                        ; implicit-def: $vgpr2
.LBB44_126:                             ; %Flow3265
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB44_128
; %bb.127:                              ; %_ZL22mul_mat_q_process_tileIL9ggml_type8ELi96ELb0ELb0EEvPKciPKiS4_PfS5_iiiiiii.exit
	s_endpgm
.LBB44_128:                             ; %_ZL22mul_mat_q_process_tileIL9ggml_type8ELi96ELb0ELb0EEvPKciPKiS4_PfS5_iiiiiii.exit.sink.split
	v_add_u32_e32 v0, v2, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	v_mov_b32_e32 v2, s9
	v_add_co_u32_e32 v0, vcc, s8, v0
	v_addc_co_u32_e32 v1, vcc, v2, v1, vcc
	global_store_dword v[0:1], v26, off
	global_store_dword v[0:1], v25, off offset:256
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZL9mul_mat_qIL9ggml_type8ELi96ELb0EEvPKcPKiS4_S4_PfS5_15HIP_vector_typeIjLj3EEiiiiiS7_S7_iiiS7_S7_iiiS7_b
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 424
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 98
		.amdhsa_next_free_sgpr 76
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZL9mul_mat_qIL9ggml_type8ELi96ELb0EEvPKcPKiS4_S4_PfS5_15HIP_vector_typeIjLj3EEiiiiiS7_S7_iiiS7_S7_iiiS7_b,"axG",@progbits,_ZL9mul_mat_qIL9ggml_type8ELi96ELb0EEvPKcPKiS4_S4_PfS5_15HIP_vector_typeIjLj3EEiiiiiS7_S7_iiiS7_S7_iiiS7_b,comdat
```

---

### 2.5 Repack `mmq_gemm_q8_0_repacked<false, 2, 4>` ISA body

Source: `repack-gcn-hip-amdgcn-amd-amdhsa-gfx906.s`  |  Mangled: `mmq_gemm_q8_0_repacked<false, 2, 4>`  |  Body lines: 1320  |  Total instructions: 1165

```asm
_ZL22mmq_gemm_q8_0_repackedILb0ELi2ELi4EEvPKhPK10block_q8_1PfjjjjPKiS7_S7_S7_jmj: ; @_ZL22mmq_gemm_q8_0_repackedILb0ELi2ELi4EEvPKhPK10block_q8_1PfjjjjPKiS7_S7_S7_jmj
; %bb.0:
	s_load_dwordx4 s[12:15], s[4:5], 0x18
	s_load_dwordx4 s[16:19], s[4:5], 0x0
	s_load_dwordx2 s[10:11], s[4:5], 0x10
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s15, s6, 6
	s_lshl_b32 s26, s7, 7
	s_lshr_b32 s12, s12, 5
	s_mov_b32 s27, 0
	s_cbranch_scc0 .LBB2_16
; %bb.1:                                ; %.preheader29.lr.ph.i
	s_load_dword s0, s[4:5], 0x6c
	s_add_i32 s1, s12, -1
	v_lshrrev_b32_e32 v3, 1, v0
	v_and_b32_e32 v2, 63, v0
	v_and_b32_e32 v3, 16, v3
	s_waitcnt lgkmcnt(0)
	s_and_b32 s0, s0, 0xffff
	v_mad_u32_u24 v53, v1, s0, v0
	s_and_b32 s0, s12, s1
	s_cselect_b64 s[0:1], 0, -1
	v_and_b32_e32 v57, 3, v53
	s_cmp_lg_u64 s[0:1], 0
	v_xor_b32_e32 v5, v3, v2
	v_lshrrev_b32_e32 v54, 2, v53
	v_lshlrev_b32_e32 v2, 5, v53
	v_lshlrev_b32_e32 v3, 2, v57
	s_movk_i32 s2, 0x100
	s_addc_u32 s6, s12, 0
	v_and_b32_e32 v56, 0x60, v2
	v_or_b32_e32 v2, s15, v54
	v_lshl_or_b32 v3, v54, 4, v3
	s_mul_hi_u32 s1, s13, s6
	s_mul_i32 s0, s13, s6
	v_cmp_gt_u32_e32 vcc, s2, v53
	v_cmp_gt_u32_e64 s[2:3], s13, v2
	v_add_u32_e32 v58, 0x7a00, v3
	v_mad_u64_u32 v[2:3], s[6:7], v2, s6, 0
	v_lshlrev_b32_e32 v4, 4, v57
	s_lshl_b64 s[0:1], s[0:1], 5
	v_lshl_or_b32 v4, v54, 6, v4
	s_add_u32 s8, s16, s0
	v_add_u32_e32 v59, 0x6a00, v4
	v_add_u32_e32 v60, 0x5a00, v4
	v_lshlrev_b64 v[3:4], 1, v[2:3]
	s_addc_u32 s9, s17, s1
	v_mov_b32_e32 v6, s9
	v_add_co_u32_e64 v61, s[6:7], s8, v3
	v_addc_co_u32_e64 v62, s[6:7], v6, v4, s[6:7]
	v_add_u32_e32 v3, 64, v0
	v_lshrrev_b32_e32 v3, 6, v3
	v_mul_u32_u24_e32 v5, 0xb4, v5
	s_movk_i32 s6, 0x2d00
	s_movk_i32 s0, 0x200
	s_movk_i32 s20, 0x7a00
	v_mov_b32_e32 v6, 0x5a00
	v_mad_u32_u24 v66, v3, s6, v5
	v_lshrrev_b32_e32 v3, 6, v0
	v_cmp_gt_u32_e64 s[0:1], s0, v53
	v_bfe_u32 v55, v53, 2, 6
	v_mov_b32_e32 v4, 0
	v_lshrrev_b32_e32 v63, 3, v53
	v_lshl_add_u32 v64, v1, 4, s20
	v_lshl_add_u32 v65, v1, 6, v6
	s_movk_i32 s28, 0xb4
	v_mad_u32_u24 v67, v3, s6, v5
	s_movk_i32 s29, 0x90
	s_movk_i32 s30, 0xff
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
.LBB2_2:                                ; %.preheader29.i
                                        ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB2_9 Depth 2
                                        ;     Child Loop BB2_14 Depth 2
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB2_6
; %bb.3:                                ; %.lr.ph.i
                                        ;   in Loop: Header=BB2_2 Depth=1
	v_or_b32_e32 v5, s27, v57
	v_cmp_gt_u32_e64 s[6:7], s12, v5
	s_and_b64 s[6:7], s[2:3], s[6:7]
	v_mov_b32_e32 v3, 0
	s_and_saveexec_b64 s[20:21], s[6:7]
	s_xor_b64 s[20:21], exec, s[20:21]
	s_cbranch_execz .LBB2_5
; %bb.4:                                ;   in Loop: Header=BB2_2 Depth=1
	v_mov_b32_e32 v6, v4
	v_add_u32_e32 v3, v5, v2
	v_lshlrev_b64 v[5:6], 1, v[5:6]
	v_lshlrev_b64 v[7:8], 5, v[3:4]
	v_add_co_u32_e64 v5, s[6:7], v61, v5
	v_addc_co_u32_e64 v6, s[6:7], v62, v6, s[6:7]
	global_load_ushort v15, v[5:6], off
	v_mov_b32_e32 v3, s17
	v_add_co_u32_e64 v13, s[6:7], s16, v7
	v_addc_co_u32_e64 v14, s[6:7], v3, v8, s[6:7]
	global_load_dwordx4 v[5:8], v[13:14], off
	global_load_dwordx4 v[9:12], v[13:14], off offset:16
	s_waitcnt vmcnt(1)
	ds_write_b128 v59, v[5:8]
	s_waitcnt vmcnt(0)
	ds_write_b128 v60, v[9:12]
	v_cvt_f32_f16_e32 v3, v15
.LBB2_5:                                ; %.preheader28.i.sink.split
                                        ;   in Loop: Header=BB2_2 Depth=1
	s_or_b64 exec, exec, s[20:21]
	ds_write_b32 v58, v3
.LBB2_6:                                ; %Flow795
                                        ;   in Loop: Header=BB2_2 Depth=1
	s_or_b64 exec, exec, s[8:9]
	s_and_saveexec_b64 s[20:21], s[0:1]
	s_cbranch_execz .LBB2_13
; %bb.7:                                ; %.lr.ph32.i
                                        ;   in Loop: Header=BB2_2 Depth=1
	s_lshr_b32 s6, s27, 2
	s_mul_i32 s7, s6, s14
	s_mul_hi_u32 s6, s6, s14
	s_mulk_i32 s6, 0x90
	s_mul_hi_u32 s8, s7, 0x90
	s_add_i32 s8, s8, s6
	s_mulk_i32 s7, 0x90
	s_add_u32 s22, s18, s7
	s_addc_u32 s23, s19, s8
	s_mov_b64 s[24:25], 0
	v_mov_b32_e32 v3, v54
	v_mov_b32_e32 v5, v63
	v_mov_b32_e32 v6, v53
	s_branch .LBB2_9
.LBB2_8:                                ;   in Loop: Header=BB2_9 Depth=2
	s_or_b64 exec, exec, s[8:9]
	v_add_u32_e32 v7, 0x100, v6
	v_cmp_lt_u32_e64 s[6:7], s30, v6
	v_add_u32_e32 v5, 32, v5
	v_add_u32_e32 v3, 64, v3
	s_or_b64 s[24:25], s[6:7], s[24:25]
	v_mov_b32_e32 v6, v7
	s_andn2_b64 exec, exec, s[24:25]
	s_cbranch_execz .LBB2_13
.LBB2_9:                                ;   Parent Loop BB2_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_and_b32_e32 v8, 3, v6
	v_and_b32_e32 v10, 16, v5
	v_or_b32_e32 v7, s27, v8
	v_add_u32_e32 v9, s26, v3
	v_xor_b32_e32 v10, v10, v55
	v_cmp_le_u32_e64 s[6:7], s14, v9
	v_and_or_b32 v10, v3, 64, v10
	v_cmp_le_u32_e64 s[8:9], s12, v7
	v_mul_u32_u24_e32 v7, 36, v8
	s_or_b64 s[6:7], s[6:7], s[8:9]
	v_mad_u32_u24 v7, v10, s28, v7
	s_and_saveexec_b64 s[8:9], s[6:7]
	s_xor_b64 s[6:7], exec, s[8:9]
; %bb.10:                               ;   in Loop: Header=BB2_9 Depth=2
	ds_write_b32 v7, v4
                                        ; implicit-def: $vgpr9
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $vgpr7
; %bb.11:                               ; %Flow792
                                        ;   in Loop: Header=BB2_9 Depth=2
	s_andn2_saveexec_b64 s[8:9], s[6:7]
	s_cbranch_execz .LBB2_8
; %bb.12:                               ;   in Loop: Header=BB2_9 Depth=2
	v_mov_b32_e32 v10, s22
	v_mov_b32_e32 v11, s23
	v_mad_u64_u32 v[9:10], s[6:7], v9, s29, v[10:11]
	v_lshlrev_b32_e32 v8, 2, v8
	v_add_co_u32_e64 v12, s[6:7], v9, v8
	v_addc_co_u32_e64 v13, s[6:7], 0, v10, s[6:7]
	v_add_co_u32_e64 v14, s[6:7], v9, v56
	v_addc_co_u32_e64 v15, s[6:7], 0, v10, s[6:7]
	global_load_dword v16, v[12:13], off
	global_load_dwordx4 v[8:11], v[14:15], off offset:16
                                        ; kill: killed $vgpr12 killed $vgpr13
	s_nop 0
	global_load_dwordx4 v[12:15], v[14:15], off offset:32
	s_waitcnt vmcnt(2)
	v_cvt_f16_f32_e32 v16, v16
	s_waitcnt vmcnt(1)
	ds_write2_b32 v7, v10, v11 offset0:3 offset1:4
	ds_write2_b32 v7, v8, v9 offset0:1 offset1:2
	s_waitcnt vmcnt(0)
	ds_write2_b32 v7, v14, v15 offset0:7 offset1:8
	ds_write_b32 v7, v16
	ds_write2_b32 v7, v12, v13 offset0:5 offset1:6
	s_branch .LBB2_8
.LBB2_13:                               ; %Flow794
                                        ;   in Loop: Header=BB2_2 Depth=1
	s_or_b64 exec, exec, s[20:21]
	s_mov_b32 s6, 0
	v_mov_b32_e32 v3, v65
	v_mov_b32_e32 v68, v64
	s_waitcnt lgkmcnt(0)
	s_barrier
.LBB2_14:                               ; %.preheader27.i
                                        ;   Parent Loop BB2_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_u32_e32 v5, s6, v67
	v_add_u32_e32 v69, s6, v66
	ds_read2_b32 v[13:14], v5 offset1:1
	ds_read2_b32 v[17:18], v5 offset0:2 offset1:3
	ds_read2_b32 v[15:16], v5 offset0:4 offset1:5
	ds_read2_b32 v[19:20], v5 offset0:6 offset1:7
	ds_read_b32 v70, v5 offset:32
	ds_read2_b32 v[5:6], v69 offset1:1
	ds_read2_b32 v[11:12], v69 offset0:2 offset1:3
	ds_read2_b32 v[7:8], v69 offset0:4 offset1:5
	ds_read2_b32 v[9:10], v69 offset0:6 offset1:7
	ds_read_b32 v69, v69 offset:32
	ds_read_b128 v[71:74], v3 offset:4096
	ds_read_b128 v[75:78], v3
	ds_read2_b32 v[79:80], v68 offset1:16
	s_add_i32 s6, s6, 36
	s_cmpk_eq_i32 s6, 0x90
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v71, v14, 0
	v_dot4_i32_i8 v71, v71, v6, 0
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v71, v75, v8, v71
	v_dot4_i32_i8 v71, v72, v11, v71
	v_dot4_i32_i8 v71, v76, v9, v71
	v_dot4_i32_i8 v71, v73, v12, v71
	v_dot4_i32_i8 v71, v77, v10, v71
	v_dot4_i32_i8 v81, v75, v16, v81
	v_dot4_i32_i8 v71, v74, v7, v71
	v_dot4_i32_i8 v81, v72, v17, v81
	v_dot4_i32_i8 v71, v78, v69, v71
	v_dot4_i32_i8 v81, v76, v19, v81
	v_cvt_f32_i32_e32 v71, v71
	v_dot4_i32_i8 v81, v73, v18, v81
	v_dot4_i32_i8 v81, v77, v20, v81
	v_dot4_i32_i8 v81, v74, v15, v81
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v72, v5, v79, neg(0) op_sel_hi:[1,0,0]
	v_dot4_i32_i8 v81, v78, v70, v81
	v_fmac_f32_e32 v51, v72, v71
	ds_read_b128 v[71:74], v3 offset:4352
	ds_read_b128 v[75:78], v3 offset:256
	v_fma_mix_f32 v82, v13, v79, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f32_i32_e32 v81, v81
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v79, v71, v14, 0
	v_dot4_i32_i8 v71, v71, v6, 0
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v79, v75, v16, v79
	v_dot4_i32_i8 v71, v75, v8, v71
	v_dot4_i32_i8 v79, v72, v17, v79
	v_dot4_i32_i8 v71, v72, v11, v71
	v_dot4_i32_i8 v79, v76, v19, v79
	v_dot4_i32_i8 v71, v76, v9, v71
	v_dot4_i32_i8 v79, v73, v18, v79
	v_dot4_i32_i8 v71, v73, v12, v71
	v_dot4_i32_i8 v79, v77, v20, v79
	v_dot4_i32_i8 v71, v77, v10, v71
	v_dot4_i32_i8 v79, v74, v15, v79
	v_dot4_i32_i8 v71, v74, v7, v71
	v_dot4_i32_i8 v79, v78, v70, v79
	v_dot4_i32_i8 v71, v78, v69, v71
	v_cvt_f32_i32_e32 v79, v79
	v_cvt_f32_i32_e32 v71, v71
	v_fmac_f32_e32 v52, v82, v81
	v_fma_mix_f32 v81, v13, v80, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v72, v5, v80, neg(0) op_sel_hi:[1,0,0]
	v_fmac_f32_e32 v50, v81, v79
	v_fmac_f32_e32 v49, v72, v71
	ds_read_b128 v[71:74], v3 offset:4608
	ds_read_b128 v[75:78], v3 offset:512
	ds_read2_b32 v[79:80], v68 offset0:32 offset1:48
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v71, v14, 0
	v_dot4_i32_i8 v71, v71, v6, 0
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v71, v75, v8, v71
	v_dot4_i32_i8 v71, v72, v11, v71
	v_dot4_i32_i8 v71, v76, v9, v71
	v_dot4_i32_i8 v71, v73, v12, v71
	v_dot4_i32_i8 v71, v77, v10, v71
	v_dot4_i32_i8 v81, v75, v16, v81
	v_dot4_i32_i8 v71, v74, v7, v71
	v_dot4_i32_i8 v81, v72, v17, v81
	v_dot4_i32_i8 v71, v78, v69, v71
	v_dot4_i32_i8 v81, v76, v19, v81
	v_cvt_f32_i32_e32 v71, v71
	v_dot4_i32_i8 v81, v73, v18, v81
	v_dot4_i32_i8 v81, v77, v20, v81
	v_dot4_i32_i8 v81, v74, v15, v81
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v72, v5, v79, neg(0) op_sel_hi:[1,0,0]
	v_dot4_i32_i8 v81, v78, v70, v81
	v_fmac_f32_e32 v47, v72, v71
	ds_read_b128 v[71:74], v3 offset:4864
	ds_read_b128 v[75:78], v3 offset:768
	v_fma_mix_f32 v82, v13, v79, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f32_i32_e32 v81, v81
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v79, v71, v14, 0
	v_dot4_i32_i8 v71, v71, v6, 0
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v79, v75, v16, v79
	v_dot4_i32_i8 v71, v75, v8, v71
	v_dot4_i32_i8 v79, v72, v17, v79
	v_dot4_i32_i8 v71, v72, v11, v71
	v_dot4_i32_i8 v79, v76, v19, v79
	v_dot4_i32_i8 v71, v76, v9, v71
	v_dot4_i32_i8 v79, v73, v18, v79
	v_dot4_i32_i8 v71, v73, v12, v71
	v_dot4_i32_i8 v79, v77, v20, v79
	v_dot4_i32_i8 v71, v77, v10, v71
	v_dot4_i32_i8 v79, v74, v15, v79
	v_dot4_i32_i8 v71, v74, v7, v71
	v_dot4_i32_i8 v79, v78, v70, v79
	v_dot4_i32_i8 v71, v78, v69, v71
	v_cvt_f32_i32_e32 v79, v79
	v_cvt_f32_i32_e32 v71, v71
	v_fmac_f32_e32 v48, v82, v81
	v_fma_mix_f32 v81, v13, v80, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v72, v5, v80, neg(0) op_sel_hi:[1,0,0]
	v_fmac_f32_e32 v46, v81, v79
	v_fmac_f32_e32 v45, v72, v71
	ds_read_b128 v[71:74], v3 offset:5120
	ds_read_b128 v[75:78], v3 offset:1024
	ds_read2_b32 v[79:80], v68 offset0:64 offset1:80
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v71, v14, 0
	v_dot4_i32_i8 v71, v71, v6, 0
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v71, v75, v8, v71
	v_dot4_i32_i8 v71, v72, v11, v71
	v_dot4_i32_i8 v71, v76, v9, v71
	v_dot4_i32_i8 v71, v73, v12, v71
	v_dot4_i32_i8 v71, v77, v10, v71
	v_dot4_i32_i8 v81, v75, v16, v81
	v_dot4_i32_i8 v71, v74, v7, v71
	v_dot4_i32_i8 v81, v72, v17, v81
	v_dot4_i32_i8 v71, v78, v69, v71
	v_dot4_i32_i8 v81, v76, v19, v81
	v_cvt_f32_i32_e32 v71, v71
	v_dot4_i32_i8 v81, v73, v18, v81
	v_dot4_i32_i8 v81, v77, v20, v81
	v_dot4_i32_i8 v81, v74, v15, v81
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v72, v5, v79, neg(0) op_sel_hi:[1,0,0]
	v_dot4_i32_i8 v81, v78, v70, v81
	v_fmac_f32_e32 v43, v72, v71
	ds_read_b128 v[71:74], v3 offset:5376
	ds_read_b128 v[75:78], v3 offset:1280
	v_fma_mix_f32 v82, v13, v79, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f32_i32_e32 v81, v81
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v79, v71, v14, 0
	v_dot4_i32_i8 v71, v71, v6, 0
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v79, v75, v16, v79
	v_dot4_i32_i8 v71, v75, v8, v71
	v_dot4_i32_i8 v79, v72, v17, v79
	v_dot4_i32_i8 v71, v72, v11, v71
	v_dot4_i32_i8 v79, v76, v19, v79
	v_dot4_i32_i8 v71, v76, v9, v71
	v_dot4_i32_i8 v79, v73, v18, v79
	v_dot4_i32_i8 v71, v73, v12, v71
	v_dot4_i32_i8 v79, v77, v20, v79
	v_dot4_i32_i8 v71, v77, v10, v71
	v_dot4_i32_i8 v79, v74, v15, v79
	v_dot4_i32_i8 v71, v74, v7, v71
	v_dot4_i32_i8 v79, v78, v70, v79
	v_dot4_i32_i8 v71, v78, v69, v71
	v_cvt_f32_i32_e32 v79, v79
	v_cvt_f32_i32_e32 v71, v71
	v_fmac_f32_e32 v44, v82, v81
	v_fma_mix_f32 v81, v13, v80, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v72, v5, v80, neg(0) op_sel_hi:[1,0,0]
	v_fmac_f32_e32 v42, v81, v79
	v_fmac_f32_e32 v41, v72, v71
	ds_read_b128 v[71:74], v3 offset:5632
	ds_read_b128 v[75:78], v3 offset:1536
	ds_read2_b32 v[79:80], v68 offset0:96 offset1:112
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v71, v14, 0
	v_dot4_i32_i8 v71, v71, v6, 0
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v71, v75, v8, v71
	v_dot4_i32_i8 v71, v72, v11, v71
	v_dot4_i32_i8 v71, v76, v9, v71
	v_dot4_i32_i8 v71, v73, v12, v71
	v_dot4_i32_i8 v71, v77, v10, v71
	v_dot4_i32_i8 v81, v75, v16, v81
	v_dot4_i32_i8 v71, v74, v7, v71
	v_dot4_i32_i8 v81, v72, v17, v81
	v_dot4_i32_i8 v71, v78, v69, v71
	v_dot4_i32_i8 v81, v76, v19, v81
	v_cvt_f32_i32_e32 v71, v71
	v_dot4_i32_i8 v81, v73, v18, v81
	v_dot4_i32_i8 v81, v77, v20, v81
	v_dot4_i32_i8 v81, v74, v15, v81
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v72, v5, v79, neg(0) op_sel_hi:[1,0,0]
	v_dot4_i32_i8 v81, v78, v70, v81
	v_fmac_f32_e32 v39, v72, v71
	ds_read_b128 v[71:74], v3 offset:5888
	ds_read_b128 v[75:78], v3 offset:1792
	v_fma_mix_f32 v82, v13, v79, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f32_i32_e32 v81, v81
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v79, v71, v14, 0
	v_dot4_i32_i8 v71, v71, v6, 0
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v79, v75, v16, v79
	v_dot4_i32_i8 v71, v75, v8, v71
	v_dot4_i32_i8 v79, v72, v17, v79
	v_dot4_i32_i8 v71, v72, v11, v71
	v_dot4_i32_i8 v79, v76, v19, v79
	v_dot4_i32_i8 v71, v76, v9, v71
	v_dot4_i32_i8 v79, v73, v18, v79
	v_dot4_i32_i8 v71, v73, v12, v71
	v_dot4_i32_i8 v79, v77, v20, v79
	v_dot4_i32_i8 v71, v77, v10, v71
	v_dot4_i32_i8 v79, v74, v15, v79
	v_dot4_i32_i8 v71, v74, v7, v71
	v_dot4_i32_i8 v79, v78, v70, v79
	v_dot4_i32_i8 v71, v78, v69, v71
	v_cvt_f32_i32_e32 v79, v79
	v_cvt_f32_i32_e32 v71, v71
	v_fmac_f32_e32 v40, v82, v81
	v_fma_mix_f32 v81, v13, v80, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v72, v5, v80, neg(0) op_sel_hi:[1,0,0]
	v_fmac_f32_e32 v38, v81, v79
	v_fmac_f32_e32 v37, v72, v71
	ds_read_b128 v[71:74], v3 offset:6144
	ds_read_b128 v[75:78], v3 offset:2048
	ds_read2_b32 v[79:80], v68 offset0:128 offset1:144
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v71, v14, 0
	v_dot4_i32_i8 v71, v71, v6, 0
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v71, v75, v8, v71
	v_dot4_i32_i8 v71, v72, v11, v71
	v_dot4_i32_i8 v71, v76, v9, v71
	v_dot4_i32_i8 v71, v73, v12, v71
	v_dot4_i32_i8 v71, v77, v10, v71
	v_dot4_i32_i8 v81, v75, v16, v81
	v_dot4_i32_i8 v71, v74, v7, v71
	v_dot4_i32_i8 v81, v72, v17, v81
	v_dot4_i32_i8 v71, v78, v69, v71
	v_dot4_i32_i8 v81, v76, v19, v81
	v_cvt_f32_i32_e32 v71, v71
	v_dot4_i32_i8 v81, v73, v18, v81
	v_dot4_i32_i8 v81, v77, v20, v81
	v_dot4_i32_i8 v81, v74, v15, v81
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v72, v5, v79, neg(0) op_sel_hi:[1,0,0]
	v_dot4_i32_i8 v81, v78, v70, v81
	v_fmac_f32_e32 v35, v72, v71
	ds_read_b128 v[71:74], v3 offset:6400
	ds_read_b128 v[75:78], v3 offset:2304
	v_fma_mix_f32 v82, v13, v79, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f32_i32_e32 v81, v81
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v79, v71, v14, 0
	v_dot4_i32_i8 v71, v71, v6, 0
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v79, v75, v16, v79
	v_dot4_i32_i8 v71, v75, v8, v71
	v_dot4_i32_i8 v79, v72, v17, v79
	v_dot4_i32_i8 v71, v72, v11, v71
	v_dot4_i32_i8 v79, v76, v19, v79
	v_dot4_i32_i8 v71, v76, v9, v71
	v_dot4_i32_i8 v79, v73, v18, v79
	v_dot4_i32_i8 v71, v73, v12, v71
	v_dot4_i32_i8 v79, v77, v20, v79
	v_dot4_i32_i8 v71, v77, v10, v71
	v_dot4_i32_i8 v79, v74, v15, v79
	v_dot4_i32_i8 v71, v74, v7, v71
	v_dot4_i32_i8 v79, v78, v70, v79
	v_dot4_i32_i8 v71, v78, v69, v71
	v_cvt_f32_i32_e32 v79, v79
	v_cvt_f32_i32_e32 v71, v71
	v_fmac_f32_e32 v36, v82, v81
	v_fma_mix_f32 v81, v13, v80, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v72, v5, v80, neg(0) op_sel_hi:[1,0,0]
	v_fmac_f32_e32 v34, v81, v79
	v_fmac_f32_e32 v33, v72, v71
	ds_read_b128 v[71:74], v3 offset:6656
	ds_read_b128 v[75:78], v3 offset:2560
	ds_read2_b32 v[79:80], v68 offset0:160 offset1:176
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v71, v14, 0
	v_dot4_i32_i8 v71, v71, v6, 0
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v71, v75, v8, v71
	v_dot4_i32_i8 v71, v72, v11, v71
	v_dot4_i32_i8 v71, v76, v9, v71
	v_dot4_i32_i8 v71, v73, v12, v71
	v_dot4_i32_i8 v71, v77, v10, v71
	v_dot4_i32_i8 v81, v75, v16, v81
	v_dot4_i32_i8 v71, v74, v7, v71
	v_dot4_i32_i8 v81, v72, v17, v81
	v_dot4_i32_i8 v71, v78, v69, v71
	v_dot4_i32_i8 v81, v76, v19, v81
	v_cvt_f32_i32_e32 v71, v71
	v_dot4_i32_i8 v81, v73, v18, v81
	v_dot4_i32_i8 v81, v77, v20, v81
	v_dot4_i32_i8 v81, v74, v15, v81
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v72, v5, v79, neg(0) op_sel_hi:[1,0,0]
	v_dot4_i32_i8 v81, v78, v70, v81
	v_fmac_f32_e32 v31, v72, v71
	ds_read_b128 v[71:74], v3 offset:6912
	ds_read_b128 v[75:78], v3 offset:2816
	v_fma_mix_f32 v82, v13, v79, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f32_i32_e32 v81, v81
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v79, v71, v14, 0
	v_dot4_i32_i8 v71, v71, v6, 0
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v79, v75, v16, v79
	v_dot4_i32_i8 v71, v75, v8, v71
	v_dot4_i32_i8 v79, v72, v17, v79
	v_dot4_i32_i8 v71, v72, v11, v71
	v_dot4_i32_i8 v79, v76, v19, v79
	v_dot4_i32_i8 v71, v76, v9, v71
	v_dot4_i32_i8 v79, v73, v18, v79
	v_dot4_i32_i8 v71, v73, v12, v71
	v_dot4_i32_i8 v79, v77, v20, v79
	v_dot4_i32_i8 v71, v77, v10, v71
	v_dot4_i32_i8 v79, v74, v15, v79
	v_dot4_i32_i8 v71, v74, v7, v71
	v_dot4_i32_i8 v79, v78, v70, v79
	v_dot4_i32_i8 v71, v78, v69, v71
	v_cvt_f32_i32_e32 v79, v79
	v_cvt_f32_i32_e32 v71, v71
	v_fmac_f32_e32 v32, v82, v81
	v_fma_mix_f32 v81, v13, v80, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v72, v5, v80, neg(0) op_sel_hi:[1,0,0]
	v_fmac_f32_e32 v30, v81, v79
	v_fmac_f32_e32 v29, v72, v71
	ds_read_b128 v[71:74], v3 offset:7168
	ds_read_b128 v[75:78], v3 offset:3072
	ds_read2_b32 v[79:80], v68 offset0:192 offset1:208
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v71, v14, 0
	v_dot4_i32_i8 v71, v71, v6, 0
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v71, v75, v8, v71
	v_dot4_i32_i8 v71, v72, v11, v71
	v_dot4_i32_i8 v71, v76, v9, v71
	v_dot4_i32_i8 v71, v73, v12, v71
	v_dot4_i32_i8 v71, v77, v10, v71
	v_dot4_i32_i8 v81, v75, v16, v81
	v_dot4_i32_i8 v71, v74, v7, v71
	v_dot4_i32_i8 v81, v72, v17, v81
	v_dot4_i32_i8 v71, v78, v69, v71
	v_dot4_i32_i8 v81, v76, v19, v81
	v_cvt_f32_i32_e32 v71, v71
	v_dot4_i32_i8 v81, v73, v18, v81
	v_dot4_i32_i8 v81, v77, v20, v81
	v_dot4_i32_i8 v81, v74, v15, v81
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v72, v5, v79, neg(0) op_sel_hi:[1,0,0]
	v_dot4_i32_i8 v81, v78, v70, v81
	v_fmac_f32_e32 v27, v72, v71
	ds_read_b128 v[71:74], v3 offset:7424
	ds_read_b128 v[75:78], v3 offset:3328
	v_fma_mix_f32 v82, v13, v79, neg(0) op_sel_hi:[1,0,0]
	v_cvt_f32_i32_e32 v81, v81
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v79, v71, v14, 0
	v_dot4_i32_i8 v71, v71, v6, 0
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v79, v75, v16, v79
	v_dot4_i32_i8 v71, v75, v8, v71
	v_dot4_i32_i8 v79, v72, v17, v79
	v_dot4_i32_i8 v71, v72, v11, v71
	v_dot4_i32_i8 v79, v76, v19, v79
	v_dot4_i32_i8 v71, v76, v9, v71
	v_dot4_i32_i8 v79, v73, v18, v79
	v_dot4_i32_i8 v71, v73, v12, v71
	v_dot4_i32_i8 v79, v77, v20, v79
	v_dot4_i32_i8 v71, v77, v10, v71
	v_dot4_i32_i8 v79, v74, v15, v79
	v_dot4_i32_i8 v71, v74, v7, v71
	v_dot4_i32_i8 v79, v78, v70, v79
	v_dot4_i32_i8 v71, v78, v69, v71
	v_cvt_f32_i32_e32 v79, v79
	v_cvt_f32_i32_e32 v71, v71
	v_fmac_f32_e32 v28, v82, v81
	v_fma_mix_f32 v81, v13, v80, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v72, v5, v80, neg(0) op_sel_hi:[1,0,0]
	v_fmac_f32_e32 v26, v81, v79
	v_fmac_f32_e32 v25, v72, v71
	ds_read_b128 v[71:74], v3 offset:7680
	ds_read_b128 v[75:78], v3 offset:3584
	ds_read2_b32 v[79:80], v68 offset0:224 offset1:240
	v_add_u32_e32 v68, 4, v68
	s_waitcnt lgkmcnt(2)
	v_dot4_i32_i8 v81, v71, v14, 0
	v_dot4_i32_i8 v71, v71, v6, 0
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v71, v75, v8, v71
	v_dot4_i32_i8 v71, v72, v11, v71
	v_dot4_i32_i8 v71, v76, v9, v71
	v_dot4_i32_i8 v71, v73, v12, v71
	v_dot4_i32_i8 v71, v77, v10, v71
	v_dot4_i32_i8 v81, v75, v16, v81
	v_dot4_i32_i8 v71, v74, v7, v71
	v_dot4_i32_i8 v81, v72, v17, v81
	v_dot4_i32_i8 v71, v78, v69, v71
	v_dot4_i32_i8 v81, v76, v19, v81
	v_cvt_f32_i32_e32 v71, v71
	v_dot4_i32_i8 v81, v73, v18, v81
	v_dot4_i32_i8 v81, v77, v20, v81
	v_dot4_i32_i8 v81, v74, v15, v81
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v72, v5, v79, neg(0) op_sel_hi:[1,0,0]
	v_dot4_i32_i8 v81, v78, v70, v81
	v_fmac_f32_e32 v23, v72, v71
	ds_read_b128 v[71:74], v3 offset:7936
	ds_read_b128 v[75:78], v3 offset:3840
	v_cvt_f32_i32_e32 v81, v81
	v_fma_mix_f32 v82, v13, v79, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v13, v13, v80, neg(0) op_sel_hi:[1,0,0]
	s_waitcnt lgkmcnt(1)
	v_dot4_i32_i8 v14, v71, v14, 0
	v_dot4_i32_i8 v6, v71, v6, 0
	s_waitcnt lgkmcnt(0)
	v_dot4_i32_i8 v14, v75, v16, v14
	v_dot4_i32_i8 v6, v75, v8, v6
	v_dot4_i32_i8 v14, v72, v17, v14
	v_dot4_i32_i8 v6, v72, v11, v6
	v_dot4_i32_i8 v14, v76, v19, v14
	v_dot4_i32_i8 v6, v76, v9, v6
	v_dot4_i32_i8 v14, v73, v18, v14
	v_dot4_i32_i8 v6, v73, v12, v6
	v_dot4_i32_i8 v14, v77, v20, v14
	v_dot4_i32_i8 v6, v77, v10, v6
	v_dot4_i32_i8 v14, v74, v15, v14
	v_dot4_i32_i8 v6, v74, v7, v6
	v_dot4_i32_i8 v14, v78, v70, v14
	v_dot4_i32_i8 v6, v78, v69, v6
	v_cvt_f32_i32_e32 v14, v14
	v_cvt_f32_i32_e32 v6, v6
	v_fma_mix_f32 v5, v5, v80, neg(0) op_sel_hi:[1,0,0]
	v_fmac_f32_e32 v24, v82, v81
	v_fmac_f32_e32 v22, v13, v14
	v_fmac_f32_e32 v21, v5, v6
	v_add_u32_e32 v3, 16, v3
	s_cbranch_scc0 .LBB2_14
; %bb.15:                               ;   in Loop: Header=BB2_2 Depth=1
	s_add_i32 s27, s27, 4
	s_cmp_ge_u32 s27, s12
	s_barrier
	s_cbranch_scc0 .LBB2_2
	s_branch .LBB2_17
.LBB2_16:
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
.LBB2_17:                               ; %.preheader26.i
	s_load_dword s4, s[4:5], 0x58
	v_add_u32_e32 v1, s15, v1
	v_add_u32_e32 v4, s26, v0
	v_cmp_gt_u32_e32 vcc, s13, v1
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_22
; %bb.18:                               ; %.preheader.i
	v_mov_b32_e32 v2, 0
	v_lshlrev_b64 v[2:3], 2, v[1:2]
	v_mov_b32_e32 v5, s11
	v_add_co_u32_e32 v0, vcc, s10, v2
	v_addc_co_u32_e32 v2, vcc, v5, v3, vcc
	v_cmp_gt_u32_e32 vcc, s14, v4
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB2_20
; %bb.19:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[6:7], s4, v4, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v52, off
.LBB2_20:
	s_or_b64 exec, exec, s[2:3]
	v_add_u32_e32 v3, 64, v4
	v_cmp_gt_u32_e32 vcc, s14, v3
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_22
; %bb.21:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[2:3], s4, v3, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v51, off
.LBB2_22:                               ; %Flow791
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 4, v1
	v_cmp_gt_u32_e32 vcc, s13, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_27
; %bb.23:                               ; %.preheader.1.i
	v_mov_b32_e32 v3, 0
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_mov_b32_e32 v5, s11
	v_add_co_u32_e32 v0, vcc, s10, v2
	v_addc_co_u32_e32 v2, vcc, v5, v3, vcc
	v_cmp_gt_u32_e32 vcc, s14, v4
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB2_25
; %bb.24:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[6:7], s4, v4, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v50, off
.LBB2_25:
	s_or_b64 exec, exec, s[2:3]
	v_add_u32_e32 v3, 64, v4
	v_cmp_gt_u32_e32 vcc, s14, v3
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_27
; %bb.26:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[2:3], s4, v3, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v49, off
.LBB2_27:                               ; %Flow789
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 8, v1
	v_cmp_gt_u32_e32 vcc, s13, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_32
; %bb.28:                               ; %.preheader.2.i
	v_mov_b32_e32 v3, 0
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_mov_b32_e32 v5, s11
	v_add_co_u32_e32 v0, vcc, s10, v2
	v_addc_co_u32_e32 v2, vcc, v5, v3, vcc
	v_cmp_gt_u32_e32 vcc, s14, v4
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB2_30
; %bb.29:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[6:7], s4, v4, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v48, off
.LBB2_30:
	s_or_b64 exec, exec, s[2:3]
	v_add_u32_e32 v3, 64, v4
	v_cmp_gt_u32_e32 vcc, s14, v3
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_32
; %bb.31:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[2:3], s4, v3, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v47, off
.LBB2_32:                               ; %Flow787
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 12, v1
	v_cmp_gt_u32_e32 vcc, s13, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_37
; %bb.33:                               ; %.preheader.3.i
	v_mov_b32_e32 v3, 0
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_mov_b32_e32 v5, s11
	v_add_co_u32_e32 v0, vcc, s10, v2
	v_addc_co_u32_e32 v2, vcc, v5, v3, vcc
	v_cmp_gt_u32_e32 vcc, s14, v4
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB2_35
; %bb.34:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[6:7], s4, v4, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v46, off
.LBB2_35:
	s_or_b64 exec, exec, s[2:3]
	v_add_u32_e32 v3, 64, v4
	v_cmp_gt_u32_e32 vcc, s14, v3
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_37
; %bb.36:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[2:3], s4, v3, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v45, off
.LBB2_37:                               ; %Flow785
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 16, v1
	v_cmp_gt_u32_e32 vcc, s13, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_42
; %bb.38:                               ; %.preheader.4.i
	v_mov_b32_e32 v3, 0
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_mov_b32_e32 v5, s11
	v_add_co_u32_e32 v0, vcc, s10, v2
	v_addc_co_u32_e32 v2, vcc, v5, v3, vcc
	v_cmp_gt_u32_e32 vcc, s14, v4
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB2_40
; %bb.39:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[6:7], s4, v4, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v44, off
.LBB2_40:
	s_or_b64 exec, exec, s[2:3]
	v_add_u32_e32 v3, 64, v4
	v_cmp_gt_u32_e32 vcc, s14, v3
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_42
; %bb.41:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[2:3], s4, v3, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v43, off
.LBB2_42:                               ; %Flow783
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 20, v1
	v_cmp_gt_u32_e32 vcc, s13, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_47
; %bb.43:                               ; %.preheader.5.i
	v_mov_b32_e32 v3, 0
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_mov_b32_e32 v5, s11
	v_add_co_u32_e32 v0, vcc, s10, v2
	v_addc_co_u32_e32 v2, vcc, v5, v3, vcc
	v_cmp_gt_u32_e32 vcc, s14, v4
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB2_45
; %bb.44:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[6:7], s4, v4, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v42, off
.LBB2_45:
	s_or_b64 exec, exec, s[2:3]
	v_add_u32_e32 v3, 64, v4
	v_cmp_gt_u32_e32 vcc, s14, v3
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_47
; %bb.46:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[2:3], s4, v3, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v41, off
.LBB2_47:                               ; %Flow781
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 24, v1
	v_cmp_gt_u32_e32 vcc, s13, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_52
; %bb.48:                               ; %.preheader.6.i
	v_mov_b32_e32 v3, 0
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_mov_b32_e32 v5, s11
	v_add_co_u32_e32 v0, vcc, s10, v2
	v_addc_co_u32_e32 v2, vcc, v5, v3, vcc
	v_cmp_gt_u32_e32 vcc, s14, v4
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB2_50
; %bb.49:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[6:7], s4, v4, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v40, off
.LBB2_50:
	s_or_b64 exec, exec, s[2:3]
	v_add_u32_e32 v3, 64, v4
	v_cmp_gt_u32_e32 vcc, s14, v3
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_52
; %bb.51:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[2:3], s4, v3, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v39, off
.LBB2_52:                               ; %Flow779
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 28, v1
	v_cmp_gt_u32_e32 vcc, s13, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_57
; %bb.53:                               ; %.preheader.7.i
	v_mov_b32_e32 v3, 0
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_mov_b32_e32 v5, s11
	v_add_co_u32_e32 v0, vcc, s10, v2
	v_addc_co_u32_e32 v2, vcc, v5, v3, vcc
	v_cmp_gt_u32_e32 vcc, s14, v4
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB2_55
; %bb.54:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[6:7], s4, v4, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v38, off
.LBB2_55:
	s_or_b64 exec, exec, s[2:3]
	v_add_u32_e32 v3, 64, v4
	v_cmp_gt_u32_e32 vcc, s14, v3
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_57
; %bb.56:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[2:3], s4, v3, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v37, off
.LBB2_57:                               ; %Flow777
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 32, v1
	v_cmp_gt_u32_e32 vcc, s13, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_62
; %bb.58:                               ; %.preheader.8.i
	v_mov_b32_e32 v3, 0
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_mov_b32_e32 v5, s11
	v_add_co_u32_e32 v0, vcc, s10, v2
	v_addc_co_u32_e32 v2, vcc, v5, v3, vcc
	v_cmp_gt_u32_e32 vcc, s14, v4
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB2_60
; %bb.59:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[6:7], s4, v4, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v36, off
.LBB2_60:
	s_or_b64 exec, exec, s[2:3]
	v_add_u32_e32 v3, 64, v4
	v_cmp_gt_u32_e32 vcc, s14, v3
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_62
; %bb.61:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[2:3], s4, v3, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v35, off
.LBB2_62:                               ; %Flow775
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 36, v1
	v_cmp_gt_u32_e32 vcc, s13, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_67
; %bb.63:                               ; %.preheader.9.i
	v_mov_b32_e32 v3, 0
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_mov_b32_e32 v5, s11
	v_add_co_u32_e32 v0, vcc, s10, v2
	v_addc_co_u32_e32 v2, vcc, v5, v3, vcc
	v_cmp_gt_u32_e32 vcc, s14, v4
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB2_65
; %bb.64:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[6:7], s4, v4, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v34, off
.LBB2_65:
	s_or_b64 exec, exec, s[2:3]
	v_add_u32_e32 v3, 64, v4
	v_cmp_gt_u32_e32 vcc, s14, v3
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_67
; %bb.66:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[2:3], s4, v3, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v33, off
.LBB2_67:                               ; %Flow773
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 40, v1
	v_cmp_gt_u32_e32 vcc, s13, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_72
; %bb.68:                               ; %.preheader.10.i
	v_mov_b32_e32 v3, 0
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_mov_b32_e32 v5, s11
	v_add_co_u32_e32 v0, vcc, s10, v2
	v_addc_co_u32_e32 v2, vcc, v5, v3, vcc
	v_cmp_gt_u32_e32 vcc, s14, v4
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB2_70
; %bb.69:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[6:7], s4, v4, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v32, off
.LBB2_70:
	s_or_b64 exec, exec, s[2:3]
	v_add_u32_e32 v3, 64, v4
	v_cmp_gt_u32_e32 vcc, s14, v3
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_72
; %bb.71:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[2:3], s4, v3, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v31, off
.LBB2_72:                               ; %Flow771
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 44, v1
	v_cmp_gt_u32_e32 vcc, s13, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_77
; %bb.73:                               ; %.preheader.11.i
	v_mov_b32_e32 v3, 0
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_mov_b32_e32 v5, s11
	v_add_co_u32_e32 v0, vcc, s10, v2
	v_addc_co_u32_e32 v2, vcc, v5, v3, vcc
	v_cmp_gt_u32_e32 vcc, s14, v4
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB2_75
; %bb.74:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[6:7], s4, v4, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v30, off
.LBB2_75:
	s_or_b64 exec, exec, s[2:3]
	v_add_u32_e32 v3, 64, v4
	v_cmp_gt_u32_e32 vcc, s14, v3
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_77
; %bb.76:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[2:3], s4, v3, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v29, off
.LBB2_77:                               ; %Flow769
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 48, v1
	v_cmp_gt_u32_e32 vcc, s13, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_82
; %bb.78:                               ; %.preheader.12.i
	v_mov_b32_e32 v3, 0
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_mov_b32_e32 v5, s11
	v_add_co_u32_e32 v0, vcc, s10, v2
	v_addc_co_u32_e32 v2, vcc, v5, v3, vcc
	v_cmp_gt_u32_e32 vcc, s14, v4
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB2_80
; %bb.79:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[6:7], s4, v4, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v28, off
.LBB2_80:
	s_or_b64 exec, exec, s[2:3]
	v_add_u32_e32 v3, 64, v4
	v_cmp_gt_u32_e32 vcc, s14, v3
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_82
; %bb.81:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[2:3], s4, v3, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v27, off
.LBB2_82:                               ; %Flow767
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 52, v1
	v_cmp_gt_u32_e32 vcc, s13, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_87
; %bb.83:                               ; %.preheader.13.i
	v_mov_b32_e32 v3, 0
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_mov_b32_e32 v5, s11
	v_add_co_u32_e32 v0, vcc, s10, v2
	v_addc_co_u32_e32 v2, vcc, v5, v3, vcc
	v_cmp_gt_u32_e32 vcc, s14, v4
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB2_85
; %bb.84:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[6:7], s4, v4, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v26, off
.LBB2_85:
	s_or_b64 exec, exec, s[2:3]
	v_add_u32_e32 v3, 64, v4
	v_cmp_gt_u32_e32 vcc, s14, v3
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_87
; %bb.86:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[2:3], s4, v3, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v25, off
.LBB2_87:                               ; %Flow765
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 56, v1
	v_cmp_gt_u32_e32 vcc, s13, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_92
; %bb.88:                               ; %.preheader.14.i
	v_mov_b32_e32 v3, 0
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_mov_b32_e32 v5, s11
	v_add_co_u32_e32 v0, vcc, s10, v2
	v_addc_co_u32_e32 v2, vcc, v5, v3, vcc
	v_cmp_gt_u32_e32 vcc, s14, v4
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB2_90
; %bb.89:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[6:7], s4, v4, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v24, off
.LBB2_90:
	s_or_b64 exec, exec, s[2:3]
	v_add_u32_e32 v3, 64, v4
	v_cmp_gt_u32_e32 vcc, s14, v3
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_92
; %bb.91:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[5:6], s[2:3], s4, v3, 0
	v_lshlrev_b64 v[5:6], 2, v[5:6]
	v_add_co_u32_e32 v5, vcc, v0, v5
	v_addc_co_u32_e32 v6, vcc, v2, v6, vcc
	global_store_dword v[5:6], v23, off
.LBB2_92:                               ; %Flow763
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v0, 60, v1
	v_cmp_gt_u32_e32 vcc, s13, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_97
; %bb.93:                               ; %.preheader.15.i
	v_mov_b32_e32 v1, 0
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	v_mov_b32_e32 v2, s11
	v_add_co_u32_e32 v0, vcc, s10, v0
	v_addc_co_u32_e32 v1, vcc, v2, v1, vcc
	v_cmp_gt_u32_e32 vcc, s14, v4
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_95
; %bb.94:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[2:3], s[2:3], s4, v4, 0
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_add_co_u32_e32 v2, vcc, v0, v2
	v_addc_co_u32_e32 v3, vcc, v1, v3, vcc
	global_store_dword v[2:3], v22, off
.LBB2_95:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 64, v4
	v_cmp_gt_u32_e32 vcc, s14, v2
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB2_97
; %bb.96:
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[2:3], s[0:1], s4, v2, 0
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_add_co_u32_e32 v0, vcc, v0, v2
	v_addc_co_u32_e32 v1, vcc, v1, v3, vcc
	global_store_dword v[0:1], v21, off
.LBB2_97:                               ; %_ZL27mmq_gemm_q8_0_repacked_implILb0ELi64ELi2ELi4EEvPKhPK10block_q8_1PfjjjjPKiS7_S7_S7_jmj.exit
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZL22mmq_gemm_q8_0_repackedILb0ELi2ELi4EEvPKhPK10block_q8_1PfjjjjPKiS7_S7_S7_jmj
		.amdhsa_group_segment_fixed_size 32256
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 352
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 85
		.amdhsa_next_free_sgpr 98
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZL22mmq_gemm_q8_0_repackedILb0ELi2ELi4EEvPKhPK10block_q8_1PfjjjjPKiS7_S7_S7_jmj,"axG",@progbits,_ZL22mmq_gemm_q8_0_repackedILb0ELi2ELi4EEvPKhPK10block_q8_1PfjjjjPKiS7_S7_S7_jmj,comdat
```
