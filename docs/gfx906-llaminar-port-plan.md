# gfx906 Optimization Porting Plan — Llaminar → mx-llama.cpp-gfx906

> **Source:** `github.com/Llaminar/llaminar` (LLM inference engine in C++, gfx906-only ROCm path)
> **Target:** `mx-llama.cpp-gfx906` (llama.cpp fork with existing gfx906 kernels)
> **Goal:** Select Llaminar's gfx906-specific optimizations and plan their port into this project.

---

## 1. Context

### 1.1 Llaminar's gfx906 path
Llaminar is a standalone engine with custom ROCm kernels written *exclusively* for gfx906 (Vega20 / MI50 / MI60). Hardware constraints driving every decision:

- **GCN 5.0**, 60 CUs, **64-lane wavefronts** (not CUDA's 32)
- **No matrix cores / no MFMA** — packed-math dot instructions (`v_dot2_f32_f16`, `v_dot4_i32_i8`) are the only throughput paths
- **64 KB LDS per CU** (vs 160+ KB on CDNA2+)
- 16 KB L1I per CU (branchy loops hurt)
- ~378–416 GB/s HBM; decode is **latency/bandwidth-bound**, not compute-bound
- Supports FP32, FP16, **no native BF16** (emulated via FP32)
- Cross-lane ops (`__shfl_xor`/`__shfl_down`) compile to `ds_bpermute`/`ds_permute` (DPP-style) — zero LDS, zero barrier

Key ROCm source: `src/v2/kernels/rocm/{attention,gdn,gemm,moe,ops,kvcache,repack}` and `src/v2/backends/rocm/`.

### 1.2 This project's existing gfx906 coverage (already present — excluded from porting)
The following are **already implemented** and were verified against the repo:

| Capability | File(s) in this repo |
|---|---|
| DPP warp reductions, `GGML_CUDA_CC_IS_GCN` | `ggml-cuda/common.cuh` |
| Q8 flash attention tile kernel | `ggml-cuda/fattn-mma-f16.cuh` |
| Warp-cooperative MMVQ (Q4_0/Q4_1/Q8_0) | `ggml-cuda/mmvq.cu` |
| Software-pipelined MMQ, vectorized loads | `ggml-cuda/mmq.cuh` |
| DPP-based Q8 quantize warp reductions | `ggml-cuda/quantize.cu`, `quantize.cuh` |
| GCN weight repack (VNNI-like) | `ggml-cuda/repack-gcn.cu`, `repack-gcn.cuh` |
| Custom TP allreduce (BF16-on-wire) | `ggml-cuda/tp-allreduce.cu` |
| Gated delta net chunk kernel (prefill) | recent commits (`ggml-cuda`) |
| gfx906 dispatch, SOLVE_TRI limits, turbo3 ops | `ggml-cuda/ggml-cuda.cu` |

→ The porting effort focuses on **gaps**, not re-implementing what exists.

---

## 2. Prioritized optimization backlog

Ranked by **value ÷ effort**. Each entry: technique, why gfx906-relevant, Llaminar source, current project state, porting plan, effort.

---

### P1 — FP16-KV Flash Attention via `__builtin_amdgcn_fdot2`
- **Technique:** Store K/V as native FP16 in LDS; compute `S = Q·Kᵀ` with single-instruction packed FP16 dot `score = __builtin_amdgcn_fdot2(q_h2, k_h2, score, false)`, FP32 accumulation. Use **4 independent accumulators** to hide `v_dot2` latency (ILP). (`attention/ROCmFlashAttentionKernels.hip:982-1010`, `:1377`, `:1377-1381`)
- **Why gfx906:** Only 2×-FP32-rate ALU path; also halves KV LDS/HBM traffic vs Q8.
- **Current state:** `fattn-mma-f16.cuh` is Q8-centric; no native FP16-KV path.
- **Port plan:** Add a FP16-KV variant of the FA tile. Keep online softmax, reuse existing causal-mask fast path.
- **Effort:** Medium · **Value:** High

### P2 — `exp2f(x*log2e)` softmax / fast exp
- **Technique:** Replace `expf`/`__expf` with hardware `v_exp_f32` via `exp2f(x * 1.442695f)`. (`attention/ROCmFlashAttentionKernels.hip:1196-1199`, used `:1435`, `:1890`)
- **Why gfx906:** ~4 cyc vs ~50+ cyc per element in the hottest scalar op.
- **Current state:** Likely using `expf`.
- **Port plan:** Drop-in swap in softmax, MoE, sampling kernels.
- **Effort:** Trivial · **Value:** Medium

### P3 — Shuffle-only Q8_1 block quant (0 LDS, 0 barriers)
- **Technique:** 32-lane wavefront reduces absmax/sum via `__shfl_xor` butterflies only. (`kvcache/ROCmRingKVCacheKernels.hip:431-475`)
- **Why gfx906:** Removes prior 10-barrier/LDS implementation → higher occupancy.
- **Current state:** `quantize.cu` uses DPP but with LDS staging in places.
- **Port plan:** Template the shuffle-only reduction for `quantize.cu` Q8 paths.
- **Effort:** Low · **Value:** High

### P4 — RoPE on-GPU `inv_freq` + contiguous position + graph-capture params
- **Technique:** Compute `inv_freq` on-device (`rope_populate_inv_freq_kernel`), compute `pos` on-GPU for contiguous path, thread stable `device_params` through launcher. (`ops/ROCmRoPEKernels.hip:1473-1519`, `:230-280`)
- **Why gfx906:** Avoids synchronous H2D `inv_freq` copy (measured ~11 ms/layer pipeline drain during prefill).
- **Current state:** RoPE already optimized; H2D `inv_freq` likely re-copied.
- **Port plan:** Move `inv_freq` to a device buffer populated once; add contiguous-pos kernel.
- **Effort:** Low · **Value:** High

### P5 — K-block (kb) wave-targeting sharding + split-reduce for decode MMVQ
- **Technique:** Split K into `kb` shards across `blockIdx.y`; choose `kb` to hit a fixed **waves-per-CU** budget (`MIN_KGROUPS_PER_WAVE=16`), force `kb` to evenly divide K (uneven shards cost ~23%); reduce partials via a staging buffer or `atomicAdd`. (`gemm/ROCmGemvKernel_INT8_VNNI.hip:73-285`, `gemm/ROCmGemvKernel_native_VNNI.hip:337-452`)
- **Why gfx906:** Your `mmvq.cu` sizes by thread/warp count; this saturates MI50's 60 CUs and hides HBM latency better.
- **Current state:** MMVQ present, no K-sharding heuristic.
- **Port plan:** Add K-shard dimension + split-reduce buffer to `mmvq.cu`; tune `MIN_KGROUPS_PER_WAVE`.
- **Effort:** Medium · **Value:** High

### P6 — IQ-grid LDS preload for low-bpw IQ formats
- **Technique:** Cooperatively load 2–16 KB dequant grids from `__constant__` into LDS once per workgroup; decode via `ds_read`. (`gemm/ROCmGemvKernel_native_VNNI.hip:3387-3474`, `README.native-vnni-isa-analysis.md:3378`)
- **Why gfx906:** Documented: IQ1_S/M **+190%**, IQ3_S **+115%**, IQ2_S **+82%**, IQ2_XS **+76%**. Fixes the "death by LUT" bottleneck.
- **Current state:** `repack-gcn.cu` covers Q4/Q8; IQ dequant still scattered constant/global lookups.
- **Port plan:** Add LDS grid preload to MMVQ for IQ formats.
- **Effort:** Medium · **Value:** High (for IQ users)

### P7 — V7 runtime safe-tile / boundary split for prefill MMQ
- **Technique:** Split K-loop into unconditional vectorized **safe loop** (zero boundary checks) + rare **boundary loop**; eliminates ~99 branches/iter. (`gemm/ROCmQuantisedGemmKernel_INT8_VNNI.hip:72-83`, `README.vnni-gemm-tuning.md:582-698`)
- **Why gfx906:** 16 KB L1I pollution; documented as closing the remaining 6.6% gap to Composable Kernel.
- **Current state:** `mmq.cuh` uses single loop with boundary checks.
- **Port plan:** Restructure `mmq.cuh` loop into safe/boundary lambdas.
- **Effort:** Medium · **Value:** High (prefill)

### P8 — Software-pipelined decode K-loop with explicit `s_waitcnt`
- **Technique:** Issue next-block loads at *top* of loop so ~120-cycle decode phase hides HBM latency. (`README.native-vnni-isa-analysis.md:421-533`)
- **Why gfx906:** Estimated 1.9–2.4× decode speedup.
- **Current state:** MMVQ interleaves dequant+reduce but isn't explicitly staged.
- **Port plan:** Add staging-register double-buffer + controlled `s_waitcnt`. **Caveat:** VGPR pressure 4→3 waves.
- **Effort:** Medium · **Value:** High

### P9 — HIP graph capture with device-scalar dynamics + pooled/workspace KV scratch
- **Technique:** Record scalars (head/pos) into device buffers; use `hipGraphExecUpdate` instead of re-capture; pooled single-`hipMalloc` KV + workspace scratch. (`backends/rocm/HIPGraphCapture.cpp`, `kvcache/ROCmRingKVCache.h:704-753`, `ROCmBackend.cpp:2534`)
- **Why gfx906:** Decode is launch-bound (RMSNorm ~3.4 µs); `hipMalloc` churn costs 100–500 µs.
- **Current state:** Project has its own graph machinery.
- **Port plan:** Harvest the *dynamic-device-scalar* + pool/workspace pattern into existing graph path.
- **Effort:** Medium · **Value:** High

### P10 — Grouped MoE single-launch + descriptor slab
- **Technique:** One grid (`blockIdx.z = expert`) for all experts; device-resident `d_group_counts/offsets`; weights packed into one contiguous slab via descriptor table. Replaces ~800 per-expert launches/layer. (`moe/ROCmMoEGroupedPrefillKernels.hip:1-30`, `moe/ROCmMoEKernel.h:239-279`, `ROCmWeightPacker.cpp:476-567`)
- **Why gfx906:** Launch overhead dominates prefill; one launch keeps all 60 CUs busy across experts.
- **Current state:** `mmvq.cu` launches per-expert GEMMs.
- **Port plan:** Pack experts into a slab at load; add `z = expert` grid dimension + device grouping.
- **Effort:** High · **Value:** High

### P11 — TurboQuant asymmetric KV compression (K=TQ8, V=TQ4)
- **Technique:** Quantize KV with `__constant__` codebooks, on-device RoPE freqs, coalesced tiled rotation, fused incremental K+V decode. (`kvcache/ROCmTurboQuantKernels.hip`)
- **Why gfx906:** Cuts KV memory ~4–8× (big for MI50's 16 GB); asymmetric K/V precision.
- **Current state:** Project has turbo3 work already; this is a different/broader scheme.
- **Port plan:** Don't port wholesale. Harvest building blocks (constant codebook, on-device rope freq, coalesced tiled rotation, fused K+V) onto existing turbo3.
- **Effort:** High · **Value:** High (if KV compression is a goal)

### P12 — Register-cached GDN recurrence state (scalar-VGPR trick)
- **Technique:** Keep 64-KB/head recurrence state in **individual scalar VGPRs** (not an array) to defeat AMD stack/LDS spill; eliminates ~1190 global state round-trips. (`gdn/ROCmGatedDeltaNetKernels.hip:256-334`)
- **Why gfx906:** State is the bandwidth bottleneck for hybrid models; arrays get spilled.
- **Current state:** GDN chunk kernel exists but may use array/LDS state.
- **Port plan:** Rework GDN state to scalar VGPRs if pushing GDN harder.
- **Effort:** Medium · **Value:** Medium (GDN only)

### P13 — Fused residual-add + RMSNorm register cache
- **Technique:** Pass 1 adds residual and caches values in register array (`kMaxPerThread=16`); pass 2 normalizes from registers — no global re-read. (`ops/ROCmFusedOpsKernels.hip:145-207`)
- **Why gfx906:** Kills a redundant HBM read of the whole hidden state at decode.
- **Current state:** RMSNorm likely re-reads residual.
- **Port plan:** Add register-cache in fused norm kernel. Watch register pressure (`cols<=16384`).
- **Effort:** Low–Medium · **Value:** Medium

### P14 — `silu_fast` with `__builtin_amdgcn_rcpf`; vectorized `float4` elementwise
- **Technique:** `silu = x * __builtin_amdgcn_rcpf(1+exp(-x))` (single `v_rcp_f32` SFU vs IEEE `v_div`); `float4` loads in RMSNorm/ResidualAdd/SwiGLU. (`ops/ROCmSwiGLUKernels.hip:75-79`, `ops/ROCmRMSNormKernels.hip:124-139`)
- **Why gfx906:** ~50 cyc/element saved on SiLU; vectorized loads cut traffic.
- **Current state:** Probably uses `__frcp_rn` / scalar loads.
- **Port plan:** Intrinsic swap + vectorized loads.
- **Effort:** Low · **Value:** Medium

### P15 — TurboQuant building blocks (sub-set of P11)
- `__constant__` codebook upload with idempotent guard (`ROCmTurboQuantKernels.hip:28-56`)
- On-device RoPE freq table (`d_HIP_ROPE_FREQS[64]`, `:36`, `:58-74`)
- Coalesced tiled rotation access pattern (`R[j][tid]` row-major, `:489-635`)
- Single-wavefront Q8_1-style shuffle quant (`ropm...`)
- **Effort:** Low–Medium each · **Value:** Medium (reusable)

---

## 3. Suggested execution order

| Phase | Items | Rationale |
|---|---|---|
| **Phase 1 — quick wins** | P2, P3, P4, P14, P13 | Trivial/Low effort, drop-in, no structural change; validates approach |
| **Phase 2 — decode & IQ** | P1, P5, P6, P8 | Biggest decode/token-rate gains where project is weakest vs Llaminar |
| **Phase 3 — prefill** | P7 | Restructure `mmq.cuh` loop for prefill throughput |
| **Phase 4 — backend** | P9 | Graph capture + pool/workspace; launch-overhead kill |
| **Phase 5 — architectural** | P10, P11/P15, P12 | MoE grouping, KV compression, GDN state — large, dedicated effort |

---

## 4. Cross-cutting gfx906 levers (apply everywhere)
1. **64-lane wavefront** → warp-shuffle reductions compile to DPP/`ds_swizzle` with **0 LDS, 0 barriers**.
2. **SFU fast math** → `__expf`, `__sinf`/`__cosf`, `__builtin_amdgcn_rcpf` over IEEE.
3. **`float4` / packed-`uint32`** vectorized loads.
4. **Tiny static LDS + minimal `s_barrier`**.
5. **Graph capture with device-scalar dynamics** to kill launch overhead.
6. **Coalesced tiled access** for rotation/dequant/gather.
7. **`__constant__` precomputed tables** (codebooks, RoPE freqs) + on-GPU `inv_freq`.

---

## 5. References (Llaminar source)
- Attention: `src/v2/kernels/rocm/attention/ROCmFlashAttentionKernels.hip`
- GDN: `src/v2/kernels/rocm/gdn/ROCmGatedDeltaNetKernels.hip`
- GEMM/quant: `src/v2/kernels/rocm/gemm/*.hip`, `README.vnni-gemm-tuning.md`, `README.native-vnni-isa-analysis.md`
- MoE: `src/v2/kernels/rocm/moe/*.hip`, `ROCmWeightPacker.cpp`
- Ops: `src/v2/kernels/rocm/ops/*.hip`
- KV cache: `src/v2/kernels/rocm/kvcache/*.hip`
- Backend: `src/v2/backends/rocm/HIPGraphCapture.cpp`, `ROCmRingKVCache.h`, `ROCmBackend.cpp`

---
*Generated from analysis of github.com/Llaminar/llaminar vs mx-llama.cpp-gfx906.*
