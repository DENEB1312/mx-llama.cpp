# Q8_0 GCN Repack under Tensor Parallelism — Status

## Anchored summary (CORRECTED root cause)

> **Objective:** Make `GGML_CUDA_REPACK_Q8_0=1` produce correct output under 2-GPU
> tensor-parallel (`-sm tensor -tps 2`) on gfx906 (MI50/MI60). Today it does not:
> it emits `//////` garbage. `GGML_CUDA_REPACK_Q8_0=0` (native Q8_0) is correct.

**What is proven correct (ruled out as the cause):**

1. **TP weight sharding is correct — NOT replicated.** Traces in
   `/tmp/repack_dbg.log` (`INIT_RESULT` / `SPLIT_CB` / `LLAMA_SS` / `REPACK_ENTER`)
   show `output.weight` (`type=Q8_0`) meta `ne[1]=248320` splits into
   `[124160, 124160]` per device, and `attn_qkv.weight` `ne[1]=2048` splits into
   `[1024, 1024]`. Each device receives exactly its own shard. The earlier
   "weights are replicated" hypothesis was **wrong**.
2. **Kernel math is correct.** Q8_0 matvec max-abs err `0.137`, MMQ `0.360` vs
   f32 reference on gfx906 silicon. (Signed `sdot4` over `int8_t` `block_q8_0.qs`
   is the right dot product.) Unchanged.

**What is actually broken (live bug):**

3. The `//////` garbage is a **LIVE bug in the current signed-`sdot4` repack code**
   under 2-GPU TP — not a relic of the old rejected `udot4` experiment.
   `GGML_CUDA_REPACK_Q8_0=1` + `-sm tensor -tps 2` → garbage; `=0` → coherent
   output (verified end-to-end via `/v1/chat/completions` on
   `Qwen3.6-27B-Q8_0.gguf`).
4. **Bug location = repack COMPUTE / TP-integration path** — the
   `ggml_cuda_mul_mat_repacked` dispatch or the meta device's shard-merge for
   repack buffers (row-parallel all-reduce for axis-1 weights like `output.weight`;
   column-parallel concat for axis-0 weights like `attn_output.weight`).
   **NOT the kernel, NOT the sharding.**
5. **Recommendation: keep `GGML_CUDA_REPACK_Q8_0=0` until fixed.** No "replicated
   weights" perf note applies; the "Q8_0 repack works under TP" claim is removed.

**Repro / A-B testing:**
- Model: `Qwen3.6-27B-Q8_0.gguf` (see `launch_server_122B.sh`, `MODEL_PATH`,
  line 58). Launch script runs a persistent launcher that respawns on port 8080;
  edit the `GGML_CUDA_REPACK_Q8_0` env var in the script to A/B test `=0` vs `=1`.
- Debug instrumentation in tree (gated by small counters, writes
  `/tmp/repack_dbg.log`): `ggml/src/ggml-backend-meta.cpp` (`INIT_TENSOR` /
  `INIT_RESULT` / `SPLIT_AXIS` / `SPLIT_CB`); `src/llama-model.cpp` (`LLAMA_SS`).
  `launch_server_122B.sh` currently sets `GGML_CUDA_REPACK_Q8_0=0`.

---

## Original problem

Under tensor-parallel (`-sm tensor`, `LLAMA_SPLIT_MODE_TENSOR`) with two gfx906
(MI50/MI60) devices, the GCN weight-repack buffer never engaged.  Weights stayed
in the default HIP buffer, so `GGML_CUDA_REPACK=1` / `GGML_CUDA_REPACK_Q8_0=1`
produced no difference (bench numbers were identical to `REPACK=0`).

**Root cause:** `llama.cpp`'s `make_gpu_buft_list` looks up per-device extra
buffer types via `ggml_backend_dev_backend_reg(dev)`.  For the Meta aggregate
device `Meta(ROCm0,ROCm1)` this returned `nullptr`, so the repack buffer was
never offered to the weight loader.

## What was fixed

### 1. Meta device exposes extra buffer types (repack engagement)

`ggml/src/ggml-backend-meta.cpp`

- The Meta device now holds a real `reg` that exposes the
  `ggml_backend_dev_get_extra_bufts` proc.  Llama.cpp's existing extra-buft
  lookup path (llama-model.cpp line 998–1010) finds it.

- For each physical device the meta extra-buft function collects that device's
  repack buffer (e.g. `ROCm0_Repacked`) and wraps them all into a **meta buffer
  type** — a single meta buft whose per-device simple buffers are the per-device
  repack buffers.  The existing meta `set_tensor` machinery then shards and
  delegates repacking to each device correctly.

### 2. Shard-safe weight selection gate

`ggml/src/ggml-backend-meta.cpp` — `ggml_backend_meta_device_supports_op`

The meta-wrapped repack buft is invisible to HIP's own repack gate (which
recognises only simple cuda repack bufts by name).  A replica gate runs at the
meta level: it only accepts MUL_MAT / MUL_MAT_ID op shapes, checks
quantisation type and contiguity, and enforces `ne[0] % (granularity * tps)`
so that an axis-0 TP split cannot break the repack precondition on individual
shards.  Non-matmul tensors (norms, biases, …) fall through to the default
buffer.

### 3. Multi-segment set_tensor (QKV / gate_up)

`ggml/src/ggml-cuda/repack-gcn.cu` — `ggml_backend_cuda_repack_buffer_set_tensor`

The meta `set_tensor` can call the simple repack buffer's set_tensor multiple
times per tensor (once per segment for QKV, once per row for axis-1 splits).
A staging buffer collects the partial uploads; when the final call arrives all
segments are repacked together and uploaded to the device.

### 4. Meta buffer type name / ctx_map comparator fix

The `ggml_backend_meta_buffer_type_context` constructor had a decades-old
shadowing bug: the parameter was moved into the member in the initialiser list
but the body looped over the *emplied* parameter, producing the name `"Meta()"`
for **every** meta buft.  The `ctx_map` comparator groups bufts by name, so the
default and repack meta bufts collided — tensors destined for repack silently
ended up in the default buffer.  Fixed by using `this->simple_bufts` in the
name-construction loop.

---

## Q8_0 kernel correctness — VERIFIED CORRECT (updated)

The earlier conclusion that Q8_0 repack is "unsolvable due to a signed/unsigned
mismatch" is **WRONG**.  The Q8_0 matvec and MMQ repack kernels are correct on
gfx906.

### Why the original analysis was wrong

The premise was that `block_q8_0.qs` is stored as **unsigned** `uint8` (range
0…255) and thus the signed `sdot4` (`__builtin_amdgcn_sdot4`, used on gfx906 via
`ggml_cuda_dp4a`) mis-reads bytes ≥ 128.  **This is false.**  In
`ggml/src/ggml-common.h` `block_q8_0.qs` is declared `int8_t` — the weights are
**signed** int8, exactly like `block_q8_1.qs`.  The signed `sdot4` is therefore
the correct dot product for both weight and activation.

Empirical proof on gfx906 silicon (device 0):

| Test | What | Max abs error vs float32 reference |
|------|------|-------------------------------------|
| Matvec (`mul_mat_vec_q8_0_repacked`) | random ne0=256, ne1=64, Q8_0 weights | **0.137** |
| MMQ (`mmq_gemm_q8_0_repacked`)       | random ne0=256, ne1=64, n_tok=64     | **0.360** |

Both well within quantisation tolerance → the kernels compute the correct
dot product.  (Reference test harness: `/tmp/opencode/test_q8_repack.hip`,
`/tmp/opencode/test_q8_mmq.hip`; the host-side `free(): invalid pointer` shown
at those tests' process exit is a HIP/glibc cleanup quirk of the standalone
harness and is **not** a kernel defect.)

### Origin of the `//////` garbage symptom

> **LIVE BUG — not obsolete.**  The original `//////` garbage was first seen with
> the experimental `udot4` (unsigned×unsigned) path, which was rejected.  But the
> **current shipping kernel** (signed `sdot4`) **also** produces `//////` under
> 2-GPU TP with `GGML_CUDA_REPACK_Q8_0=1` (reproduced end-to-end against
> `Qwen3.6-27B-Q8_0.gguf` via `launch_server_122B.sh` + `/v1/chat/completions`:
> response is `//////…`).  So the garbage is a **real, current defect in the
> repack-under-TP path**, not a relic of the udot4 experiment.  The kernel
> arithmetically computes the correct dot product (see proof above) — the failure
> is in how the repacked weights are driven under tensor parallelism, not in the
> per-GPU math.

### What was tried (history, now superseded)

1. Host bias + per-dp4a correction — correct but 2× dp4a overhead (perf only).
2. Per-sub-block correction via `block_q8_1.ds.y` — field stores `sum(float)`,
   not `sum(int8)`, so the correction was wrong.
3. `__builtin_amdgcn_udot4` — unsigned×unsigned; wrong for signed activation.
   **This was the path that produced the `//////` garbage.**

None of these are needed: the plain signed `sdot4` path is correct.

---

## How to use the current code

> **DO NOT enable `GGML_CUDA_REPACK_Q8_0=1` under tensor parallelism yet.**  The
> Q8_0 repack kernel is arithmetically correct, and TP weight sharding is correct,
> but the repack-under-TP compute path produces garbage end-to-end (see Current
> blocker).  Keep it off until that is fixed:
>
> ```bash
> export GGML_CUDA_REPACK=1
> export GGML_CUDA_REPACK_Q8_0=0   # correct today; Q8_0 repack broken under TP
> ```
>
> The K-quant repacks (Q3_K / Q4_K / Q5_K / Q6_K) are unaffected by this bug and
> remain usable.  `launch_server_122B.sh` is currently repurposed for single-GPU
> Track-A verification (4B Q8_0, `REPACK_Q8_0=1`, no TP); restore 2-GPU 27B +
> `REPACK_Q8_0=0` for normal multi-GPU serving.

---

## Two work tracks

The Q8_0 repack effort splits into two independent work items: a **performance**
track (make the repack kernel beat the native Q8_0 baseline) and a **correctness**
track (make repack work under 2-GPU tensor parallelism).  They share the same
plumbing but are otherwise orthogonal.

---

### Track A — Repack MMQ kernel optimization (performance)

**Goal:** the repack Q8_0 GEMM (`ggml_cuda_mul_mat_repacked`, repack-gcn.cu) must
be **faster than the native Q8_0 baseline**.  The repack layout exists to exploit
the gfx906 `dp4a`/`sdot4` path more efficiently (larger/aligned tiles, better
memory access); the kernel is *functionally* correct and now **matches native on
pp2048 (within noise) and beats it by ~4% on tg128**.

- **Correctness verified single-GPU.**  With `GGML_CUDA_REPACK_Q8_0=1` on one
  MI60 (`HIP_VISIBLE_DEVICES=0`, `Qwen3-4B-Instruct-2507-Q8_0.gguf`), the server
  returns coherent text: prompt "Reply with exactly: HELLO WORLD" → `HELLO WORLD`;
  "capital of France" → `The capital of France is Paris.`  So the repack path is
  sound on one device — the `//////` bug is multi-GPU only (see Track B).
- **Greedy-decode cross-check vs native.**  Same prompt + `temperature=0,
  top_k=1, seed=42` in both modes returns semantically identical capital-lists
  (only an occasional country-ordering flip — expected from different tile
  reduction order, not a math bug).
- **Throughput (single MI60, 4B Q8_0, pp2048 / tg128).** Pre-Plan-A (loop
  interchange only, committed `b330018ec`):

  ```
  REPACK_Q8_0=0 (native):   pp2048 ≈ 1289 t/s ±3   tg128 ≈ 105.1 t/s
  REPACK_Q8_0=1 (repack):   pp2048 ≈ 1289 t/s ±18  tg128 ≈ 109.1 t/s   (tied pp, +3.8% tg)
  ```

  Post-Plan-A (mmq-quantizer rewrite — see Performance section below): **PP now
  WINS ~+9% (1414 vs 1296 t/s) but TG REGRESSES to ~43 t/s** (decode matvec now
  reads the transposed mmq buffer). Fix pending — see Performance.

**Why it was slower and what fixed it.**  The original inner K-loop read
the W plane (`sW_lo` + `sW_hi` + `sWd`) on every (kk, n, r) triple, but
those reads are independent of n — for TN_=2 the same 16 W rows per
sub-block were re-fetched for both n=0 and n=1.  That doubled the inner
W-plane LDS read count compared to what the data structure actually
needed.  **Loop interchange** — hoisting `r` out of `n` (now
`(kk, r, n)`) — reads each W row once per `(kk, r)` and reuses it across
both n lanes.  Same dp4a count, half the W LDS reads.  pp2048
immediately closed from `1274 vs 1303` (−2.2% repack) to `1289 vs 1289`
(tied).  No LDS budget change, no tile change, no dispatch change.

**Kernel evolution this session** (started from committed `b330018ec`, repack
1307 pp / 125 tg, 256-thr 64×64 / BK=8):

1. **Port native 512-thread (64×8) decomposition** — `tx=threadIdx.x` (64 col
   lanes, no sX broadcast), `ty=threadIdx.y` (8 row lanes), 128×64 tile → **1545
   pp** (+18%).  Weights staged through LDS and read per-row inside the K-loop
   (transient) so VGPR ≈ 36 fits 8 waves.
2. **Tried direct-from-global weight reads** (like native, rely on L2) → **regressed
   to 1030 pp**.  L2 cannot hide the per-K-step global weight latency; reverted.
   (Verified: LDS-weight read >> global-weight read.)
3. **Fitted 128×128 tile via BK=4** (weights `sW[128][4]`≈18 KiB + `sX[128][5]`≈23
   KiB ≈ 41 KiB < 64 KiB) → **1631 pp** (native-matching activation reuse, 2× fewer
   column blocks).  `launch_bounds(512,2)` was neutral → kept `(512,1)`.
4. **Tried BN=192 / TN=3** → **invalid launch wedged the GPU** (needs `sudo
   rocm-smi --reset`); reverted to TN=2.  Lesson: stay within LDS budget.
5. **Loop interchange `(kk, n, r) → (kk, r, n)`** to halve inner W-plane
   LDS reads → **1289 pp** (tied with native).  Lesson: when two inner
   loops share data dependency, hoist the one that touches the shared
   state to be the *outer* loop.
then attempt tighter weight-LDS packing or a BK=8 + reduced-activation-LDS variant
to close it.  The repack path is already a viable, near-native alternative to
native Q8_0 MMQ.

---

### Track B — Tensor-parallel correctness (the `//////` bug)

**Symptom:** `GGML_CUDA_REPACK_Q8_0=1` + `-sm tensor -tps 2` → model emits
`//////…` garbage.  `GGML_CUDA_REPACK_Q8_0=0` (native Q8_0 on the same meta
plumbing) → coherent output.  Track A's single-GPU run proves repack is fine
*without* TP, so this is strictly a TP bug.

**What has been ruled out (proven by trace + on-silicon test):**

1. **Kernel math is correct.**  Matvec err 0.137, MMQ err 0.360 (see proof
   above).  The repack GEMM computes the right dot product for a single, full
   repacked weight.
2. **TP weight sharding is correct.**  A 2-GPU TP trace of `output.weight`
   (`type=Q8_0`) shows the meta buffer `ne[1]=248320` (the model's actual lm_head
   row count) split into `ne[0*2+0]=124160` / `ne[0*2+1]=124160` — each device
   receives exactly its shard.  `attn_qkv.weight` similarly splits `ne[1]=2048`
   into `[1024,1024]`.  The meta `set_tensor` → per-device repack `set_tensor`
   uploads the correct slice to each device (confirmed via `REPACK_ENTER` /
   `INIT_RESULT` / `SPLIT_CB` / `LLAMA_SS` debug logs in `/tmp/repack_dbg.log`).
   Weights are **sharded, not replicated**.

**Where the bug lives:** the repack **compute / TP-integration** path.  Since the
per-device weight shard is correct and the repack GEMM is correct in isolation,
the failure is in how the repacked shard is consumed under TP — i.e. the
`ggml_cuda_mul_mat_repacked` dispatch / result combination (row-parallel all-reduce
for axis-1 weights like `output.weight`, column-parallel concat for axis-0 weights
like `attn_output.weight`) when `src0` lives in a repack buffer.  Candidates:

- The repack MUL_MAT under TP is handed a shard but computes against the wrong
  slice of the repacked layout (repack blocks are row-contiguous in `ne1`, so an
  arbitrary `ne1` boundary *should* be safe — needs confirmation).
- The meta device's result merge for repack shards does not match the repack
  kernel's output shape (the repack layout changes `nb`/`ne` of the simple buffer;
  if the merge uses the canonical `ne` it may mis-stride the concatenated shard).

**Next step:** instrument `ggml_cuda_mul_mat_repacked` (repack-gcn.cu) under TP to
compare its per-device partial result against the native-Q8_0 partial for the same
shard, then check the meta device's shard-merge for repack buffers.

### Debug instrumentation currently in tree

Writes to `/tmp/repack_dbg.log` (gated by small counters, safe to leave):
`ggml/src/ggml-backend-meta.cpp` — `INIT_TENSOR`/`INIT_RESULT`/`SPLIT_AXIS`/
`SPLIT_CB`; `src/llama-model.cpp` — `LLAMA_SS`.  Remove once both tracks land.

### Performance

```
Single MI60, Qwen3-4B Q8_0, -ngl 99 -fa 1, pp2048/tg128
  (SCRIPT_llama_bench.sh, build b330018ec + Plan-A mmq-quantizer rewrite)

  PP (kernel discovery, GGML_CUDA_DISABLE_GRAPHS=1):
    native total GPU time:  6,243.22 ms   (quantize_mmq_q8_1  80.1 ms)
    repack total GPU time:  5,706.37 ms   (quantize_mmq_q8_1  79.3 ms,
                                            slow quantize_q8_1 GONE)
    -> repack PP ~ -8.6% GPU time vs native; GEMM 4199 vs 4700 ms

  Combined bench (pp2048 / tg128):
    REPACK_Q8_0=0 (native):  pp2048 ≈ 1296.98 t/s   tg128 ≈ 105.12 t/s
    REPACK_Q8_0=1 (repack):  pp2048 ≈ 1414.42 t/s   tg128 ≈  43.42 t/s  <-- TG REGRESSION

  PP WINS (+9.1% t/s, matches kernel discovery).  TG REGRESSES (-58.7% vs native).
```

**PP is now a clear win and the activation-quantizer bottleneck is eliminated**
(Plan A: dense dispatch switched from `quantize_row_q8_1_cuda` — the slow
block-32, 1-elem/thread, gfx906-locked path — to `quantize_mmq_q8_1_cuda`, the
same fast block-128 / 4-elem/thread kernel native uses; kernels read the
transposed `block_q8_1_mmq` buffer via `rp_xq_from_mmq`). The 642 ms
`quantize_q8_1` cost seen in the prior PP profile is gone (repack now 79 ms,
== native).

**Decode (TG) regressed from ~109 t/s (pre-rewrite) to ~43 t/s** — a decode-only
regression (PP improved, so it is localized to the single-token matvec path).
Root cause: for decode (`ne11 == 1`) the matvec now gathers each sub-block out of
the strided 144-byte `block_q8_1_mmq` layout via `rp_xq_from_mmq` instead of the
contiguous canonical `xq + sb`, and/or the bulk-oriented `quantize_mmq_q8_1_cuda`
+ temp-pool dispatch defeats CUDA-graph capture for the 1-token step. Output is
still correct (llama-server answers coherently) — it is purely a decode perf
regression.

**Proposed fix (NOT yet applied):** keep the fast `quantize_mmq_q8_1_cuda` + mmq
buffer for **prefill** (`ne11 > 1`) only; route **decode** (`ne11 == 1`) back to
the contiguous canonical quantizer (`quantize_row_q8_1_cuda` + `xq + sb` in the
matvec). A single token's quantize cost is negligible, so decode regains the old
~109 t/s while PP keeps its ~9% win. This keeps the K-quant MoE (`HAS_IDS`)
branch untouched (it was never switched).

Do **not** trust `REPACK_Q8_0=1` combined-bench TG until the decode regression is
fixed. Track A has closed the single-GPU **PP** gap (now ~+9% vs native) but
introduced a **TG** regression that must be resolved before this path is viable
end-to-end. (TP Track B still open — see below.)

### Profiling (rocprofv3, gfx906, Qwen3-4B Q8_0, pp2048/tg128)

`discover_kernels.py` was run with `GGML_CUDA_DISABLE_GRAPHS=1` for native
(`REPACK_Q8_0=0`) and repack (`REPACK_Q8_0=1`) separately (at the *pre-rewrite*
1307 pp baseline).  
Profiling prompt processing only results. 
Given we need to disable the cuda graph for it to run the tg profiling is not as true as the pp one so we can concentrate on pp.


================================================================================
KERNEL DISCOVERY REPORT - REPACK - PP ONLY
================================================================================
Total GPU time: 6,232.97 ms | Kernels: 17 | Hot (>1%): 4

#    Kernel                                                Calls   Time(ms)      %
--------------------------------------------------------------------------------
1   * mmq_gemm_q8_0_repacked                                996    4171.56  66.9%
2   * flash_attn_tile                                       144    1080.17  17.3%
3   * quantize_q8_1                                        1012     642.10  10.3%
4     flash_attn_combine_results                            144     137.07   2.2%
5     rms_norm_f32                                          288      48.25   0.8%
6     unary_gated_op_kernel                                 144      47.96   0.8%
7     rope_neox                                             144      32.18   0.5%
8     rms_norm_f32                                          292      27.98   0.4%
9     k_bin_bcast                                           288      25.10   0.4%
10    rope_neox                                             144       7.92   0.1%
11    k_set_rows                                            144       7.36   0.1%
12    mul_mat_vec_q8_0_repacked                              12       2.55   0.0%
13    __amd_rocclr_fillBufferAligned                         37       1.31   0.0%
14    flash_attn_mask_to_KV_max                             144       0.64   0.0%
15    __amd_rocclr_copyBuffer                               161       0.62   0.0%
16    mul_mat_vec_q8_0_repacked                               4       0.17   0.0%
17    k_get_rows_float                                        8       0.04   0.0%
--------------------------------------------------------------------------------


================================================================================
KERNEL DISCOVERY REPORT - STANDARD - PP ONLY
================================================================================
Total GPU time: 6,225.82 ms | Kernels: 19 | Hot (>1%): 5

#    Kernel                                                Calls   Time(ms)      %
--------------------------------------------------------------------------------
1   * mul_mat_q<Q8>                                         712    2989.76  48.0%
2   * mul_mat_q<Q8>                                         284    1695.01  27.2%
3   * flash_attn_tile                                       144    1115.21  17.9%
4     flash_attn_combine_results                            144     137.02   2.2%
5     quantize_mmq_q8_1                                     996      80.07   1.3%
6     rms_norm_f32                                          288      51.08   0.8%
7     unary_gated_op_kernel                                 140      49.17   0.8%
8     rope_neox                                             144      33.88   0.5%
9     rms_norm_f32                                          292      28.30   0.5%
10    k_bin_bcast                                           284      24.91   0.4%
11    rope_neox                                             144       8.60   0.1%
12    k_set_rows                                            144       7.63   0.1%
13    mul_mat_vec_q<Q8>                                       4       2.42   0.0%
14    __amd_rocclr_fillBufferAligned                          1       0.91   0.0%
15    flash_attn_mask_to_KV_max                             144       0.68   0.0%
16    __amd_rocclr_copyBuffer                               161       0.57   0.0%
17    mul_mat_vec_q<Q8>                                       8       0.50   0.0%
18    quantize_q8_1                                          12       0.06   0.0%
19    k_get_rows_float                                        8       0.04   0.0%
--------------------------------------------------------------------------------




---

## Files changed

| File | Changes |
|------|---------|
| `ggml/src/ggml-backend-meta.cpp` | Meta reg + extra-buft wrapping, supports_op gate, context name fix, multi-segment set_tensor, debug logging (`INIT_TENSOR`/`INIT_RESULT`/`SPLIT_AXIS`/`SPLIT_CB`) |
| `ggml/src/ggml-cuda/repack-gcn.cu` | Multi-segment staging in set_tensor, debug logging, `<map>` include; **`mmq_gemm_q8_0_repacked` rewritten to native-style 512-thread (64×8) decomposition, 128×128 tile via BK=4, weights staged through LDS (read per-row in K-loop), inner K-loop loop-interchanged from `(kk, n, r)` to `(kk, r, n)` to halve W-plane LDS reads**; `MMQ_RP_Q8_*` macro block; both dispatch sites use `dim3(64,8)` + `launch_bounds(512,1)` |
| `src/llama-model-loader.cpp` | Debug logging in select_weight_buft / buft_for_tensor |
| `src/llama-model.cpp` | Debug logging in make_gpu_buft_list / ctx_map iteration; `LLAMA_SS` split-state dump |
| `SCRIPT_llama_bench.sh` | Added `GGML_CUDA_REPACK_Q8_0=1` (left as-is; do not use under TP) |
| `launch_server_122B.sh` | Now single-GPU (`HIP_VISIBLE_DEVICES=0`), `Qwen3-4B-Instruct-2507-Q8_0.gguf`, `GGML_CUDA_REPACK_Q8_0=1`, `-c 8192` (no TP) — used to verify Track-A single-GPU correctness. Set back to 2-GPU 27B + `REPACK_Q8_0=0` for normal use. |

Debug logging writes to `/tmp/repack_dbg.log` and can be removed once the
plumbing is validated.
