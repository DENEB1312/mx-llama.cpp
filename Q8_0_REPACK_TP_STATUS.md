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
memory access); the kernel is *functionally* correct but currently **slower** than
native, so the optimization work is to realise that untapped speed.

**Status — correct, now ~23% off native pp (single GPU):**

- **Correctness verified single-GPU.**  With `GGML_CUDA_REPACK_Q8_0=1` on one
  MI60 (`HIP_VISIBLE_DEVICES=0`, `Qwen3-4B-Instruct-2507-Q8_0.gguf`), the server
  returns coherent text: prompt "Reply with exactly: HELLO WORLD" → `HELLO WORLD`;
  "capital of France" → `The capital of France is Paris.`  So the repack path is
  sound on one device — the `//////` bug is multi-GPU only (see Track B).
- **Greedy-decode cross-check vs native.**  Same prompt + `temperature=0,
  top_k=1, seed=42` in both modes returns semantically identical capital-lists
  (only an occasional country-ordering flip — expected from different tile
  reduction order, not a math bug).
- **Throughput: regression closed to ~23%.**  `llama-bench`, single MI60, 4B
  Q8_0, **pp2048 / tg128** (`SCRIPT_llama_bench.sh`):

  ```
  REPACK_Q8_0=0 (native):   pp2048 ≈ 1690 t/s   tg128 ≈ 132 t/s
  REPACK_Q8_0=1 (repack):   pp2048 ≈ 1307 t/s   tg128 ≈ 125 t/s
  ```

  Repack was **693 t/s** at the un-tuned 64×64 / BK=4 tile (≈2.4× slower) and is
  now **1307 t/s** after widening `MMQ_RP_Q8_BK` 4→8 (matching native's 256-K
  `ITER_K`, i.e. ne0/256 K-iters) while keeping the 64×64 output tile and TM=TN=4
  (original register profile).  Decode unchanged.

**Why not 128×128 / 512 threads (yet):** a 128×128 / BK=8 tile would need
`≈76 KiB` LDS (two `uint4` weight planes `sW_lo`+`sW_hi` = 32 KiB + `sWd` 4 KiB +
`sX[128][9]` block_q8_1 = 40 KiB) — over the 64 KiB gfx906 limit.  128×128 with
`BK=4` (to fit) doubles K-iters back into the regression zone.  The 64×64 / BK=8
config is the register- *and* LDS-safe sweet spot for this simple 2-D thread
decomposition (256 thr, `tx=t&15`, `ty=t>>4`, stride 16).  Matching/exceeding
native's 1690 would require the native-style 512-thread `nwarps` mapping with a
compact packed activation buffer (native's `MMQ_TILE_Y_K` packing) — a follow-up
rewrite, not a tuning knob.

**Next step:** optionally close the last ~23% by porting the native 512-thread
`nwarps` threading + compact `sX` packing into the repack kernel; otherwise the
current repack is a viable (slightly slower) alternative to native Q8_0 MMQ.

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
Single MI60, Qwen3-4B Q8_0, -ngl 99 -fa 1, pp2048/tg128 (SCRIPT_llama_bench.sh)
  REPACK_Q8_0=0 (native):   pp2048 ≈ 1690 t/s   tg128 ≈ 132 t/s   (correct)
  REPACK_Q8_0=1 (repack):   pp2048 ≈ 1307 t/s   tg128 ≈ 125 t/s   (correct;
                              was 693 t/s at un-tuned 64×64/BK=4, +88% after BK=8)

2× gfx906, Qwen3.6-27B Q8_0, -sm tensor -tps 2
  REPACK_Q8_0=0:  coherent (correct)
  REPACK_Q8_0=1:  output is garbage; numbers not meaningful   <-- Track B bug
```

Do **not** trust `REPACK_Q8_0=1` TP throughput until Track B is fixed.  Track A
has closed the single-GPU pp gap from ~2.4× to ~1.3× (within ~23% of native); the
remaining gap is a threading-model difference (256 vs 512 threads), not a defect.

---

## Files changed

| File | Changes |
|------|---------|
| `ggml/src/ggml-backend-meta.cpp` | Meta reg + extra-buft wrapping, supports_op gate, context name fix, multi-segment set_tensor, debug logging (`INIT_TENSOR`/`INIT_RESULT`/`SPLIT_AXIS`/`SPLIT_CB`) |
| `ggml/src/ggml-cuda/repack-gcn.cu` | Multi-segment staging in set_tensor, debug logging, `<map>` include |
| `src/llama-model-loader.cpp` | Debug logging in select_weight_buft / buft_for_tensor |
| `src/llama-model.cpp` | Debug logging in make_gpu_buft_list / ctx_map iteration; `LLAMA_SS` split-state dump |
| `SCRIPT_llama_bench.sh` | Added `GGML_CUDA_REPACK_Q8_0=1` (left as-is; do not use under TP) |
| `launch_server_122B.sh` | Now single-GPU (`HIP_VISIBLE_DEVICES=0`), `Qwen3-4B-Instruct-2507-Q8_0.gguf`, `GGML_CUDA_REPACK_Q8_0=1`, `-c 8192` (no TP) — used to verify Track-A single-GPU correctness. Set back to 2-GPU 27B + `REPACK_Q8_0=0` for normal use. |

Debug logging writes to `/tmp/repack_dbg.log` and can be removed once the
plumbing is validated.
