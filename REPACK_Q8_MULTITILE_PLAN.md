# Repack Q8_0 MMQ — Multi-Tile Variant Plan

## 1. Goal

Make the repacked Q8_0 prefill MMQ **tile-selectable per N** (token count /
ubatch), the way the native `mul_mat_q<Q8_0>` already is, so repack wins across
all shapes instead of only at large N.

Current state (see `repack-gcn.cu` top-of-file `KNOWN LIMITATION`): the Q8_0
prefill kernel `mmq_gemm_q8_0_repacked` uses a **single fixed tile**
`BM=128, BN=128, BK=4` regardless of N. Native instead picks its tile width
per N, which is why repack only matches native at pp2048 and regresses sharply
at pp128 (≈ −39% on Qwen3.5-4B, ≈ −65% at ubatch16 on Qwen3-4B).

## 2. How the native path generates multiple kernels (reference)

Dispatch chain in `mmq.cu` / `mmq.cuh`:

```
ggml_cuda_mul_mat_q            (mmq.cu:77)
  -> ggml_cuda_mul_mat_q_switch_type   (mmq.cu:6)   // picks quant type
       -> mul_mat_q_case<type>          (mmq.cuh:4164) // picks mmq_x (tile width)
            -> launch_mul_mat_q<type, mmq_x>  (mmq.cuh:4051) // instantiates kernel
                 -> mul_mat_q<type, mmq_x, need_check><<<...>>>  (mmq.cuh:4086)
```

**The runtime-chosen dimension is `mmq_x` — the tile width in the N /
output-column direction.** This is what produces the multiple kernel variants
seen in profiling.

`mul_mat_q_case` (mmq.cuh:4164-4278) selects it by minimizing the number of
column tiles for the actual N:

```c
for (int mmq_x = 8; mmq_x <= mmq_x_max && ntiles_x_best > 1; mmq_x += 8) {
    const int ntiles_x = (args.ncols_max + mmq_x - 1) / mmq_x;  // ncols_max == N
    if (ntiles_x < ntiles_x_best) { mmq_x_best = mmq_x; ntiles_x_best = ntiles_x; }
}
```

Then `launch_mul_mat_q` (mmq.cuh:4051) **instantiates a distinct compiled
kernel** `mul_mat_q<type, mmq_x, need_check>` for that `mmq_x`, and sizes the
launch from it:

- `block_dims = (warp_size, nwarps)` — thread block shape (fixed per arch/type)
- `ntx = (N + mmq_x - 1) / mmq_x`, `nty = (M + mmq_y - 1) / mmq_y` — grid
- `nbytes_shared = mmq_get_nbytes_shared<type>(mmq_x, mmq_y, ...)` — LDS size

So each `mmq_x` value is its own template instantiation; the runtime picks one
→ the several `mul_mat_q<Q8_0>` variants in the profiler.

**Concrete values for gfx906 (VEGA20), Q8_0** (verified):

| Parameter | Value | Source |
|---|---|---|
| `nwarps` | `512/64 = 8` → block `(64,8)` = 512 threads | `mmq_get_nwarps_host` (mmq.cuh:309) |
| `mmq_y` (tile height, M dir) | `128` | `get_mmq_y_host` (mmq.cuh:149, AMD non-RDNA1) |
| `mmq_x` range | `8..mmq_x_max`, step 8 | `mul_mat_q_case` loop |
| `mmq_x_max` cap (gfx906) | **64** for `N < 4096`; env-overridable (default 96) for `N >= 4096` | `mmq.cuh:4182-4189` |

Net: for our benchmark ubatches (16…2048, all `< 4096`) the **only variable is
`mmq_x ∈ {8..64}`**, and the BM-side (`mmq_y=128`) is fixed. The native kernel
already matches our `BM=128` and 512-thread block shape.

## 3. Mapping to the repack kernel

Our `mmq_gemm_q8_0_repacked` (repack-gcn.cu:1781) already uses:

- `block = dim3(64, 8)` → 512 threads — matches native `nwarps=8`.
- `BM = 128` — matches native `mmq_y=128`.
- `BN = 64 * TN_` with `TN_=2` → `BN=128` — **this is the analog of native `mmq_x`.**
- `BK = 4` (K per LDS-staging step).

The fixed `TN_=2` is exactly the limitation: only `<false, 2>` is ever
instantiated (dispatch at repack-gcn.cu:2106). To mirror native we must
**instantiate several `TN_` and pick one per N**.

**Granularity constraint:** `tx = 64` column lanes are fixed, so the coarsest
natural step is `BN = 64 * TN_` → `TN_ ∈ {1 → BN=64, 2 → BN=128}`. Native goes
finer (8…64) but that would require sub-warp column masking; out of scope for
v1. For all our ubatches `< 4096`, native caps at `mmq_x <= 64`, so
`BN ∈ {64, 128}` already covers the native target set.

## 4. Implementation plan

### Step A — Parameterize the launch (host side, `ggml_cuda_mul_mat_repacked_slice`)

1. Compute the desired `BN` from `ne11` (N) with the same "minimize column
   tiles" rule, honoring the gfx906 cap:
   - `BN_best = 128` initially; if `ceil(N / 64) < ceil(N / 128)` prefer `64`
     (i.e. `TN_=1` for small N). Mirror native's "largest mmq_x that minimizes
     ntiles" — here largest `BN ∈ {64,128}` minimizing `ceil(N/BN)`.
   - Clamp to 64 for `N < 4096` (matches native; larger BN only when N >= 4096,
     or via an env override for experimentation).
2. `TN_ = BN / 64`, then dispatch through a small switch exactly like native's
   `mmq_x` switch:
   ```c
   switch (TN_) {
       case 1: mmq_gemm_q8_0_repacked<HAS_IDS, 1><<<grid, dim3(64,8),0,stream>>>(...); break;
       case 2: mmq_gemm_q8_0_repacked<HAS_IDS, 2><<<grid, dim3(64,8),0,stream>>>(...); break;
   }
   ```
   Grid: `((ne01 + BM - 1)/BM, (ne11 + BN - 1)/BN, 1)`.
3. **Keep the `ggml_cuda_memcpy_1` LDS reads and `sX_swizzle` unchanged** — they
   are already tile-width agnostic (indexed by `tx`, `kk`, `TN_` loop).

### Step B — (only if A is insufficient) BM / occupancy variants

Native fixes `mmq_y=128` (our `BM=128` already matches), so BM variants are
probably *not* needed initially. If small-N is still slow after A, consider
occupancy (`__launch_bounds__`) or a smaller `BM` for small N.

### Step C — K-quants (defer)

The shared 64×64 tile (`MMQ_RP_BM/BN`, repack-gcn.cu:1056) could get the same
per-N `BN`, but K-quants currently regress less than Q8_0. Out of v1 scope.

## 5. Hypothesis (stated honestly)

- **Large N** (pp2048, ubatch ≥ 256): `BN=128` is already what native picks →
  repack unchanged, stays ≈ native. ✅ expected.
- **Small N** (pp128, ubatch ≤ 128): `BN=64` reduces wasted column work vs the
  fixed `BN=128`, so the gap should **narrow**.

  **Caveat — do not over-promise:** native also uses `mmq_x=64` at small N and
  is *still much faster* than repack today (e.g. Qwen3-4B pp128 ubatch16:
  native 427 vs repack 149 t/s). So `BN` selection alone will likely give only
  **partial** recovery; the remaining gap probably comes from occupancy / K-loop
  efficiency / the fixed `BM`, not tile width. Measure and iterate; don't assume
  A closes it.

## 6. Benchmark plan

- `./SCRIPT_llama_bench.sh repack` already does the 16-point ubatch sweep
  (16…2048). Compare per-ubatch vs the `native` run.
- `./discover_kernels.py repack` to confirm `mmq_gemm_q8_0_repacked` call count
  and GPU time drop at small N relative to native.
- **Success criterion:** repack ≥ native at every ubatch (or regression within
  a few %), especially pp128. Until then, keep `GGML_CUDA_REPACK_Q8_0=0` as the
  safe default on gfx906.

## 7. Risks / open questions

- **LDS budget for `BN=64` (`TN_=1`):** `sX[64][5]` = 64·5·36 B ≈ 11.3 KiB;
  `sW` planes ≈ 18 KiB → ~29 KiB < 64 KiB. Fits at occupancy 1. ✅
- **Below `BN=64`:** needs partial-column / lane-mask handling; not in v1.
- **The matvec path (`ne11 == 1`, decode) is separate** and already has its own
  `ne01`-based size split (repack-gcn.cu:2045); leave it untouched.
- **Upstream alignment:** once `TN_` is selected per N, the logic should live
  next to `mul_mat_q_case` conceptually and reuse `ggml_cuda_get_max_cpy_bytes`
  / `ggml_cuda_memcpy_1` (already done for the LDS reads) for HW-agnosticism.
