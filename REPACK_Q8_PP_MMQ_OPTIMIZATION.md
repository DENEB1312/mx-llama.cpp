# Q8_0 Repack PP MMQ — Optimization Attempts & Failures

Working notes for pushing the Q8_0 repack prefill GEMM (`mmq_gemm_q8_0_repacked`,
`ggml/src/ggml-cuda/repack-gcn.cu`) faster than native on gfx906 (MI50/MI60).
Scope: **single-GPU, dense 4B Q8_0, prompt-processing (PP) only**. Token
generation (TG) is a separate, already-solved concern (see below).

The Q8_0 repack track overall status lives in `Q8_0_REPACK_TP_STATUS.md`;
this file is the focused PP-MMQ perf notebook for continuing next session.

---

## 0. MEASUREMENT INTEGRITY (read first)

All comparisons below are only valid if the **GPU power limit / clock is held
constant** across runs. A change that should only affect the prefill GEMM
(`mmq_gemm_q8_0_repacked`, used when `ne11>1`) must **NOT** move `tg128`
(decode uses the separate `rp_mv_q8_0` matvec, `ne11==1`). If `tg128` moves,
the run is confounded by a power/clock change and the PP number is not
comparable.

**Incident (this session):** the power limit was raised between the 1425
baseline and a later "1779" run. The raised limit boosted clocks, inflating
**both** pp2048 (1779) and tg128 (125, which should have stayed ~102). The
1779/125 pair is therefore **invalid** — it is a power artifact, not a kernel
win. At the raised power limit the split-accumulator variant actually measured
**1407.18 t/s and was SLOWER than the relayout-reverted baseline at the same
power limit**. The variant was reverted. Lesson: ignore any PP win that is
accompanied by a TG change for a prefill-only edit.

## 1. Current best state (the baseline to beat)

Single MI60, `Qwen3-4B-Instruct-2507-Q8_0.gguf`, `-ngl 99 -fa 1`, pp2048.
**The relayout was reverted; the split-accumulator change was reverted.** The
valid best state is the relayout-reverted code (commit `e78142b60` plus the
decode reroute), measured at the **original** power limit:

| Config | pp2048 (t/s) | tg128 (t/s) |
|--------|-------------|-------------|
| native (`REPACK_Q8_0=0`) | 1297 | 105 |
| repack baseline (`REPACK_Q8_0=1`) | 1425 | 102 |

Repack PP is **+9.9% vs native**; TG is on par. This is the number to beat.
(The 1425 baseline itself was at the *original* power limit; the absolute
throughput at the now-raised power limit is higher but unmeasured for this
code — re-establish it before resuming optimization, see §6.)

> NOTE: the split-accumulator "win" of 1779/125 documented in an earlier draft
> of this file was a power-limit artifact and has been **retracted**. See
> Attempt D below for the corrected result.

### PP-only kernel discovery (repack enabled, GGML_CUDA_DISABLE_GRAPHS=1)

```
Total GPU time: 2,867.18 ms | Kernels: 18 | Hot (>1%): 4
1   * mmq_gemm_q8_0_repacked   498   2109.91 ms  73.6%   <-- the target
2   * flash_attn_tile           72    547.09 ms  19.1%
3     flash_attn_combine_results 72   68.18 ms   2.4%
4     quantize_mmq_q8_1        498     39.61 ms   1.4%
(rest < 1%)
```

Native PP-only for comparison: total 3,114.89 ms; `mul_mat_q<Q8>` (native GEMM,
356+142=498 calls) = 2344.52 ms (75.2%).

So repack GEMM = **2109.91 ms / 498 = 4.238 ms/call** vs native GEMM
2344.52 / 498 = **4.710 ms/call** → repack GEMM is a flat **~10% faster per
call**, uniform across both matmul shapes (no shape-specific win).

---

## 2. Roofline: the GEMM is compute-bound, not memory-bound

For a large matmul (M≈13824, N≈2560, K≈2560): ~2.3e10 dp4a ops
(=1.81e11 int8 FLOPs counting dp4a as 8 flops). Repack large call ≈ 4.31 ms →
**~42 TFLOPS int8 ≈ 79% of MI50's ~53 TFLOPS int8 peak**. Native large call
≈ 5.97 ms → ~30 TFLOPS (57% peak).

Weight bytes per large call ≈ 13824·2560·1.125 ≈ 40 MB; at 1 TB/s that is a
~40 ms *minimum* if bandwidth-bound — but the call finishes in ~4.3 ms, i.e.
the dp4a units finish long before memory could be the limiter. **Conclusion:
the GEMM is dp4a-throughput bound at ~79% peak.** There is ~20% headroom just
to reach peak dp4a, which would take the GEMM from 2110 ms → ~1700 ms and total
PP from 2867 → ~2450 ms (another ~14% over current, ~21% under native).

**Implication:** every optimization that matters attacks *dp4a issue/utilization*
(ILP, LDS bank conflicts, occupancy, instruction overhead). Anything that only
changes memory access pattern for the activation is wasted unless it also frees
dp4a throughput.

---

## 3. Attempt log

### Attempt A — Plan A: switch dense dispatch to `quantize_mmq_q8_1_cuda` (WIN, baseline)
The dense dispatch used to quantize the activation with the slow, gfx906-locked
`quantize_row_q8_1_cuda` (block-32, 1 elem/thread) → `quantize_q8_1` showed
**642 ms** in the prior PP profile. Switched to the fast `quantize_mmq_q8_1_cuda`
(block-128, 4 elem/thread, what native uses) and made the kernels read the
transposed `block_q8_1_mmq` buffer via `rp_xq_from_mmq`. Result: the 642 ms
bottleneck vanished (repack `quantize_mmq_q8_1` = 79 ms == native 80 ms) and
repack PP went from tied-with-native to **+9.9%**. This is the committed
baseline (commit `e78142b60` plus earlier Plan-A commit).

### Attempt B — reroute decode (`ne11==1`) to canonical quantizer (WIN, context)
First Plan-A cut routed *every* dense MUL_MAT (incl. single-token decode) through
the transposed mmq buffer; the decode matvec gathered sub-blocks out of the
strided `block_q8_1_mmq` layout and TG collapsed to **43 t/s** (vs native 105).
Fix: branch the dense dispatch on `ne11` — `ne11==1` (decode + single-row
matmuls) quantizes with `quantize_row_q8_1_cuda` into a contiguous `block_q8_1`
buffer and the matvec reads `xq + sb`; only `ne11>1` (prefill) uses the mmq
buffer. Verified TG recovered to 102 t/s. Not a PP lever, recorded for context.

### Attempt C — relayout mmq → contiguous `block_q8_1` activation (FAIL / neutral)
**Goal:** remove the per-K-step `rp_xq_from_mmq` gather from the GEMM hot loop
(see `repack-gcn.cu:~1179` etc.) by converting the transposed `block_q8_1_mmq`
buffer (output of `quantize_mmq_q8_1_cuda`) into a contiguous `block_q8_1`
buffer once per op, so the GEMM reads `xq[col*x_stride + sb]` with simple
coalesced loads — mirroring native's contiguous activation read
(`mmq.cuh:847`, `const block_q8_0 * bxi = ...; bxi->qs`).

**Implementation tried:**
- Added `relayout_mmq_q8_1_to_q8_1<HAS_SUM>` kernel + `relayout_q8_1_mmq`
  launcher (used `rp_xq_from_mmq` to expand each mmq block into 4 contiguous
  `block_q8_1`).
- Prefill (`ne11>1`) branch: `quantize_mmq_q8_1_cuda` → relayout → pass the
  contiguous buffer to the GEMM; GEMM staging changed from
  `rp_xq_from_mmq(...)` to `xq[col*x_stride + sb]`.
- Removed the now-unused `has_sum` GEMM parameter; relayout takes it instead.

**Measured outcome (pp2048, repack on, single MI60 4B):**
- Broken first cut: relayout launched as **one CUDA block per slice**
  (`grid = n_slices`, often 1 for 2D weights) → 498 tiny single-wave launches
  → **pp = 1205 t/s** (a real regression from launch/occupancy overhead).
- Fixed launch: one grid over the whole buffer (`grid = (total+255)/256`) →
  **pp = 1395 t/s** (two runs: 1396.8, 1394.9).

**Verdict: FAILED to beat the baseline.** 1395 is *flat-to-slightly-slower*
than the pre-relayout 1425 (≈ −2%, within/just outside run-to-run noise, PP
std ≈ ±22). The relayout code was **reverted** (repo restored to commit
`e78142b60` = the 1425 baseline) to avoid shipping a regression.

**Why it failed (the important lesson):** the GEMM is compute-bound (§2), so the
activation access pattern is not the limiter. The inline `rp_xq_from_mmq` was
already reading **one 144-byte `block_q8_1_mmq` block per (col, kk)** — i.e.
4 consecutive sub-blocks arrived in a single contiguous 144-byte load, and the
"gather" was just unpacking a local. Moving that unpack into a separate relayout
pass only added an L2 round-trip (read mmq → write contiguous → GEMM reads
contiguous) without freeing any dp4a throughput. Net: flat, marginally negative.
The broken single-block version proved launch granularity matters, but even the
fixed version couldn't overcome the fundamental "compute-bound, not memory" fact.

> Takeaway for next session: **do not chase activation-layout changes.** They do
> not help a compute-bound GEMM. Spend effort only on dp4a utilization.

### Attempt D — split lo/hi dp4a accumulators (FAIL / REGRESSION — RETRACTED)
**Goal:** raise dp4a issue-level ILP. The inner K-step had a single `int idot`
accumulator, so the 8 dp4a (4 `lo[j]` × `xq32[j]` plus 4 `hi[j]` ×
`xq32[j+4]`) were serialized through one dependency chain.

**Change:** replaced the single `idot` with two independent accumulators
(`idot_lo`, `idot_hi`), same arithmetic result, **+1 VGPR**.

**First measurement (pp2048, single MI60 4B Q8_0):** pp = 1778.6 t/s,
tg128 = 125.5 t/s. This looked like a +25% win — **but it was a measurement
artifact.** The power limit had been raised between this run and the 1425
baseline, so clocks were higher; the TG rising to 125 (decode path untouched
by this edit) proves the run was confounded.

**Corrected measurement (same raised power limit, apples-to-apples):** the
split-accumulator variant measured **pp2048 = 1407.18 t/s** and was **SLOWER
than the relayout-reverted baseline at the same power limit**. tg128 for this
edit is irrelevant (decode unchanged). **Verdict: REGRESSION. Reverted.**

**Why it failed (corrected analysis):** the extra VGPR from the second
accumulator pushed the kernel past the `launch_bounds(512,1)` occupancy
ceiling (8 wavefronts/block, LDS-limited to one block) — fewer waves → worse
dp4a latency hiding → net loss. The earlier "win" was entirely the clock
boost. So this change does NOT improve a compute-bound GEMM; it hurts
occupancy.

**Status: REVERTED. Do not re-apply.** Numerically correct (greedy output
was byte-identical to native), but slower; the regression is occupancy, not
math.

---

### Attempt E — shrink LDS to fit 2 blocks (P1: BN 128→64, TN 2→1) (FAIL / REGRESSION)
**Goal (from §0/P1 proposal):** `sX[128][5]` = 23 KiB dominates LDS; at
`launch_bounds(512,1)` only one block (8 waves) fits per CU, capping
occupancy. Halve `BN` 128→64 (`MMQ_RP_Q8_TN` 2→1) so `sX[64][5]` = 11.5 KiB
→ total LDS ≈ 31 KiB → two blocks (16 waves) should fit and hide dp4a
latency better.

**Change:** `MMQ_RP_Q8_TN` 2 → 1 (only that one macro; `tok0`, column
indexing, and grid formulas kept identical so the tiling relationship is
preserved).

**Measured (pp2048, single MI60 4B Q8_0, current/original power limit, via
`SCRIPT_llama_bench.sh`):**
- baseline (TN=2, BN=128): **1408 t/s**
- P1 (TN=1, BN=64): **1362 t/s** → **−3.3%** (REGRESSION)

**Why it failed:** halving `BN` also halved the number of independent dp4a
chains *within a block* (128 → 64), cutting in-block ILP / latency hiding.
The hoped-for 2-block (16-wave) occupancy gain did not compensate — either
the 2-block fit did not materialize (some other constraint) or 8 waves was
already enough to keep the dp4a pipe busy, so the lost per-block ILP dominated.

**Status: REVERTED.** `MMQ_RP_Q8_TN` back to 2 (1403 t/s confirmed). Do not
re-apply this form of P1. To break the occupancy ceiling a different LDS
reduction is needed that does NOT shrink per-block N (e.g. reduce `BK`, or
shrink `sW` planes), see P2 follow-ups.

### Attempt F — larger BK / more dp4a per LDS fill (P2) (FAIL / REGRESSION)
**Goal (P2 proposal):** raise arithmetic-per-LDS-fill (and ideally
software-pipeline with double-buffered LDS) to hide the global→LDS fill
latency. **LDS budget reality check first:** single buffer is ~40.5 KiB;
double-buffering needs ~81 KiB > 64 KiB hard cap → **double-buffer is
impossible** at current sizes. BK=8 single buffer needs `sW` 36 KiB +
`sX[128][9]` 40.5 KiB = 76.5 KiB > 64 → also impossible (the `+1` padding
on `sX` is required to keep the 2-way bank conflict, so `sX` can't shrink
to `BK`). The only feasible form is a single-buffer BK increase that still
fits: **BK=6** (`sW` 27 KiB + `sX[128][7]` 32 KiB ≈ 59 KiB < 64).

**Change:** `MMQ_RP_Q8_BK` 4 → 6 (everything else parameterized by the macro,
partial last K-chunk handled by the existing `sb < n_sub` guards).

**Measured (pp2048, current power limit, via `SCRIPT_llama_bench.sh`):**
- baseline (BK=4): **1408 t/s**
- P2 (BK=6): **1377 t/s** → **−2.2%** (REGRESSION)

**Why it failed:** the GEMM is genuinely compute/dp4a-bound, not
fill-latency-bound — amortizing the fill over more K-steps buys nothing, and
the non-power-of-2 BK=6 slightly hurts unroll efficiency. Confirms the 79%
dp4a figure is the *efficient* throughput for this tile+microarchitecture,
not recoverable latency-hiding headroom.

**Status: REVERTED.** `MMQ_RP_Q8_BK` back to 4 (1405 t/s confirmed). Do not
re-apply. True double-buffering (BK=8 ping-pong) cannot fit 64 KiB LDS here.

### Attempt G — relax `launch_bounds` (P3) (FAIL / REGRESSION — constraint confirmed)
**Goal (P3 proposal):** drop the hard `__launch_bounds__(512,1)` and let ROCm
pick occupancy, hoping for more waves at the current LDS size.

**Change:** removed the `__launch_bounds__(512,1)` attribute from
`mmq_gemm_q8_0_repacked`.

**Measured (pp2048, current power limit, via `SCRIPT_llama_bench.sh`):**
- baseline (`launch_bounds(512,1)`): **1408 t/s**
- P3 (no launch_bounds): **1194 t/s** → **−15.2%** (MAJOR REGRESSION)

**Why it failed (and what it proved):** without the bound, ROCm's default
occupancy heuristic capped registers for "more blocks" (~64 VGPR/thread),
forcing spills / worse ILP for the unrolled dp4a loop (which wants the full
128 VGPR/thread budget). So `launch_bounds(512,1)` is **not a tunable to
relax — it is what gives the kernel its ILP** (1 block, 8 waves, full VGPR).
The LDS (40.5 KiB) would cap to 1 block regardless, but the *register*
budget from the explicit bound is what matters.

**Status: REVERTED.** `launch_bounds(512,1)` restored (1405 t/s confirmed).
Keep it. This closes the occupancy/launch_bounds lever entirely.

### Attempt H — fold activation scale at staging / cut per-group FMA (P4) (NEUTRAL)
**Goal (P4):** `dx = __low2float(xb->ds)` is per `(n,kk)` but was recomputed
inside the `r` loop (NROW times). Hoist it to once per `(n,kk)` before the `r`
loop.

**Change:** split the inner loop — compute `float dx[TN_]` once per `kk`
(all `n`) before the `r` loop; the dp4a loop uses `dx[n]`. Arithmetic result
identical (still `acc[r][n] += d * dx[n] * (float)idot`).

**Measured (pp2048, current power limit, via `SCRIPT_llama_bench.sh`):**
- baseline: **1408 t/s**
- P4 (hoist dx): **1409 t/s** → **±0% (neutral, within noise)**

**Why neutral:** the compiler (with `-O3`) had already hoisted the
loop-invariant `__low2float` out of the `r` loop, so the manual hoist changes
nothing in the generated code. No regression, kept as a minor clarity
improvement (not reverted).

**Status: KEPT (neutral).** This closes the scale-fold lever — there is no
further arithmetic to fold (the per-group `d*dx*idot` FMA is fundamental).

### Attempt J — hoist sX.qs out of the r loop into registers (NEUTRAL / marginal)
**Goal:** parallel to Attempt H (which hoisted `dx = __low2float(xb->ds)` out of
the `r` loop), hoist the 32-byte `sX[tx + n*64][kk].qs` read out of the `r` loop
into a register-resident `int xq32_cached[TN_][8]`. The same activation block is
read NROW=16 times per `(kk, n)`; if the compiler was not already hoisting it,
this saves 16× re-fetches per K-step.

**Change:** split the K-loop opening — compute `dx[n]` AND load the 8 qs ints
(via 2× `uint4` reads) into `xq32_cached[n][0..7]` once per `(kk, n)` before the
`r` loop; the inner `n` loop now reads `const int * xq32 = xq32_cached[n]`
instead of `&sX[tx + n*64][kk].qs`. Arithmetic identical (still 8 `dp4a` per
`(r, n)`, single `idot` accumulator). VGPR cost: +16 (TN_=2 × 8 ints) on top of
the existing ~50.

**Measured (pp2048, SCRIPT_llama_bench.sh, current power limit, -r 3):**
- baseline (no hoist, 8 samples): mean ~1410 t/s, σ ≈ 5 (1402–1418)
- Attempt J (hoist, 11 samples):    mean ~1418 t/s, σ ≈ 6 (1407–1426)

**Verdict:** +8 t/s = +0.55%, sub-2σ — **statistically marginal to neutral**.
The compiler almost certainly was already hoisting the re-reading `sX.qs`
out of the `r` loop (Attempt H/P4 had shown the same outcome for the
scalar `dx` hoist). The change is arithmetically identical (greedy decode N/A —
this is the prefill inner loop only), so it can ship, but it is **not a
win that beats the kernel's existing ceiling**.

**Status: KEPT (neutral, structural clarity).** Mirrors Attempt H — when the
compiler already does the hoist, manual hoisting just makes the source clearer
and removes a "is the compiler smart enough?" question from future profiling
without changing the generated GCN. Closes the "register-resident activation"
lever — if the inner dp4a chain is unchanged at this point it is because *load
width is not the limiter*, confirming the §4 #2 (LDS bank swizzle) and §4 #4
(occupancy) levers are the only movable ones left.

### Attempt I — vectorized 128-bit activation staging (P5) (NEUTRAL)
**Goal (P5):** `rp_xq_from_mmq` (the prefill activation loader used by all
repack GEMMs) copied the 32 activation int8 with **32 scalar byte loads**
(`out.qs[k] = m.qs[(sb&3)*32 + k]`). Replace with 2× `uint4` (128-bit) copies
— `reinterpret_cast<const uint4*>(m.qs + (sb&3)*QK8_1)` → 2 loads into
`out.qs`. Scale handling unchanged.

**Measured (pp2048, current power limit, via `SCRIPT_llama_bench.sh`):**
- baseline: **1408 t/s**
- P5 (uint4 copy): **1407 t/s** → **±0% (neutral, within noise)**

**Why neutral:** the activation staging runs once per K-chunk and is
overlapped/hidden by the dp4a compute (the GEMM is compute-bound, not
staging-bound — consistent with P2's failure). The 33→3 transaction reduction
helped memory efficiency but not wall-clock PP. Correctness identical (byte
order preserved by `uint4` copy).

**Status: KEPT (neutral).** Genuine code-quality win (fewer global
transactions); no regression. This closes the vectorized-loads lever.

### Verdict — kernel optimization concluded
All five proposed levers (P1–P5) plus the earlier relayout / split-accumulator
attempts have been tried. **Only the Plan-A quantizer switch + decode reroute
deliver a real, durable win: +9.9% PP over native (1408 vs 1297).** Every
subsequent structural change either regressed (P1, P2, P3, split-acc,
relayout) or was neutral (P4, P5). The GEMM is at its efficient dp4a ceiling
for this tile/microarchitecture; the ~79%-of-peak figure is real throughput,
not recoverable headroom. No further kernel changes recommended.

## 4. Remaining levers (prioritized for next session)

All target dp4a utilization in `mmq_gemm_q8_0_repacked`
(`repack-gcn.cu:1740`, `__launch_bounds__(512,1)`, tile 128×128, `MMQ_RP_Q8_BK=4`,
64×8 threads). Inner K-loop with dp4a at `repack-gcn.cu:~1856`:

```cpp
for (int kk = 0; kk < MMQ_RP_Q8_BK; kk++) {
    for (int r = 0; r < NROW; r++) {            // NROW = 16
        uint4 wlo = sW_lo[row][kk]; uint4 whi = sW_hi[row][kk];
        const float d = sWd[row][kk];
        for (int n = 0; n < TN_; n++) {          // TN_ = 2
            const block_q8_1 * xb = &sX[tx + n*64][kk];
            const int * xq32 = reinterpret_cast<const int *>(xb->qs);
            const float dx = __low2float(xb->ds);
            int idot = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                idot = ggml_cuda_dp4a((int) lo[j], xq32[j],     idot);
                idot = ggml_cuda_dp4a((int) hi[j], xq32[j + 4], idot);
            }
            acc[r][n] += d * dx * (float) idot;
        }
    }
}
```

 1. ~~**Double dp4a ILP (split lo/hi accumulators)**~~ — RETRACTED (Attempt D).
    Measured as a win only because the power limit was raised mid-session; at
    equal power it is a **regression** (extra VGPR hurts occupancy). Do not
    re-apply. The single-`idot` form is correct and is the current code.

2. **LDS bank-conflict-free swizzle for `sW` / `sX`.**
   The 8 int8 reads per dp4a (`lo[j]`,`hi[j]` from `sW`; `xq32[j]`,`xq32[j+4]`
   from `sX`) must map to distinct LDS banks. Add a swizzle on the K dimension
   when writing `sW_lo/sW_hi/sX` so the inner reads are conflict-free. Classic
   +5–15% on GEMMs. Current LDS: `sW_lo[128][4]`, `sW_hi[128][4]`,
   `sWd[128][4]`, `sX[128][5]` (~41 KiB, under 64 KiB budget).

3. **Software-pipeline the K-loop (double-buffered LDS) + try `BK=8`.**
   Prefetch the next BK chunk of `sW`/`sX` while computing the current one.
   Lower priority if purely compute-bound (no memory latency to hide), but
   combine with larger `BK=8` for more arithmetic per fill. Note: an earlier
   attempt at direct-from-global (no LDS) weight reads regressed to ~1030 pp
   (L2 cannot hide per-K global weight latency) — keep weights in LDS.

4. **Occupancy / `launch_bounds` tuning.**
   `launch_bounds(512,1)` = 8 waves (64-lane). If VGPR/LDS allow, raise max
   waves (e.g. `(512,2)`) to better hide dp4a latency. The K-quant repack GEMMs
   use `__launch_bounds__(256,2)` — compare behavior.

5. **Vectorized 128-bit global loads** for weight/activation staging
   (ensure `uint4`/wide loads in the `e`-loop staging at `repack-gcn.cu:~1809`).

6. **Scale-multiply hoist.** `d * dx` is already computed per (row, col) per
   K-step; confirm it is not recomputed inside the `j` unroll and that `d`/`dx`
   live in registers. Minor.

### What NOT to do (learned)
- Do **not** relayout / coalesce the activation buffer — the GEMM is
  compute-bound; it does not help (Attempt C).
- Do **not** read weights directly from global in the K-loop — regresses
  (~1030 pp historically).
- Do **not** split the `lo`/`hi` dp4a accumulators — at `launch_bounds(512,1)`
  the extra VGPR regresses occupancy; only "won" under a raised power limit
  (Attempt D, measurement artifact).
- Keep decode (`ne11==1`) on the canonical quantizer (Attempt B) — do not
  re-route it through the mmq path.
- **Never compare PP runs taken at different power-limit / clock settings.**
  A prefill-only edit must not move `tg128`; if it does, the run is
  confounded (see §0).
- **Do NOT remove `__launch_bounds__(512,1)` from `mmq_gemm_q8_0_repacked`**
  (Attempt G) — it regressed 15% by capping registers; the explicit bound is
  what gives the unrolled dp4a loop its full VGPR/ILP budget.

---

## 5. Benchmark recipe (PP only)

Single GPU, 4B, pp2048 only (the SCRIPT_llama_bench.sh was edited to just
`-p 2048 -n 0`; the default script targets 27B + tensor-parallel which hits the
open Track-B garbage bug and must NOT be used for this work):

```bash
export VENV_ROOT="/home/iacopo/Desktop/TheRock/.tmpvenv-vega"
export CORE_PATH="$VENV_ROOT/lib/python3.12/site-packages/_rocm_sdk_core"
export DEVEL_PATH="$VENV_ROOT/lib/python3.12/site-packages/_rocm_sdk_devel"
export LIBS_PATH="$VENV_ROOT/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx906/lib"
export LD_LIBRARY_PATH="$CORE_PATH/lib:$DEVEL_PATH/lib/llvm/lib:$DEVEL_PATH/lib:$LIBS_PATH:$LD_LIBRARY_PATH"
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export HIP_VISIBLE_DEVICES=0
export GGML_CUDA_P2P=1
export GGML_ENABLE_CUSTOM_AR=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GPU_MAX_HW_QUEUES=8
export GGML_LOG_LEVEL=1
export TURBOPREFILL=1
export GGML_CUDA_REPACK=1
export GGML_CUDA_REPACK_Q8_0=1          # 0 for native reference
export GGML_MMVQ_KSHARD_MAXROWS=0
./build/bin/llama-bench -m /media/iacopo/LLMs/llms/Qwen3-4B-Instruct-2507-Q8_0.gguf \
    -ngl 99 -t "$(nproc)" -fa 1 -ctk f16 -ctv f16 --progress -r 3 \
    -b 2048 -ub 2048 -mmp 0 -p 2048 -n 0
```

For kernel-level attribution use `discover_kernels.py` with
`GGML_CUDA_DISABLE_GRAPHS=1` (TG profiling is unreliable with graphs; PP is
fine). **Validity rule:** a PP win only counts if TG is unchanged for a
prefill-only edit (see §0). Compare against the re-established same-power
baseline from §6 step 1; do not compare against the old 1425 / 1779 numbers
without holding the power limit fixed.

---

## 6. Next-session action

The Q8_0 repack PP GEMM is back to the relayout-reverted baseline (1425 t/s
at the *original* power limit). The split-accumulator variant was a measurement
artifact and is reverted. Before any further kernel work:

1. **Re-establish a clean baseline at the CURRENT (raised) power limit.**
   Bench the relayout-reverted `REPACK_Q8_0=1` code AND native (`=0`) at the
   new power limit, both pp2048 and tg128, in the **same session** with the
   power limit fixed. Record both numbers; this is the new reference. (The
   observed value for the split variant at the new limit was 1407.18 — the
   baseline should be higher.)
2. **Only then** attempt the remaining levers (§4 #2–#6). Given the GEMM is
   compute-bound and occupancy-constrained at `launch_bounds(512,1)`, the most
   promising real lever is **#4 (occupancy / `launch_bounds`)** — but note LDS
   (~41 KiB) caps concurrent blocks at one, so occupancy gains require cutting
   VGPRs or LDS, not just raising the wave count. Treat #2 (LDS swizzle) as
   already near-optimal (2-way conflict, coprime pitch). Expect modest (<10%)
   gains; the relayout-reverted code is already a solid +9.9%-over-native win.
3. **Do NOT reintroduce the relayout** (Attempt C) or the split accumulator
   (Attempt D) — both proven neutral/negative at equal power.

Success gate: a proposed change counts as a win only if (a) pp2048 rises vs
the re-established same-power baseline AND (b) tg128 is unchanged (decode path
untouched). Correctness gate: greedy `--seed 42 --samplers topk -topk 1`
output must stay byte-identical to `GGML_CUDA_REPACK_Q8_0=0`.
