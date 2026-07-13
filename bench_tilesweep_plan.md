# Tile Sweep Plan v2 — repack-gcn.cu Q8_0 prefill GEMM

Goal: find the best compile-time tile `(BM, TN->BN, NRL)` for the repacked
Q8_0 MMQ prefill kernel, per ubatch, to close the small-N regression the
ubatch-only sweep could not characterize.

Revised from v1 per review (see "Review disposition" at bottom).

## Chosen options
- Injection method **(a)**: patch the `#define` lines in `repack-gcn.cu` per
  iteration (no CMake change). Source file is backed up and restored via `trap`.
- Grid: **full** — `NRL ∈ {1,2,4,8}`, `TN ∈ {1,2}`, `BM ∈ {32,64,128}`,
  `BK = 4` fixed. `BN = 64*TN` (derived). Filter: `BM % NRL == 0`
  **and** `NROW (= BM/NRL) >= 8`.

## 0. Precondition (only code change)
Wrap the 5 tile macros at `repack-gcn.cu:209-213` in `#ifndef` guards so a
patched value can override them:
```
#ifndef MMQ_RP_Q8_BK
#define MMQ_RP_Q8_BK 4
#endif
... TN, BM, BN, NROW_LANES (same pattern)
```
Defaults unchanged -> current behavior identical.

### NOTE (review #1 — already satisfied, no extra work)
The launch geometry ALREADY tracks the macros, because the NRL refactor
(commit 113c483c9, our earlier fix) parametrized it:
- kernel: `template <bool HAS_IDS, int TN_, int NRL>`
  `__launch_bounds__(64 * NRL, 1)`  (repack-gcn.cu:222-223)
- dense launch: `mmq_gemm_q8_0_repacked<…, MMQ_RP_Q8_NROW_LANES>
  <<<grid, dim3(64, MMQ_RP_Q8_NROW_LANES), …>>>`  (repack-gcn.cu:490)
- grid from `MMQ_RP_Q8_BM` / `MMQ_RP_Q8_BN`  (repack-gcn.cu:484-487)

So patching `MMQ_RP_Q8_NROW_LANES` (and BM/BN) correctly drives `NROW`,
`NTHREADS`, the `row = ty + r*NRL` map, `blockDim.y`, `launch_bounds`, and
the launch grid. The `#ifndef` guards are sufficient; no further
parametrization is required. (The MoE path at :585 hardcodes `NRL_D=8`, but
this sweep targets the dense prefill path only.)

## 1. Candidate grid (22 configs)
| BM  | TN | BN  | valid NRL (BM%NRL==0 && NROW>=8) | # |
|-----|----|-----|----------------------------------|---|
| 32  | 1  | 64  | 1, 2, 4                          | 3 |
| 32  | 2  | 128 | 1, 2, 4                          | 3 |
| 64  | 1  | 64  | 1, 2, 4, 8                       | 4 |
| 64  | 2  | 128 | 1, 2, 4, 8                       | 4 |
| 128 | 1  | 64  | 1, 2, 4, 8                       | 4 |
| 128 | 2  | 128 | 1, 2, 4, 8                       | 4 |
Total = 22. Default repack tile (`BM=64, TN=2, NRL=4`) is one of these.

## 2. Resource check (all 22 fit LDS; occupancy cliff noted)
LDS = `2*BM*BK*16` (sW_lo/hi uint4) + `BM*BK*4` (sWd float) + `BN*(BK+1)*sizeof(block_q8_1)` (~36 B).
- `BM=64, BN=128` (default): ~31.5 KiB -> ~2 blocks/CU.
- `BM=128, BN=128`: ~40.5 KiB -> **1 block/CU** (occupancy cliff — expect
  these 4 configs to be occupancy-bound regardless of VGPRs; pre-annotate).
All < 64 KiB, so no LDS filter beyond the grid above. VGPR spills (small NRL
-> large NROW) are NOT pre-filtered — the sweep surfaces them (#5 logs them).

## 3. Driver: new script `sweep_tiles.sh`  (review #6 hardening)
- `BACKUP=repack-gcn.cu.bak`; `cp` it. `trap 'cp -f "$BACKUP" repack-gcn.cu; rm -f "$BACKUP"' EXIT INT TERM` so the tree is ALWAYS restored, even on
  kill/mid-build. (We patch a tracked file while other uncommitted changes
  exist nearby — the trap is mandatory; we do NOT `git checkout` it.)
- Sets the same ROCm/GGML env as `SCRIPT_llama_bench.sh`
  (HSA_OVERRIDE_GFX_VERSION, GGML_CUDA_REPACK, GGML_MMVQ_KSHARD_MAXROWS=1
  [inert for -n 0, noted], etc.) and reuses `BENCH_PARAMS` / model path.
- `rocm-smi --setperflevel high` (or fixed sclk) to lock clocks (review #4).
- **Native baseline once** (GGML_CUDA_REPACK=0) -> recorded to `bench_tilesweep.md`.
- **Reference output for correctness gate** (#2): greedy-decode a fixed prompt
  once with native (`llama-cli -p "$PROMPT" -n 32 -temp 0 -s 1234 -ngl 99`),
  store the token string as `REF_TOKENS`.
- Loop over the 22 configs **in randomized order** (#4, avoids drift correlation):
  1. Patch the 3 varying `#define`s (`MMQ_RP_Q8_BM`, `MMQ_RP_Q8_TN`,
     `MMQ_RP_Q8_NROW_LANES`) in `repack-gcn.cu` (BN auto-derives from TN).
  2. `cmake --build build --parallel $(nproc)` — incremental, recompiles only
     `repack-gcn.cu` + relink. ~60-120 s.
  3. On build failure: log, `continue` (trap restores file).
  4. **Correctness gate** (#2): build the same `llama-cli` greedy decode with
     this binary; if token string != `REF_TOKENS` -> mark **FAIL**, skip timing.
  5. Run `llama-bench` **repack-only** (`GGML_CUDA_REPACK=1`) with
     `-ub 16,32,64,128,256,512,1024,2048 -p 2048 -n 0 -r 5` (#4:
     median + stdev instead of 1 rep). Tee into `bench_tilesweep.md`, tagged
     `### BM<T> TN<T> NRL<T>` with a **resource line** (#5: computed LDS
     bytes + occupancy class; best-effort VGPR/scratch scrape from build log if
     present, non-blocking).
  6. On bench failure: log, `continue`.
- `trap` restores `repack-gcn.cu` at exit. `rocm-smi --setperflevel auto`.

## 4. Output & aggregation
- Raw: `bench_tilesweep.md` (one tagged section per config + native baseline +
  resource lines).
- Summary matrix (appended): rows = ubatch `{16,32,64,128,256,512,1024,2048}`,
  columns = each of the 22 configs + native ref, cells = **median t/s (stdev)**.
  FAIL configs shown as `FAIL`.
- "Best tile per ubatch" table: for each ubatch, the passing config with max
  median t/s and its delta vs native — directly answers "which tile beats native
  at which N", mirroring what native's per-N `mmq_x` selection does.

## 5. Cost estimate
22 configs × (~90 s build + ~90 s repack bench + ~5 s gate) + 1 native
baseline + 1 ref decode ≈ **60-80 min** total. Tunable by pruning the grid.

## 6. Intended follow-up (review #7)
The per-ubatch table *characterizes* the ideal tile, but a single global tile
cannot beat native across all N (BM=128/TN=2 is an occupancy cliff). The
actionable fix is to instantiate a few tiles (e.g. small-N: NRL=1/2, BN=64;
large-N: NRL=4, BN=128) and **pick at launch by `ne11`**, mirroring native's
`mul_mat_q_case`. This sweep's output feeds that dispatch table directly.

## 7. Optional extensions (out of core scope)
- **BK ∈ {2,4}** (review #8): BK is the # of sub-blocks staged per K-step,
  independent of the uint4 qs layout; `BK=2` halves the three W planes and may
  rescue the BM=128 occupancy cliff. Adds a second ×2 sweep; defer unless the
  BM=128 configs underperform as predicted.
- Full VGPR/scratch extraction via ISA dump (`roc-obj-dump`) as a richer
  replacement for the best-effort build-log scrape in #5.

## 8. Review disposition
- #1 launch geometry: REJECTED as already implemented by the NRL refactor
  (evidence at §0 NOTE, file:line refs). `#ifndef` guards suffice.
- #2 correctness gate: ACCEPTED -> §3 step 4.
- #3 grid filter: ACCEPTED -> explicit `NROW >= 8` filter, 22 configs retained.
- #4 reps/clock/order: ACCEPTED -> `-r 5`, perflevel lock, randomized order.
- #5 resource logging: ACCEPTED (lightweight) -> §3 step 5.
- #6 isolation/trap: ACCEPTED -> §3 trap + backup.
- #7 dispatch follow-up: ACCEPTED as forward-looking -> §6.
- #8 BK sweep: DEFERRED optional -> §7.
- Minor: LDS formula corrected (sWd is float 4B); occupancy cliff annotated;
  `-ub 16-2048*2` syntax already validated by the prior `SCRIPT_llama_bench.sh`
  run (produced 8 ubatch rows); `GGML_MMVQ_KSHARD_MAXROWS=1` noted inert for -n 0.
