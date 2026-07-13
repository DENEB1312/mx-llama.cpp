# Tile Sweep Plan — repack-gcn.cu Q8_0 prefill GEMM

Goal: find the best compile-time tile `(BM, TN->BN, NRL)` for the repacked
Q8_0 MMQ prefill kernel, per ubatch, to close the small-N regression that
the ubatch-only sweep could not characterize.

## Chosen options
- Injection method **(a)**: patch the `#define` lines in `repack-gcn.cu` per
  iteration (no CMake change). Source file is backed up and restored at the end.
- Grid: **full** — `NRL ∈ {1,2,4,8}`, `TN ∈ {1,2}`, `BM ∈ {32,64,128}`,
  `BK = 4` fixed. `BN = 64*TN` (derived). Filter: `BM % NRL == 0`.

## 0. Precondition (only code change)
Wrap the 5 tile macros at `repack-gcn.cu:209-213` in `#ifndef` guards so a
build-time value can override them:
```
#ifndef MMQ_RP_Q8_BK
#define MMQ_RP_Q8_BK 4
#endif
... TN, BM, BN, NROW_LANES (same pattern)
```
Defaults unchanged → current behavior identical.

## 1. Candidate grid (22 configs)
| BM  | TN | BN  | valid NRL (BM%NRL==0)        | # |
|-----|----|-----|-------------------------------|---|
| 32  | 1  | 64  | 1, 2, 4                       | 3 |
| 32  | 2  | 128 | 1, 2, 4                       | 3 |
| 64  | 1  | 64  | 1, 2, 4, 8                    | 4 |
| 64  | 2  | 128 | 1, 2, 4, 8                    | 4 |
| 128 | 1  | 64  | 1, 2, 4, 8                    | 4 |
| 128 | 2  | 128 | 1, 2, 4, 8                    | 4 |
Total = 22.

Default repack tile (`BM=64, TN=2, NRL=4`) is naturally one of these, so it
needs no separate baseline run.

## 2. Resource check (all 22 fit)
LDS ≈ `3*BM*BK*16` (sW_lo/hi uint4 + sWd float counted as 16B) + `BN*(BK+1)*sizeof(block_q8_1)`.
Worst case `BM=128, BN=128`: ~41 KiB < 64 KiB. All configs fit; no LDS
filter needed. VGPR pressure (small NRL -> large NROW) is *not* pre-filtered —
the sweep is expected to surface spills as slow/failed configs, which we log and
skip.

## 3. Driver: new script `sweep_tiles.sh`
- Backs up `ggml/src/ggml-cuda/repack-gcn.cu` to `repack-gcn.cu.bak`.
- Sets the same ROCm/GGML env as `SCRIPT_llama_bench.sh` (HSA_OVERRIDE_GFX_VERSION,
  GGML_CUDA_REPACK, GGML_MMVQ_KSHARD_MAXROWS=1, etc.) and reuses the
  `BENCH_PARAMS` / model path.
- **Native baseline once** (GGML_CUDA_REPACK=0) -> recorded to `bench_tilesweep.md`.
  (Native is tile-independent, so run it a single time, not per config.)
- Loop over the 22 configs:
  1. Patch the 3 varying `#define`s (`MMQ_RP_Q8_BM`, `MMQ_RP_Q8_TN`,
     `MMQ_RP_Q8_NROW_LANES`) in `repack-gcn.cu` (BN auto-derives from TN).
  2. `cmake --build build --parallel $(nproc)` — incremental, recompiles only
     `repack-gcn.cu` + relink. ~60-120 s.
  3. On build failure: log, restore file, `continue`.
  4. Run `llama-bench` **repack-only** (`GGML_CUDA_REPACK=1`) with
     `-ub 16-2048*2 -p 2048 -n 0`. Tee into `bench_tilesweep.md`, tagged
     `### BM<T> TN<T> NRL<T>`.
  5. On bench failure: log, `continue`.
- Restore `repack-gcn.cu` from backup at the end (repo left clean).
- Note: the patched file is NOT committed; sweep is throwaway/profiling only.

## 4. Output & aggregation
- Raw: `bench_tilesweep.md` (one tagged section per config + native baseline).
- Summary matrix (appended): rows = ubatch `{16,32,64,128,256,512,1024,2048}`,
  columns = each of the 22 configs + native ref, cells = t/s.
- "Best tile per ubatch" table: for each ubatch, the config with max t/s and its
  delta vs native — directly answers "which tile beats native at which N",
  mirroring what native's per-N `mmq_x` selection does.

## 5. Cost estimate
22 configs × (~90 s build + ~90 s repack bench) + 1 native baseline
≈ **60-75 min** total. Tunable by pruning the grid later.

## 6. Risks / open items
- Patching source from a script mutates a tracked file; the backup/restore step
  is mandatory so the working tree ends clean (your other uncommitted changes in
  DENEB/ scripts are untouched — we only touch `repack-gcn.cu`).
- Configs that spill VGPRs or mis-launch will show as slow/errors; logged and
  skipped rather than aborting the sweep.
- `BK` intentionally fixed at 4 (kernel K-loop / 32-byte qs uint4 staging assume
  it); not part of this sweep.
