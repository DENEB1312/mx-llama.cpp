# Repack Q8_0 tuning — parameter linkage & sweep correctness

This document explains how every tile / kernel dimension of the Q8_0 repack
path (`ggml/src/ggml-cuda/repack-gcn.cu`) is derived, which parameters are
coupled, and — critically — what is and isn't safe to sweep when tuning.

---

## 1. The MMQ (prefill) kernel — `mmq_gemm_q8_0_repacked`

### 1.1 Master macros (`repack-gcn.cu:310-314`)

| Macro | Value | Role |
|---|---|---|
| `MMQ_RP_Q8_BK` | 4 | K-contraction depth (sub-blocks per LDS stage / per K-loop iter) |
| `MMQ_RP_Q8_TN` | 2 | token tiles along N **per block** (prefill) |
| `MMQ_RP_Q8_BM` | 64 | weight rows **per block** (base M-tile) |
| `MMQ_RP_Q8_BN` | `64 * MMQ_RP_Q8_TN` = 128 | token columns **per block** (N-tile) |
| `MMQ_RP_Q8_NROW_LANES` | 4 | rows computed per row-lane |

`TN_` is the only additional knob, and it is a **template parameter hard-wired
per call site**: `MMQ_RP_Q8_TN` (=2) for dense prefill, `TN_ID` (=1) for
MUL_MAT_ID (`repack-gcn.cu:311`, `749`).

### 1.2 Derived, still-fixed quantities

```
NROW     = MMQ_RP_Q8_BM / MMQ_RP_Q8_NROW_LANES   = 64 / 4 = 16   (rows per row-lane)
NTHREADS = 64 * MMQ_RP_Q8_NROW_LANES             = 64 * 4 = 256  (staging threads)
```

These feed directly into the code (`repack-gcn.cu:376`, `382`):

- `NROW` sizes the accumulator array `acc[NROW][TN_]` (`:377`).
- `NTHREADS` sizes the cooperative LDS fill loops (`for e = t; e < w_elm; e += NTHREADS`, `:387`, `:401`).

### 1.3 How the block shape is assembled from the thread grid

The kernel is launched with `blockDim.x = 64`, `blockDim.y ∈ {4, 8}`:

```
tx = threadIdx.x            // 0..63  — one column lane per token row (no broadcast)
ty = threadIdx.y            // 0..3 (prefill, 256-thread)  / 0..7 (ID, 512-thread)
```

**N direction (tokens).** Each token column is owned by a specific `tx`:
```
col = tx + n * 64,          n ∈ [0, TN_)          // repack-gcn.cu:491
```
Because there are exactly 64 `tx` lanes and `TN_` groups of 64, the N-tile width
is forced:

```
BN = 64 * TN            →  128 (prefill, TN=2) / 64 (ID, TN=1)
```

So `MMQ_RP_Q8_BN` is **not independent** — it is `64 * TN_` by construction;
the literal `64` in its definition is the `tx` lane count.

**M direction (weight rows).** Each weight row is owned by a `(ty, r)` pair:
```
row = row0 + ty + r * MMQ_RP_Q8_NROW_LANES,   r ∈ [0, NROW)   // repack-gcn.cu:486
```
With `ty ∈ [0, blockDim.y)` and `NROW = 16`, the per-block row span is:

```
BM = (blockDim.y) * NROW
```

For the **prefill launch `dim3(64,4)`**: `ty ∈ 0..3`, rows = `ty + 4r`, `r 0..15`
→ covers `0..63` ⇒ **effective BM = 64** = `MMQ_RP_Q8_BM`. ✅ consistent.

For the **ID launch `dim3(64,8)`**: `ty ∈ 0..7` overshoots (rows `0..67`, with
overlap) — the extra 4 row-lanes are redundant for `BM=64` but harmless; the
512-thread block exists to satisfy `__launch_bounds__(512,1)` and keep 8 waves
in flight. Net unique rows still 64 (plus wasted overlap). See §4.2 for the
correctness implication.

**K direction.** The contraction steps in `BK` chunks:
```
for sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_Q8_BK      // repack-gcn.cu:386
```
where `n_sub = ne0 >> 5` is the number of 32-weight sub-blocks. So the K-loop
**trip count** `= n_sub / 4` scales with K, but the per-step depth `BK=4` is
fixed.

### 1.4 LDS budget — the hard constraint linking BM, BN, BK

All three tile dims land in shared memory (`repack-gcn.cu:371-374`):

```
sW_lo[BM][BK]     uint4  →  64*4*16 B =  4 KB
sW_hi[BM][BK]     uint4  →  64*4*16 B =  4 KB
sWd  [BM][BK]     float  →  64*4*4  B =  1 KB
sX   [BN][BK+1]   block_q8_1 (36 B) →  BN*5*36 B
```

Total (prefill, BN=128): `9 KB + 128*5*36 ≈ 31.5 KB < 64 KB`.
Total (ID, BN=64): `9 KB + 64*5*36 ≈ 20 KB`.

This is why `BN` is capped at 128: the in-file comment (`repack-gcn.cu:343-345`)
notes the rejected `BN=64, TN=1` variant (native's 128×64 shape) regressed
because halving per-block work doubled the block count. Growing `BN` further
would breach the 64 KiB LDS limit. **BM, BN, BK are coupled through this
budget** — you cannot widen one without shrinking another or spilling LDS.

> ⚠️ NOTE: the actual `block_q8_1` size is **36 B** (32 qs + 4 ds). Any LDS
> estimator that uses 34 B (as the sweep script does, see §4.3) *under*-estimates
> and can let a config that truly exceeds 64 KiB slip through — it then fails at
> kernel launch (runtime error), not silent wrong numbers.

### 1.5 The launch grid — the only part that scales with shape

```
mmq_bm = MMQ_RP_Q8_BM   (64)
mmq_bn = MMQ_RP_Q8_BN   (128)
grid.x = ceil(ne01 / mmq_bm)     // # M tiles  ← scales with M
grid.y = ceil(ne11 / mmq_bn)     // # N tiles  ← scales with N      (repack-gcn.cu:637-638)
```

`ne11` (ubatch / N) changes **how many tiles launch**, never their size. This is
the crux of the known limitation (top-of-file doc, `repack-gcn.cu:33-44`):
native `mul_mat_q<Q8_0>` selects `mmq_x ∈ {8..128}` *per N* (capped 64 for
N<4096 on gfx906), so small N uses small N-tiles and stays efficient; repack
always uses 128-wide N-tiles, so small-N ubatches waste most of each tile
(measured ≈ −39% at pp128).

---

## 2. The matvec (decode) kernel — `mul_mat_vec_q8_0_repacked`

Runs only at `ne11 == 1`. Template `<ROWS, NWAVES, HAS_IDS>`; block is a
wave64 grid: `block = 64 * NWAVES` threads, `ROWS` output rows per wave.

```
ne01 >= 4096  →  <1,1,false>  grid(ne01),    block 64   (1 wave,  ROWS=1)
else          →  <2,4,false>  grid(ceil(ne01/8)), block 256 (4 waves, ROWS=2)   (repack-gcn.cu:608)
```
ID decode uses the identical two-config switch on `ne01` (`repack-gcn.cu:718`).

This is the **single runtime shape-dependent switch** in the whole repack path:
a coarse 2-way pick of block size / `ROWS` / `NWAVES` based on `ne01` (the
output dim M), *not* on N (which is fixed at 1 here). The rationale (comment at
`:609-617`): large out_dim needs max wavefront count (ROWS=1), small out_dim
wants the 4-wave ROWS=2 shape.

Within the kernel the work unit is a 16-weight **half sub-block** (4 dp4a each),
looped `n_half = n_blocks*2` times over `64` lanes (`repack-gcn.cu`, matvec
body). `n_blocks = ne0>>5` scales with K; the 4 dp4a per half is fixed.

---

## 3. Parameter dependency graph

```
                 ┌─────────────── compile-time macros ───────────────┐
                 │  MMQ_RP_Q8_BK=4                                   │
                 │  MMQ_RP_Q8_BM=64                                  │
                 │  MMQ_RP_Q8_NROW_LANES=4                           │
                 │  TN_  (2 prefill / 1 ID)   ← template, per call   │
                 │  (64 = blockDim.x = column-lane count, HARDCODED) │
                 └───────────────────────────────────────────────────┘
                                  │
            ┌─────────────────────┼─────────────────────────┐
            ▼                     ▼                          ▼
   NROW = BM/NROW_LANES   BN = 64 * TN_            NTHREADS = 64*NROW_LANES
   (rows/lane = 16)        (128 / 64)              (= 256 staging threads)
            │                     │                          │
            ▼                     ▼                          ▼
   acc[NROW][TN_]          blockDim.x=64, blockDim.y∈{4,8}   LDS fill stride
   row = ty + r*NROW_LANES col = tx + n*64            (w_elm/x_elm loops)
            │                     │
            ▼                     ▼
   effective BM=64         effective BN=128(prefill)/64(ID)
            │                     │
            └─────────┬───────────┘
                      ▼
            LDS: sW[BM][BK] + sX[BN][BK+1]   ← BM, BN, BK ALL coupled by 64 KiB budget
                      │
                      ▼
            grid.x = ceil(ne01/BM) ,  grid.y = ceil(ne11/BN)   ← only COUNTS scale w/ shape

   K-loop trip count = n_sub/4  (n_sub = ne0>>5)   ← scales with K, depth BK fixed
```

---

## 4. The tile sweep (`SCRIPT_llama_bench_TILE_SWEEP.sh`)

### 4.1 What the sweep covers

The script patches 4 macros + rebuilds + runs `llama-bench`, sweeping:

- **BM** ∈ {64, 128, 256}
- **BK** ∈ {2, 4, 8}
- **TN** ∈ {1, 2}
- **NROW_LANES** ∈ {4, 8}
- plus **ubatch** 16..2048 (the N axis), prompt fixed at 2048.

It derives `BN = 64·TN`, `NROW = BM/NROW_LANES`, recomputes an LDS estimate, and
skips `BM % NROW_LANES != 0` and LDS > 64 KiB. It patches the `#define`s, the
`__launch_bounds__` thread count, and the prefill `<false>` dispatch
`dim3(64, NROW_LANES)`.

### 4.2 Correctness of the current 4-macro sweep — SAFE (dense prefill only)

The four patched macros are internally consistent **because every one keeps
`blockDim.x = 64`**:

- **BM**: `row = row0 + ty + r·NROW_LANES`, `NROW = BM/NL`, `grid.x = ceil(ne01/BM)`,
  LDS `[BM][BK]`. Correct iff `BM % NL == 0` (enforced).
- **BK**: LDS `[·][BK]`, K-loop `sb0 += BK`, partial last stage guarded by
  `sb < n_sub` and `wrow < ne1`. Correct.
- **TN**: `BN = 64·TN`, `col = tx + n·64`, `grid.y = ceil(ne11/BN)`, LDS `sX[BN][BK+1]`.
  Consistent because `tx` is 0..63 and `64` is fixed.
- **NROW_LANES**: patched into both `__launch_bounds__` thread count and the prefill
  `dim3(64, NL)`, and `NTHREADS = 64·NL` matches the actual thread range. Correct.

So the 4-macro sweep, with its two constraints, is correct for the dense prefill
path.

### 4.3 Bugs / gaps in the sweep script

1. **LDS estimator uses 34 B for `block_q8_1`** (wrong — it is 36 B). It
   *under*-estimates, so a config that truly exceeds 64 KiB can pass the gate and
   then fail at kernel launch (runtime error, not silent wrong numbers). Fix
   `34 → 36`.
2. **ID (`<true>`) dispatch is never patched.** The script only rewrites the
   prefill `dim3(64, NL)`. The MoE path keeps `dim3(64,8)`, so its tile is
   whatever `NROW_LANES` the patch set (4 or 8) — unswept and unbenchmarked.
   Any "best config" is **dense-only**.
3. **Stale baseline.** The summary's "Original config BM=128 / BN=128 / NL=8 /
   Threads=512" no longer matches the current source (now BM=64, NL=4). The
   baseline row is misleading.
4. **Only prompt = 2048 is benchmarked.** Acceptable for tile efficiency (K only
   scales the K-loop trip count, not the tile), but note it is a single point.

### 4.4 Adding sweep dimensions — what is and isn't safe

**`blockDim.x` (column-lane count) sweep — UNSAFE as currently written.**
The kernel bakes the column count `64` into *five* independent places:

- `col = tx + n * 64`                 (read/write indexing)
- `tok0 = blockIdx.y * (64 * TN_)`    (N-tile origin)
- `NTHREADS = 64 * MMQ_RP_Q8_NROW_LANES` (staging stride)
- `MMQ_RP_Q8_BN (64 * MMQ_RP_Q8_TN)`  (LDS width)
- `sX[sX_swizzle(tx + n * 64)]`       (activation gather)

The script's `patch_config` only touches `#define`s and two anchors; it does
**not** replace these `64`s. If you set `blockDim.x = 32` you get 32 column
lanes but the code still computes `tx + n*64` and stages `BN = 64·TN` columns →
columns are mis-assigned and ~half dropped/aliased. That is a **silent wrong
result**, not a build error. To sweep `blockDim.x` you must introduce a
`MMQ_RP_Q8_NCOL` macro and replace all five literals; the script cannot do that
today. This is the single most important caveat: the small-N regression is
about narrow N-tiles (BN < 64), which **requires** sweeping `blockDim.x`, and
that sweep is not correctness-safe with the present patching logic.

**Occupancy (`__launch_bounds__` 2nd arg) — SAFE.** It is only a compiler hint;
no kernel logic depends on it. At worst it spills registers.

**Loop interchange / `sX_swizzle` / double-buffering / activation read width —
NOT safe by default.** These are not macros; they are code. The script doesn't
patch them, so it cannot sweep them without new edits, and a wrong edit changes
the (row, col, kk) reduction → wrong answers.

**matvec `ROWS` / `NWAVES` — needs coordinated dispatch regen.** They live in a
different kernel with its own `block = 64·NWAVES` and
`grid = ceil(ne01/(ROWS·NWAVES))` invariant. Sweeping them requires regenerating
that dispatch, which the script doesn't do.

### 4.5 Pre-existing latent MoE correctness bug (verify separately)

The ID write guard at `repack-gcn.cu:494` reads:

```
const uint32_t col = tx + n * 64;
if constexpr (HAS_IDS) {
    const uint32_t a = a_base + col;
    if (a < a_end && col < (uint32_t)(16 * TN_)) {   // ← suspicious
```

With `col = tx + n*64` and `tx ∈ 0..63`, `col < 16*TN_` drops every column
`≥ 16*TN_` (e.g. all of n=1 for TN=2, and cols 16..63 even for TN=1). That looks
like a **latent MoE-batch correctness bug** — the MMQ ID path would silently
write only a subset of its computed columns. The sweep never exercises it (dense
model; decode uses the matvec kernel, not the ID MMQ), so it is invisible to the
benchmark. Audit/quantize this guard before trusting repack on any expert model.

---

## 5. What is fixed vs. what moves with the shape

| Quantity | Driven by | Varies with shape? |
|---|---|---|
| Block size (256 / 512) | launch site | No (prefill vs ID only) |
| `BK` | macro | No |
| `BM` (64) | macro | No |
| `BN` (128/64) | `TN_` + 64 lanes | No |
| `NROW`, `NTHREADS` | `BM`, `NROW_LANES` | No |
| LDS footprint | `BM`,`BN`,`BK` | No |
| K-loop iterations | `ne0` | Yes (K) |
| `grid.x` | `ne01`, `BM` | Yes (M) |
| `grid.y` | `ne11`, `BN` | Yes (N) |
| matvec block (64 vs 256) | `ne01>=4096` | Yes, 2-way on M only |

---

## 6. Making the tile N-selectable (the key work item)

To beat native across all N, the N-tile must become a runtime choice keyed on
`ne11`, mirroring native `mmq_x`. Concretely the coupled set to parameterize:

1. **`TN_`** (template) → runtime value from `ne11` (e.g. 1 for small N, 2 for
   large N). This also resizes `BN = 64*TN_` and `sX[BN][BK+1]` — so LDS is the
   binding constraint (keep `BN ≤ 128` or shrink `BM`/`BK` to compensate).
2. **`blockDim.x`** (token lanes) — currently fixed at 64 = one wavefront. A
   narrower N-tile could use fewer column lanes (e.g. 32 → `BN=32*TN`), giving
   smaller tiles for small N **without** LDS blow-up. **This requires a new
   `NCOL` macro replacing the five hardcoded `64` literals** (see §4.4) — it is
   the real lever for the small-N regression but is not safe to sweep until that
   plumbing exists.
3. Optionally **`BM`** / `blockDim.y` to vary the M-tile with M as well, but the
   current `BM=64` is already near the sweet spot for gfx906 occupancy.

Everything else (`BK`, the dp4a inner loop, the matvec half-sub-block unit, the
LDS staging pattern) is independent of N and can stay fixed.

### Fixed-in-code but sweepable-in-theory dimensions

- **Sub-block work-unit granularity**: 8 / 16 / 32 weights per loop iter (matvec
  currently pinned to 16-weight "half"; the MMQ K-loop is per-sub-block).
- **`nsp` power-of-two padding policy**: pad +1 / no-pad / pad +2 (VRAM vs HBM
  channel aliasing).
- **LDS `sX` bank-conflict padding** (`BK+1`) and `sX_swizzle` variant.
- **Repack plane packing** (2×`uint4` per sub-block vs a different alignment) →
  changes `sW_lo/hi` layout.
- **K-stage double-buffering** (stage `2·BK` with ping-pong LDS).
- **Activation LDS read width** (`ggml_cuda_memcpy_1<16>` vs `<32>`).

**NOT sweepable (genuinely fixed):** wavefront = 64 (gfx906 HW);
`QK8_1` = 32 weights / 8 dp4a per sub-block (Q8_0 quant definition, only changes
if you redefine the repack *format*); dp4a count per sub-block = 8 (consequence
of Q8_0).
