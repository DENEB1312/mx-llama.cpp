# Q8_0 Repack PP MMQ — Open Questions

This document captures what we know, what we tried, and what we don't yet know
about optimizing `mmq_gemm_q8_0_repacked` (`repack-gcn.cu:1744`). Everything
is framed as questions. Nothing is declared impossible until it's been proved
impossible — and we haven't proved most of these impossible.

The attempt log (what was tried and what happened) lives in
`REPACK_Q8_PP_MMQ_OPTIMIZATION.md`. This file is the *forward-looking*
questioning document.

---

## 1. What is the kernel actually doing?

### 1.1 The tile

| Parameter | Value | Source |
|-----------|-------|--------|
| BM (weight rows per block) | 128 | `MMQ_RP_Q8_BM` at `repack-gcn.cu:1095` |
| BN (activation cols per block) | 128 | `MMQ_RP_Q8_BN = 64 * TN_` at `:1096`, TN_=2 |
| BK (K-sub-blocks per LDS fill) | 4 | `MMQ_RP_Q8_BK` at `:1093` |
| Threads per block | 512 | `dim3(64, 8)` at `:2079` |
| NROW_LANES | 8 | `:1097` |
| NROW = BM / NROW_LANES | 16 | rows per thread |
| TN_ | 2 | columns per thread |
| launch_bounds | `(512, 1)` | `:1744` → 1 block/CU, 8 wavefronts/SIMD |
| Accumulators per thread | `acc[16][2]` = 32 floats | `:1790` |
| VGPR/thread | 88 | from rocprofv3 CSV |
| LDS/workgroup | 41,472 bytes (40.5 KiB) | from rocprofv3 CSV |

### 1.2 The LDS arrays

```
__shared__ uint4      sW_lo[128][4];   // 8 KiB  — weight low 16 bytes
__shared__ uint4      sW_hi[128][4];   // 8 KiB  — weight high 16 bytes
__shared__ float      sWd  [128][4];   // 2 KiB  — weight scale (fp16→float)
__shared__ block_q8_1 sX   [128][5];   // 23 KiB — activation (ds + 32-byte qs)
//                               ^^^
//                    +1 padding column for bank-conflict mitigation
//                    block_q8_1 = 36 bytes (half2 ds + int8 qs[32])
//                    row stride = 5 × 36 = 180 bytes = 45 int32 words
```

### 1.3 The inner loop (the hot path)

```cpp
// repack-gcn.cu:1852-1895 (after qs-hoist, Attempt J)
for (int kk = 0; kk < BK=4; kk++) {
    float dx[2];  int xq32_cached[2][8];   // hoisted activation
    for (n = 0; n < 2; n++) {
        xb = &sX[sX_swizzle(tx + n*64)][kk];
        dx[n] = __low2float(xb->ds);
        xq32_cached[n] = {uint4×2 from xb->qs};  // 2× ds_read_b128
    }
    for (r = 0; r < 16; r++) {             // 16 weight rows
        wlo = sW_lo[ty + r*8][kk];          // broadcast — 0 conflict
        whi = sW_hi[ty + r*8][kk];          // broadcast — 0 conflict
        d   = sWd  [ty + r*8][kk];          // broadcast — 0 conflict
        for (n = 0; n < 2; n++) {           // 2 activation columns
            idot = 0;
            for (j = 0; j < 4; j++) {       // 8 dp4a, single accumulator
                idot = dp4a(lo[j],   xq32[j],   idot);  // 4× sdot4
                idot = dp4a(hi[j],   xq32[j+4], idot);  // 4× sdot4
            }
            acc[r][n] += d * dx[n] * (float)idot;
        }
    }
}
```

**Per K-chunk per wavefront:** 16 rows × 2 cols × 8 dp4a = 256 dp4a + 32 FMA.
**Per dispatch (4 K-chunks):** 1024 dp4a + 128 FMA per wavefront.

### 1.4 What the PMC counters say

From rocprofv3 profiling (2 passes, Qwen3-4B Q8_0, pp2048, repack on):

| Counter | Total (498 dispatches) | Per wavefront |
|---------|----------------------|---------------|
| GRBM_COUNT (GPU cycles) | 3,003,989,841 | 1,386 |
| SQ_WAVES | 2,167,808 | — |
| SQ_INSTS_VALU | 81,373,293,568 | 37,537 |
| SQ_INSTS_LDS | 11,676,672,000 | 5,386 |
| SQ_LDS_BANK_CONFLICT | 1,000,857,600 | 462 |
| SQ_ACTIVE_INST_VALU | 81,720,032,865 | — |
| SQ_THREAD_CYCLES_VALU | 5,230,081,998,848 | — |
| SQ_WAIT_INST_LDS | 8,442,023,820 | 3,887 |

**Key ratios:**
- VALU utilization: 100% (all 64 threads active, zero divergence)
- LDS inst / VALU inst: 14.3%
- Bank conflict overhead: **~8% of GPU time** (462 conflict-cyc/wf)
- LDS wait overhead: 0.06% (negligible)
- dp4a throughput: ~79% of peak (the kernel is compute-bound)

---

## 2. Where are the bank conflicts?

### 2.1 What we know for certain

The sX staging writes (`repack-gcn.cu:1836-1848`) are the dominant conflict
source. Each of 512 threads writes one `block_q8_1` (36 bytes) to
`sX[sX_swizzle(lr)][lk]`. The compiler emits 3 LDS instructions per write:
- `ds_write_b32` for `ds` (4 bytes @ offset 0)
- `ds_write_b128` for `qs[0..15]` (16 bytes @ offset 4)
- `ds_write_b128` for `qs[16..31]` (16 bytes @ offset 20)

The sX_swizzle function (`repack-gcn.cu:1746-1751`):
```cpp
static __device__ __forceinline__ int sX_swizzle(int lr) {
    const int n  = lr >> 6;        // 0 or 1 (upper half)
    int       tx = lr & 63;        // 0..63 within half
    tx ^= (tx >> 5) << 4;         // XOR 16 when tx >= 32
    return (n << 6) | tx;
}
```

This swizzle was designed to eliminate within-wave bank conflicts on the
**compute reads** (the hoisted `sX[sX_swizzle(tx + n*64)][kk]` at `:1866`).
It succeeds: within each 64-lane wavefront, all 16 ds reads and all qs reads
hit distinct banks. **No within-wave conflict on reads.**

### 2.2 The cross-wave conflict (the open question)

The 8 wavefronts in a block write to sX simultaneously (they're all at the
same `__syncthreads()` barrier at `:1850`). The sX_swizzle maps each wave's
16 threads to 16 sXr values. The critical question:

**Do different waves' sXr values map to the same LDS banks?**

Bank index for ds write = `(sXr × 45) / 4 mod 32 = (sXr × 9) mod 32`.

With the current swizzle, the sXr sets per wave are:
- Wave 0: {0..15} → ds banks {0,3,4,7,8,9,12,13,17,18,21,22,26,27,30,31}
- Wave 1: {16..31} → ds banks {1,2,5,6,10,11,14,15,16,19,20,23,24,25,28,29}
- Wave 2: {48..63} → ds banks {1,2,5,6,10,11,14,15,16,19,20,23,24,25,28,29}
- Wave 3: {32..47} → ds banks {0,3,4,7,8,9,12,13,17,18,21,22,26,27,30,31}
- (waves 4-7 repeat the pattern)

**Waves 0, 3, 4, 7 share one bank set. Waves 1, 2, 5, 6 share another.**
This is a **4-way bank conflict** on the ds write. The LDS arbiter must
serialize 4 waves writing to the same 16 banks → 3 extra cycles per bank.

The same pattern applies to the qs writes (2-way conflict per uint4).

### 2.3 Why does this happen?

The sXr sets for waves 0 and 3 differ by 32 (wave 0: {0..15}, wave 3: {32..47}).
For any stride S, `(sXr + 32) × S mod 32 = sXr × S mod 32` (since 32×S ≡ 0
mod 32 for all S). So any two sXr sets that differ by a multiple of 32 produce
identical bank sets, regardless of the row stride.

With 8 waves of 16 rows each (128 rows total), the sXr partition is forced:
{0..15}, {16..31}, {32..47}, {48..63}, {64..79}, {80..95}, {96..111}, {112..127}.
Waves 0 and 3 both contain elements that are 32 apart (0 and 32, 1 and 33, etc.).

**Is this a fundamental limitation of 8 waves × 16 rows × 32 banks?**

### 2.4 The question we haven't answered

We proved that **the current sX_swizzle** creates a 4-way conflict. We proved
that **any sX_swizzle based on `lr & 63`** (the current approach) has this
property. We proved that **any odd row stride** still produces the same bank
sets for sXr sets differing by 32.

**But we haven't proved that NO swizzle/layout/tile-geometry can avoid this.**

Specifically, these approaches remain unexplored:

#### Q2.4.1: Can we change the sXr partition to avoid the 32-apart property?

Current partition: waves own consecutive 16-row blocks ({0..15}, {16..31}, ...).
What if waves owned *interleaved* rows? E.g., wave 0 owns rows {0, 8, 16, 24, ...},
wave 1 owns {1, 9, 17, 25, ...}. The sXr sets would be non-consecutive, and
the 32-apart property might break.

**Status:** unexplored. The staging loop (`:1825-1848`) computes `lr = e/4`
linearly; changing to interleaved would require reworking both the staging
write and the compute read (`:1866`). The compute read uses `tx + n*64` as
the row index — the interleaving must be consistent between write and read.

#### Q2.4.2: Can we change block_q8_1 size to change the bank mapping?

If `block_q8_1` were 40 bytes instead of 36 (adding 4 bytes of padding after
`ds`), the row stride would be 5 × 40 = 200 bytes = 50 int32 words. Then
`(sXr × 50) mod 32 = (sXr × 18) mod 32`. Since gcd(18, 32) = 2, the bank
sets for even and odd sXr would be disjoint — but waves 0 and 2 would still
share banks (since 32 × 18 ≡ 0 mod 32).

**Question:** is there a block_q8_1 size where the 32-apart property breaks?
Answer: no — any stride S satisfies 32×S ≡ 0 mod 32. The only escape is to
make the sXr sets NOT differ by multiples of 32, which requires non-consecutive
partitioning (Q2.4.1).

#### Q2.4.3: Can we separate ds and qs into different LDS arrays?

If `ds` lived in `sX_ds[128][5]` (float, stride 20 bytes = 5 ints) and `qs`
lived in `sX_qs[128][5]` (32 bytes, stride 160 bytes = 40 ints), the bank
patterns would be different. The ds write would have stride 5 → bank = (sXr×5)
mod 32. Since gcd(5, 32) = 1, the bank mapping is bijective — but the 32-apart
property still holds (32×5 ≡ 0 mod 32).

**Status:** unexplored. Would require changing the `block_q8_1` struct or
using a custom staging format. The qs write stride (40 ints) has gcd(40,32) = 8,
which is worse (8-way conflict on qs). This approach likely makes things worse.

#### Q2.4.4: Can we reduce the number of waves to eliminate the conflict?

With 4 waves (256 threads, `launch_bounds(256,2)`), each wave owns 32 rows.
Bank sets: wave 0 = {0..31}, wave 1 = {32..63}, wave 2 = {64..95}, wave 3 = {96..127}.
With stride 45: wave 0 banks = {(0×9)..(31×9)} mod 32 = all 32 banks.
Wave 1 banks = {(32×9)..(63×9)} mod 32 = all 32 banks. **Still 4-way conflict.**

With 2 waves (128 threads): each wave owns 64 rows. Bank sets: wave 0 = {0..63},
wave 1 = {64..127}. Both span all 32 banks. **Still 2-way conflict.**

**Question:** is there ANY wave count where the bank sets are disjoint?
Answer: only if each wave's sXr set maps to ≤16 banks and the sets are disjoint.
With 32 banks and 8 waves, each wave can use at most 4 banks — but each wave
has 16 rows, and stride 45 maps 16 consecutive sXr values to 16 distinct banks.
So each wave uses 16 banks, and 8 × 16 = 128 > 32. **Pigeonhole: conflict is
unavoidable with ≥3 waves and stride coprime to 32.**

This is the closest thing to a proof we have. But it only applies to
*consecutive* sXr partitions. Interleaved partitions (Q2.4.1) might break it.

#### Q2.4.5: Can we use a different swizzle that makes waves 0 and 3 hit different banks?

The swizzle `tx ^= (tx >> 5) << 4` was chosen to eliminate within-wave read
conflicts. What if we used a different swizzle that also accounts for cross-wave
conflicts? E.g., `tx ^= ((tx >> 5) << 4) ^ (wave_id << 2)` where `wave_id`
is derived from the staging loop's `lr`.

**Status:** unexplored. The compute read at `:1866` uses `tx + n*64` as the
row index — it doesn't know which wave wrote the data. The swizzle must be
a pure function of `lr` (or `tx`) alone, not of the wave index. But the
staging write CAN use a wave-dependent swizzle, as long as the compute read
uses the same function.

**The key constraint:** the staging write and compute read must agree on the
sXr for a given row. If the staging write uses `sX_swizzle(lr)` and the
compute read uses `sX_swizzle(tx + n*64)`, they must produce the same sXr.
Since `lr` in the staging loop ranges 0..127 and `tx + n*64` in the compute
loop also ranges 0..127, the function must be the same. But the staging loop
processes `lr` in wave-dependent chunks (wave 0: lr=0-15, wave 1: lr=16-31,
etc.), so the function CAN be wave-dependent — as long as the output is the
same for the same `lr` value regardless of which wave computes it.

**This means: the sXr values ARE determined solely by `lr`, and different
waves DO write to different sXr values (since they own different `lr` ranges).
The conflict is not about writing to the same address — it's about writing
to different addresses that happen to be in the same bank.**

#### Q2.4.6: Can we change the block_q8_1 ds field to float (4 bytes → 4 bytes)?

Currently `ds` is `half2` (4 bytes). If we changed it to `float` (4 bytes),
the size wouldn't change but the read pattern might. This doesn't help with
bank conflicts (same size, same stride).

#### Q2.4.7: Can we use LDS read-modify-write to avoid the conflict?

Instead of writing ds and qs separately, what if we wrote the entire 36-byte
`block_q8_1` as a single `ds_write_b256` (32 bytes) + `ds_write_b32` (4 bytes)?
On gfx906, `ds_write_b256` doesn't exist — the widest is `ds_write_b128` (16 bytes).
So we're stuck with 3 writes per element.

#### Q2.4.8: Can we use a different tile shape to reduce the conflict?

The current tile is 128×128 (BM=128, BN=128). What about:
- 128×64 (BM=128, BN=64): 4 waves instead of 8. Attempt E showed -3.3% regression
  from reduced in-block ILP. But the bank conflict would be 2-way instead of 4-way.
  Net: conflict cost halves (~4%), but ILP loss costs ~3%. Might be neutral or slight win.
- 64×128 (BM=64, BN=128): 4 waves. Same analysis.
- 64×64 (BM=64, BN=64): 2 waves. 2-way conflict. But even less ILP.

**Status:** Attempt E tried 128×64 and regressed. But that was without the
swizzle. With the swizzle eliminating within-wave read conflicts, the balance
might shift. **Unexplored with swizzle.**

#### Q2.4.9: Can we use a different GEMM algorithm entirely?

The native `mul_mat_q` uses a different tiling: `mmq_x=64` (on gfx906 for Q8_0),
`mmq_y=128`, `nwarps=8`, `__launch_bounds__(512,2)`. It uses stream-K work
partitioning. The repack kernel uses a simpler 2D grid.

**Question:** could we port the native tiling (64×128, stream-K, 2 blocks/CU)
to the repack path? The native kernel has `load_tiles_q8_0` which reads
`block_q8_0` directly — the repack equivalent would read from the repacked
planes. The LDS layout would be different (native uses a flat `tile_x` array
with `2*MMQ_TILE_NE_K+1` stride per row).

**Status:** unexplored. This is a kernel rewrite, not a swizzle tweak. But
it might avoid the bank conflict entirely by using a different LDS geometry.

---

## 3. The dp4a utilization question

### 3.1 What does "79% of peak" mean?

The roofline (§2 of the attempt log) says the kernel runs at ~79% of MI50's
int8 dp4a peak. The PMC confirms VALU utilization at 100% (all 64 threads
active). The 21% gap is NOT from idle threads or divergent branches.

**Where does the 21% go?**

The inner loop has 8 dp4a per (r, n) on a single `idot` accumulator. Each
`sdot4` has ~4-cycle latency on gfx906. The 8 dp4a are chained (each depends
on the previous result), so the chain takes ~32 cycles. With 8 wavefronts
per SIMD, the dp4a pipeline should be fully utilized (8 waves × 4 cycles =
32 cycles of latency hiding).

**But is it?** The PMC shows `SQ_ACTIVE_INST_VALU = 81.7B` (quad-cycles) and
`SQ_THREAD_CYCLES_VALU = 5.23T` (thread-cycles). The ratio is 100% — meaning
every VALU cycle has all 64 threads active. This suggests the dp4a pipeline
IS fully utilized, and the 21% gap is from non-dp4a instructions (FFMA,
addressing, control flow).

### 3.2 The question: can we reduce non-dp4a overhead?

The kernel has:
- 256 dp4a per K-chunk per wavefront
- 32 FMA per K-chunk per wavefront (`acc += d * dx * idot`)
- LDS staging overhead (writes + reads)
- Loop control (kk, r, n, j indices)

The FMA is 32/256 = 12.5% of the dp4a count. If the FMA and dp4a can be
pipelined (they use different execution units on GCN — VALU dp4a vs VALU FMA),
the FMA should be hidden. But if they compete for the same VALU pipe, the
FMA adds 12.5% overhead.

**Question:** can we restructure the inner loop so the FMA is overlapped with
the dp4a of the NEXT iteration? Currently the loop is:
```
idot = dp4a_chain(...)   // 32 cycles
acc += d * dx * idot     // 1 FMA, ~4 cycles
```
If we could start the next dp4a chain before the FMA completes, the FMA
would be hidden. But the FMA depends on `idot` (the dp4a result), so it
can't start until the chain finishes. The next iteration's dp4a chain
depends on the next `lo[j]` and `xq32[j]` (from LDS), which are independent
of the current FMA. So the compiler COULD interleave:
```
idot1 = dp4a_chain(iter1_lo, iter1_xq)  // 32 cycles
idot2 = dp4a_chain(iter2_lo, iter2_xq)  // starts at cycle 32, finishes at 64
acc += d * dx * idot1                    // starts at cycle 32, finishes at 36
// idot2 finishes at cycle 64, FMA for iter2 starts at 64
```
This would hide the FMA if the compiler schedules it correctly.

**Status:** unknown. We haven't inspected the generated GCN assembly to see
if the compiler is actually interleaving FMA with dp4a across iterations.
This is a **read-the-assembly** question.

---

## 4. The stream-K question

The native `mul_mat_q` kernel uses stream-K work partitioning
(`mmq.cuh:3632`, https://arxiv.org/abs/2301.03598). The repack kernel
uses a plain 2D grid.

**Question:** could stream-K help the repack kernel?

**What stream-K does:** instead of each block owning one output tile and
iterating over all K sub-blocks, stream-K breaks K into smaller chunks and
distributes them across a 1D grid. Multiple blocks reduce different K-slices
of the same output tile, then atomic-add their partial results. This
eliminates wave quantization — the last wave of tiles that underutilizes CUs.

**Current grid for typical 4B layer (M≈2560, N≈2048, K≈2560):**
- Grid = (2560/128) × (2048/128) = 20 × 16 = 320 blocks
- 60 CUs → 6 waves → last wave has 20 idle CUs (33% waste)

**With stream-K:** 320 tiles × (K/4 K-chunks) = 1280 blocks → 22 waves,
near-perfect utilization in the last wave.

**Why it might not help:**
1. The kernel is dp4a-throughput-bound, not load-bound. Better load balancing
   in the last wave doesn't make each CU faster.
2. gfx906 atomic performance is poor (~100-cycle latency vs ~10 on CDNA).
3. The native kernel already tried and regressed on GCN (the `#if !defined(CDMA)`
   guard at `mmq.cuh:3686`).

**Why it might help:**
1. The 33% waste in the last wave IS real GPU time. For 320 blocks / 60 CUs,
   that's ~5% of total dispatch time.
2. Stream-K could reduce per-block work, improving latency for individual
   dispatches (important for autoregressive decoding where PP latency matters).
3. The atomic overhead might be acceptable if the output tile is small
   (M×N = 32K elements, 32K atomics per dispatch).

**Status:** unexplored for the repack kernel. The native kernel's regression
on GCN was for a different tile geometry (64×128 vs our 128×128). **We haven't
tried stream-K with the repack tile.**

---

## 5. The occupancy question

### 5.1 Current state

`launch_bounds(512, 1)` → 1 block/CU, 8 wavefronts/SIMD, 88 VGPR/thread.
LDS = 40.5 KiB → 1 block fits (64 KiB cap). The `launch_bounds` is what
gives the kernel its VGPR budget (Attempt G showed removing it regresses 15%).

### 5.2 The question: can we fit 2 blocks per CU?

If we could reduce LDS to ~32 KiB, 2 blocks would fit (16 wavefronts/SIMD).
This would double the dp4a latency hiding (16 waves instead of 8).

**Approaches to reduce LDS:**
- **Shrink sX:** `sX[128][5]` = 23 KiB. If we used `sX[128][4]` (remove the +1
  padding), it would be 18.4 KiB. But the +1 padding is what gives the 2-way
  bank conflict on reads (with stride 4×36=144, gcd(144/4, 32)=gcd(36,32)=4,
  a 4-way conflict). So removing the padding would make reads worse.
- **Shrink sW:** `sW_lo + sW_hi` = 16 KiB. If we used `sW_lo[64][4]` (half the
  rows), it would be 8 KiB. But this halves the tile's BM dimension.
- **Use a smaller tile:** 64×128 or 128×64. Attempt E showed 128×64 regresses.
  But that was without the swizzle. **Unexplored with swizzle.**

### 5.3 The question: does 2 blocks actually help?

Attempt E tried 128×64 (4 waves instead of 8) and regressed -3.3%. The
explanation was reduced in-block ILP. But with 2 blocks × 4 waves = 8 waves
total (same as current), the ILP per block is halved but the total wave count
is the same. The question is whether the dp4a pipeline cares about per-block
ILP or total wave count.

**Status:** unexplored. The interaction between per-block ILP, total wave
count, and dp4a latency hiding is not well understood for this kernel.

---

## 6. The assembly question

We have PMC counters but we haven't read the generated GCN assembly for the
inner loop. Key questions that only the assembly can answer:

1. **Is the compiler interleaving FMA with dp4a across loop iterations?**
   (§3.2 above)

2. **How many LDS instructions does the compiler actually emit per K-chunk?**
   The PMC shows 5,386 LDS inst/wf. Our estimate was ~2,000. The discrepancy
   might be from the compiler splitting struct loads into multiple instructions.

3. **Is the compiler hoisting the qs read out of the r loop?**
   Attempt J (manual hoist) was neutral, suggesting yes. But we haven't verified.

4. **What is the actual instruction mix in the inner loop?**
   How many dp4a, FMA, LDS, SALU, branch instructions per K-iteration?

**To answer these:** use `rocprofv3 --disasm` or `llvm-objdump -d` on the
compiled `.so` to inspect the generated GCN assembly for `mmq_gemm_q8_0_repacked`.

---

## 7. The model-size question

All measurements are on Qwen3-4B (M≈2560, N≈2048, K≈2560 for the largest
layer). The kernel behavior might differ for:
- **Larger models (27B, 70B):** larger matmul shapes → more K-chunks per
  dispatch → staging overhead amortized differently. Stream-K might help more.
- **Different quantization mixes:** Q8_0-only (4B) vs mixed Q4_K/Q8_0 (27B).
  The repack kernel handles Q8_0 only; K-quant layers use different kernels.
- **Different batch sizes:** pp2048 vs pp512 vs pp4096. The tile count changes;
  wave quantization changes.

**Question:** does the kernel's performance profile (79% dp4a, 8% bank conflict)
hold across different model sizes and batch sizes?

---

## 8. Summary of open questions

| # | Question | Status | Expected impact |
|---|----------|--------|-----------------|
| Q2.4.1 | Interleaved sXr partition (non-consecutive row ownership) | Unexplored | Could eliminate 4-way → 0-way conflict |
| Q2.4.5 | Wave-aware swizzle (same function, different mapping per wave) | Unexplored | Could eliminate cross-wave conflict |
| Q2.4.8 | 128×64 tile WITH swizzle (Attempt E revisited) | Unexplored | 2-way instead of 4-way conflict |
| Q2.4.9 | Port native tiling (64×128, stream-K) to repack | Unexplored | Different LDS geometry, may avoid conflict entirely |
| Q3.2 | Read GCN assembly for FMA/dp4a interleaving | Unexplored | Could reveal hidden overhead |
| Q4 | Stream-K for repack kernel | Unexplored | ~5% from wave quantization |
| Q5.2 | 2 blocks/CU with smaller tile + swizzle | Unexplored | Better dp4a latency hiding |
| Q6 | Full assembly audit of inner loop | Unexplored | Unknown |
| Q7 | Profile across model sizes | Unexplored | Unknown |

**The most promising unexplored question is Q2.4.1 (interleaved sXr partition).**
If waves own non-consecutive rows that don't have the 32-apart property, the
bank conflict could be eliminated entirely. The implementation requires reworking
both the staging write and compute read to use the same interleaved mapping.
This is a moderate refactor but not a kernel rewrite.

**The second most promising is Q2.4.8 (128×64 tile with swizzle).** Attempt E
regressed -3.3% without the swizzle. With the swizzle eliminating within-wave
read conflicts, the balance might shift — the conflict reduction (4-way → 2-way)
could outweigh the ILP loss. This is a one-line macro change.

---

## 9. What NOT to do (confirmed by evidence)

- Do **not** relayout the activation buffer (Attempt C — neutral, compute-bound).
- Do **not** read weights from global in the K-loop (historical -30% regression).
- Do **not** split lo/hi dp4a accumulators (Attempt D — occupancy regression).
- Do **not** remove `launch_bounds(512,1)` (Attempt G — -15% regression).
- Do **not** increase BK beyond 4 (Attempt F — -2.2% regression).
- Do **not** compare PP runs at different power limits (§0 incident).
- Keep decode on canonical quantizer (Attempt B).

---

## 10. How to test

```bash
# Rebuild after editing repack-gcn.cu
cmake --build build --target llama-bench -j24

# Bench (pp2048 only, repack)
bash SCRIPT_llama_bench.sh

# Bench native (flip line 35 of SCRIPT_llama_bench.sh to GGML_CUDA_REPACK_Q8_0=0)
bash SCRIPT_llama_bench.sh

# Profile with rocprofv3 (2 passes needed)
# Pass 1: instruction/cache counters
rocprofv3 --pmc SQ_WAVES SQ_INSTS_VALU SQ_INSTS_LDS SQ_LDS_BANK_CONFLICT \
    GRBM_COUNT GRBM_GUI_ACTIVE TCC_HIT TCC_MISS \
    -o /tmp/pmc-pass1 --output-format csv --kernel-trace -- \
    ./build/bin/llama-bench -m <model> -ngl 99 -fa 1 -p 2048 -n 0 -r 1

# Pass 2: stall/active counters
rocprofv3 --pmc SQ_ACTIVE_INST_VALU SQ_THREAD_CYCLES_VALU SQ_WAIT_INST_LDS \
    -o /tmp/pmc-pass2 --output-format csv --kernel-trace -- \
    ./build/bin/llama-bench -m <model> -ngl 99 -fa 1 -p 2048 -n 0 -r 1

# Correctness check (greedy decode must match native)
./build/bin/llama-bench -m <model> -ngl 99 -fa 1 -p 2048 -n 0 --seed 42 \
    --samplers topk -topk 1 -r 1
```

**Validity rule:** a PP win only counts if tg128 is unchanged (decode path
untouched). Never compare PP runs at different power limits.
