# Repack Q8_0 — peak-performance proposals (honest & critical)

This document collects proposals to push the Q8_0 repack **prefill (MMQ) kernel
to higher peak throughput** at large N (where repack already wins ~+10% over
native, per the A/B in `bench_results.md`). It is deliberately critical: several
ideas below are plausible but *unvalidated*, and the real bottleneck may not be
where it looks.

---

## 0. Honesty caveat — we do not yet know what is binding

The A/B shows repack at **+10.5%** over native at `pp2048` (`ubatch=2048`). That
is a *relative* number. It tells us nothing about how close the kernel is to the
**gfx906 HBM bandwidth roof** (~1 TB/s on MI50/MI60). If repack is already at
90% of achievable bandwidth, the headroom is a few percent; if it is at 40%,
there is a lot.

**Before optimizing anything, measure:**
1. **Achieved HBM bandwidth** of the kernel (counter, or byte/flop accounting:
   weights ≈ `1.0625 B/weight × params` read once per prompt, activations
   `N×K` bytes, divided by the measured pp time). Compare to the ~1 TB/s ceiling.
2. **Where the time actually goes** — is prefill MMQ even the dominant phase at
   `ubatch≥512`, or are the `quantize_mmq_q8_1_cuda` pass, attention, or the
   *decode* (matvec) path eating the budget? The A/B measures prefill only.
3. **Roofline** — confirm the kernel is bandwidth-bound (not latency- or
   occupancy-bound) before spending effort on latency hiding.

Without (1)–(3) these proposals are shots in the dark. Several below could yield
<2% or even regress.

---

## 1. The real binding constraint: LDS, not the tile

A critical correction to the earlier tuning discussion: **LDS is the coupling
resource that makes most "obvious" wins mutually exclusive.** Per block today:

```
sW_lo/hi/d  ≈ 9 KB      (BM=64, BK=4)
sX          ≈ 22.5 KB   (BN=128, BK+1)
total        ≈ 31.5 KB  of 64 KiB
```

Consequences that constrain the proposals:

- **Occupancy = 2 needs 2× LDS per CU.** Two blocks/CU ⇒ `2 × 31.5 KB ≈ 63 KB`,
  which *just* fits 64 KiB — but any larger tile, or double-buffering, blows it.
  So "raise occupancy" and "bigger tile" and "double-buffer" cannot all be true
  at once. This also means my earlier offhand "occupancy=2 is safe" was
  incomplete: it is only safe at the *current* small tile, and it is the thing
  most likely to be killed by any other change.
- **Double-buffering doubles sW+sX** ⇒ ~63 KB at the current tile, leaving zero
  headroom and making occupancy=1 mandatory. So double-buffer trades occupancy
  for overlap.

This is the central tension: **latency hiding (occ=2, double-buffer) and reuse
(bigger tile) both spend the same 64 KiB.** Pick one axis.

---

## 2. Ranked proposals (with honest pros/cons)

### 2.1 Double-buffer the K-loop — *highest leverage, but LDS-expensive*
Overlap the next `BK` stage's global→LDS fill with the current stage's compute.
gfx906 has **no `cp-async`**, so this is a *manual* ping-pong: two `sW`/`sX`
regions, alternating, with carefully placed `__syncthreads`.

- **Pro:** classic win for bandwidth-bound GEMM; removes the per-stage DMA stall.
- **Con:** ~2× LDS ⇒ forces occ=1 and forbids any tile growth; also more
  registers/scratch and a real **correctness risk** (the ping-pong index and the
  two syncthreads must be exactly right or you read staged data mid-fill).
- **Validation:** implement only after (§0) confirms the kernel is stall-bound,
  not ceiling-bound. Measure with both occ=1 and the current tile.

### 2.2 Bigger tile within LDS (BM=128/256, BN=128) — *reuse, not overlap*
More weight reuse per stage, fewer blocks, fewer `__syncthreads` passes, better
L2 hit rate. The sweep already covers BM∈{128,256}.

- **Pro:** if the limiter is barrier/launch overhead or L2 thrash, this helps.
- **Con:** at BM=128/BN=128 LDS ≈ 41 KB; at BM=256 ≈ 57 KB — both *kill*
  occupancy=2 and leave no room for double-buffer. Also a bigger tile is *worse*
  at small N (the very regression we already have), so it must stay N-selectable
  or it broadens the loss region.
- **Validation:** the tile sweep (`SCRIPT_llama_bench_TILE_SWEEP.sh`) is the
  right vehicle — but it currently only benchmarks a dense model at one prompt
  and **does not patch the ID/matvec paths** (see `repack_tuning.md` §4). Run it
  before concluding.

### 2.3 Occupancy = 2 — *latency hiding, LDS-tight*
`__launch_bounds__(512, 2)`.

- **Pro:** ~16 waves/CU hides HBM latency better than the current 8.
- **Con:** only fits at the *current* 31.5 KiB tile; conflicts with 2.1 and 2.2.
  Also risks VGPR spill (compiler may exceed the ~32 VGPR/thread budget for
  16 waves ⇒ register spilling ⇒ *slower*). Must verify no spill via
  `cuobjdump`/`--save-temps` register report.
- **Validation:** try alone at the current tile; check VGPR and that LDS×2 < 64 KiB.

### 2.4 Larger BK (8) — *fewer barriers*
Halves K-loop iterations ⇒ fewer `__syncthreads` round-trips for the same K.

- **Pro:** cheap; LDS grows only for `sW` (still fits).
- **Con:** more LDS per stage and more registers per thread (8 sub-blocks staged
  at once); the win is small unless barriers are a measured bottleneck. Also
  interacts badly with double-buffer (even more LDS).
- **Validation:** try in the sweep (BK∈{2,4,8} already covered).

### 2.5 Fix activation read coalescing — *hidden bandwidth tax*
`rp_xq_from_mmq` reads the **transposed** `block_q8_1_mmq` buffer; the gather
pattern may not be 128-bit coalesced for the `sX` fill, and the activations are
re-read every K-stage.

- **Pro:** if the activation fill is the bandwidth bottleneck (it scales with N,
  unlike weights which are reused), this is where large-N peak is actually lost.
- **Con:** requires touching the `quantize_mmq_q8_1_cuda` output layout — a second
  file, second kernel, second set of correctness tests. Easy to break MoE.
- **Validation:** profile the `sX` fill bandwidth vs the `sW` fill; only worth it
  if `sX` dominates.

---

## 3. What probably will NOT help (be skeptical)

- **Making the tile N-selectable** — that is a *small-N* fix (closes the −53%..−6%
  regression). It does **nothing** for peak. Do not conflate the two goals.
- **More dp4a micro-optimization** (e.g. different int8 packing) — the kernel is
  not compute-bound at peak, so this is wasted effort.
- **Bigger BN alone** — BN is already 128 and LDS-capped; 256 would blow LDS and
  force occ=1 with no overlap benefit.
- **Tuning the matvec (decode) path** — decode is a separate kernel; optimizing
  prefill peak does not move decode, where repack currently *loses* ~6%.

---

## 4. Honest bottom line

- Repack already **beats native at peak (+10%)**, so the absolute upside over
  native is modest — probably single-digit % once you're near the BW roof.
- The kernel is almost certainly **bandwidth/latency bound**, and the only
  durable levers are **latency hiding (occ=2 or double-buffer) vs weight reuse
  (bigger tile)** — which fight over the same 64 KiB LDS. You cannot max both.
- **Recommended sequence:** (1) profile achieved BW and confirm stall-bound
  (§0); (2) try **occupancy=2 at the current tile** (cheapest, safe-ish); (3) if
  stall-bound, prototype **double-buffer** (highest ceiling, highest risk);
  (4) let the **tile sweep** decide BM/BN/BK rather than guessing. Skip activation
  coalescing unless profiling shows `sX` is the bottleneck.
- **Reality check:** if real workloads are decode-dominated, optimizing prefill
  peak may not improve end-to-end latency at all — and repack's decode loss would
  then dominate. Measure a realistic (pp + tg) mix before declaring victory.

---

## 5. Open questions to resolve before coding

1. What is the achieved HBM bandwidth of the current repack MMQ at `ubatch=2048`?
2. Is the kernel stall-bound (double-buffer helps) or ceiling-bound (nothing
   helps much)?
3. At `ubatch≥512`, what fraction of end-to-end time is prefill MMQ vs the mmq
   quantizer vs attention vs decode?
4. Does the `sX` (activation) fill or the `sW` (weight) fill dominate LDS
   traffic?
