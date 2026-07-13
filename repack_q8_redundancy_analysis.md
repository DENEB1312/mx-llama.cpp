# Is the Q8_0 repack path redundant? (analysis & how to settle it)

> Context: a benchmark (`pp2048`, Qwen3.5-4B Q8_0, gfx906) showed repack beating
> native by **+10.5%** at ubatch 2048, while losing up to **−53%** at small
> ubatch. An earlier hypothesis claimed repack was "redundant for Q8_0 — just
> raise native's tile cap." This document **retracts and corrects** that
> hypothesis with a verified, factor-based analysis.

---

## 1. Direct answer

**No — repack is not proven redundant for Q8_0.** Its large-N prefill win is a
combination of at least three independent factors; only one of them is
addressable by tuning native MMQ. The other two are repack-specific.

---

## 2. The tile cap is real — but LDS is *not* the blocker (verified)

Native Q8_0 tile selection lives in `mul_mat_q_case` (`mmq.cuh:4164`). The
gfx906-specific override (`mmq.cuh:4173-4191`):

```cpp
if constexpr (type == GGML_TYPE_Q8_0) {
    static const int q8_gfx906_xmax_env = []() {
        const char * env = getenv("GGML_CUDA_MMQ_GFX906_Q8_XMAX");
        return (env == nullptr || env[0] == '\0') ? 96 : atoi(env);
    }();
    type_gfx906_xmax = q8_gfx906_xmax_env;
    if (type_gfx906_xmax > 0 && cc == GGML_CUDA_CC_VEGA20) {
        if (args.ids_dst != nullptr)        mmq_x_max = std::min(mmq_x_max, 64);
        else if (args.ncols_x >= 4096)      mmq_x_max = type_gfx906_xmax;   // env honored
        else                                 mmq_x_max = std::min(mmq_x_max, 64);  // N<4096 HARD-CAPPED 64
    }
}
```

`args.ncols_x` is the N/ubatch (token count). For the A/B (ubatch 16…2048) this
hits the **`else`** branch → `mmq_x_max = 64` unconditionally. The env knob
`GGML_CUDA_MMQ_GFX906_Q8_XMAX` is wired **only into the `ncols_x >= 4096`
branch**, so at the tested ubatch range it does nothing. That 64-vs-128 gap is a
real contributor to repack's +10.5% at N=2048.

**LDS check (verified in `mmq_get_nbytes_shared`, `mmq.cuh:4041`):**

Native's LDS footprint is dominated by `nbs_y = mmq_x * sizeof(block_q8_1_mmq)`.
From `mmq.cuh:48-60`, `block_q8_1_mmq` = `4*QK8_1 + 4*half2` = **144 B**.
With `nwarps=8, warp_size=64`:

| mmq_x | approx LDS used |
|------:|----------------:|
| 64    | ~9.5 KB |
| 128   | ~19 KB |

Both are far under gfx906's opt-in LDS budget. So **the `else`-branch cap — not
LDS — is the only thing stopping native from using 128.** Raising it is viable.
*(Earlier fear that native@128 might be rejected by the `smpbo` LDS check was
checked and is false.)*

---

## 3. Three factors behind repack's Q8_0 win

| # | Factor | Repack | Native@128 would gain it? |
|---|--------|--------|---------------------------|
| A | Tile width (128 vs native's 64 cap) | 128 fixed | **Yes** — cap is raisable |
| B | Plane-layout coalescing | 32 B-aligned qs stream | **No** — native reads 34 B-interleaved `block_q8_0` |
| C | Hand-written micro-kernel (LDS transient staging, hoisted r-loop, occ=8 @ VGPR≈36) | yes | **No** — upstream ggml kernel is a different micro-architecture |

### (B) is bigger than first claimed
`block_q8_0` is **34 B** (32 int8 + 2 B fp16 d) at a 2-byte stride. A 128-wide
weight stream therefore **straddles a 64 B cacheline every sub-block**, roughly
**doubling cacheline traffic** for the same 32 B of weights. Repack's qs-plane is
**32 B-aligned** → ~2× fewer cachelines. This is a *large* effective-BW edge,
not "modest" — and native **cannot** obtain it without repacking.

### (C) is unmeasured
Repack's LDS transient staging, hoisted r-loop and occupancy are hand-tuned for
gfx906. Even at identical 128 width, native's older micro-architecture can
trail. Magnitude unknown — it is the main reason the repack tile *sweep* exists.

**Conclusion:** raising the native cap buys factor (A) only. (B)+(C) stay
repack's, so native@128 will almost certainly **land short of repack's +10.5% at
N=2048**.

---

## 4. Retraction: the "~58% HBM bandwidth" comparison was apples-to-oranges

The "~58% HBM BW" figure in `repack_peak_optimization.md` was measured on the
**matvec (N=1)** path. Comparing it to repack's **prefill** BW is invalid. We
have **not** measured native prefill BW, so the split of repack's win across
(A)/(B)/(C) is currently unquantified. The repack sweep + the test below are how
to quantify it.

---

## 5. Where repack is the best / only option

- **K-quants (Q4_K / Q5_K / Q6_K):** the plane layout separates nibbles/scales →
  a *real, large* bandwidth win native fundamentally cannot match without
  repacking. Repack is clearly useful here.
- **Large-N Q8_0 prefill:** repack currently holds the best known number;
  native@128 is the only candidate to beat it.

## 6. Where repack is clearly harmful

- **Small-N (ubatch < ~512):** fixed 128 tile → up to **−53%**.
- **Decode (N=1):** repack matvec loses **~6%** vs canonical `mmvq`.

---

## 7. How to actually settle "is repack redundant for Q8_0?" (zero code change)

Partition the factors with the **existing** env knob:

1. **A/B at ubatch = 4096** with `GGML_CUDA_MMQ_GFX906_Q8_XMAX=128`.
   At `ncols_x >= 4096` the env is *already* honored (`mmq.cuh:4185`), so native
   runs 128-wide with **no code edit**. Compare `native@4096` vs `repack@4096`:
   - native ≈ repack → (A) dominates → repack redundant for large-N Q8_0.
   - native still trails → (B)/(C) matter → repack stays useful.
2. **Run the repack tile sweep** (`SCRIPT_llama_bench_TILE_SWEEP.sh`) to size
   factor (C) for repack itself.

This is the correct way to answer the question rather than assuming repack is
useless. The earlier one-line "fix" (honor the env in the `else` branch) is still
worth applying for N<4096 testing, but should not be assumed to close the gap.

---

## 8. Source map

| Claim | Location |
|---|---|
| Q8_0 gfx906 tile-cap override | `mmq.cuh:4173-4191` |
| `else`-branch hard cap at 64 | `mmq.cuh:4188` |
| env honored only for N≥4096 | `mmq.cuh:4185` |
| LDS footprint formula | `mmq_get_nbytes_shared`, `mmq.cuh:4041` |
| `block_q8_1_mmq` = 144 B | `mmq.cuh:48-60` |
| tile-width selection loop | `mul_mat_q_case`, `mmq.cuh:4210` |
| repack fixed 128 tile | `repack-gcn.cu` `MMQ_RP_Q8_BN=128` |
| benchmark numbers | `bench_results.md` |
