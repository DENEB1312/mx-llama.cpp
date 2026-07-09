#include "tp-allreduce.cuh"

#include <type_traits>

namespace ggml_cuda_tp {

// ============================================================================
// Barrier primitives — GPU-side synchronization across devices
// ============================================================================
//
// Protocol (per CTA block):
//   1. Writer: store incremented flag to PEER's signal buffer (cross-device write)
//   2. Reader: poll OWN signal buffer until all peers have written (local read)
//   3. __syncthreads() to synchronize within the block
//
// Signal buffers are in fine-grained/uncached memory so cross-device writes
// are immediately visible without explicit cache flushes.

#if defined(GGML_USE_MUSA)
// ---- MUSA path (compile-only) ----
// This fork does not target MUSA, and the custom AllReduce is never enabled on
// MUSA at run time (the coherence gate is AMD/NVIDIA only). These device helpers
// exist solely so ggml-musa compiles. MUSA cannot parse the PTX inline asm the
// NVIDIA path uses, so we fall back to portable system-scope fences.

static __device__ __forceinline__ void st_flag_volatile(FlagType * flag_addr, FlagType flag) {
    __threadfence_system();
    *reinterpret_cast<volatile FlagType *>(flag_addr) = flag;
}

static __device__ __forceinline__ FlagType ld_flag_volatile(FlagType * flag_addr) {
    FlagType flag = *reinterpret_cast<volatile FlagType *>(flag_addr);
    __threadfence_system();
    return flag;
}

template <int NRANKS>
static __device__ __forceinline__ void barrier_start(
        const RankSignals & sg, Signal * self_sg, int rank) {
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if (threadIdx.x < NRANKS) {
        st_flag_volatile(&sg.signals[threadIdx.x]->start[blockIdx.x][rank], flag);
        while (ld_flag_volatile(&self_sg->start[blockIdx.x][threadIdx.x]) != flag)
            ;
    }
    __syncthreads();
    if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

template <int NRANKS>
static __device__ __forceinline__ void barrier_end(
        const RankSignals & sg, Signal * self_sg, int rank) {
    __syncthreads();
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if (threadIdx.x < NRANKS) {
        *reinterpret_cast<volatile FlagType *>(&sg.signals[threadIdx.x]->end[blockIdx.x][rank]) = flag;
        FlagType val;
        do {
            val = *reinterpret_cast<volatile FlagType *>(&self_sg->end[blockIdx.x][threadIdx.x]);
        } while (val != flag);
    }
    if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

#elif !defined(GGML_USE_HIP)
// ---- NVIDIA CUDA path ----

static __device__ __forceinline__ void st_flag_volatile(FlagType * flag_addr, FlagType flag) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("st.release.sys.global.u32 [%1], %0;" :: "r"(flag), "l"(flag_addr));
#else
    asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" :: "r"(flag), "l"(flag_addr));
#endif
}

static __device__ __forceinline__ FlagType ld_flag_volatile(FlagType * flag_addr) {
    FlagType flag;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
#else
    asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;" : "=r"(flag) : "l"(flag_addr));
#endif
    return flag;
}

template <int NRANKS>
static __device__ __forceinline__ void barrier_start(
        const RankSignals & sg, Signal * self_sg, int rank) {
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if (threadIdx.x < NRANKS) {
        // Write our flag to peer threadIdx.x's signal buffer
        st_flag_volatile(&sg.signals[threadIdx.x]->start[blockIdx.x][rank], flag);
        // Wait until peer threadIdx.x has written to our buffer
        while (ld_flag_volatile(&self_sg->start[blockIdx.x][threadIdx.x]) != flag)
            ;
    }
    __syncthreads();
    if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

template <int NRANKS>
static __device__ __forceinline__ void barrier_end(
        const RankSignals & sg, Signal * self_sg, int rank) {
    __syncthreads();
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if (threadIdx.x < NRANKS) {
        // Relaxed semantics — no downstream reads depend on this barrier's data.
        asm volatile("st.volatile.global.u32 [%1], %0;" :: "r"(flag),
                     "l"(&sg.signals[threadIdx.x]->end[blockIdx.x][rank]));
        FlagType val;
        do {
            asm volatile("ld.volatile.global.u32 %0, [%1];"
                         : "=r"(val)
                         : "l"(&self_sg->end[blockIdx.x][threadIdx.x]));
        } while (val != flag);
    }
    if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

#else
// ---- AMD HIP/ROCm path ----
// On MI50/gfx906 PCIe (no XGMI), RELEASE/ACQUIRE is required for correctness —
// RELAXED produces garbage even with fine-grained staging (tested). vLLM uses
// RELAXED on MI300X where XGMI provides HW cache coherence. PCIe has none;
// we rely on the C++11 atomic ordering to emit the right fences.
//
// SYSTEM scope on both sides is required: the load must re-fetch HBM each
// iteration (DEVICE scope + RELAXED can cache in L1/L2 even for "uncached"
// memory under some access patterns).

template <int NRANKS>
static __device__ __forceinline__ void barrier_start(
        const RankSignals & sg, Signal * self_sg, int rank) {
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if (threadIdx.x < NRANKS) {
        __hip_atomic_store(
            &sg.signals[threadIdx.x]->start[blockIdx.x][rank],
            flag, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        while (__hip_atomic_load(
                   &self_sg->start[blockIdx.x][threadIdx.x],
                   __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < flag)
            ;
    }
    __syncthreads();
    if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

template <int NRANKS>
static __device__ __forceinline__ void barrier_end(
        const RankSignals & sg, Signal * self_sg, int rank) {
    __syncthreads();
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if (threadIdx.x < NRANKS) {
        __hip_atomic_store(
            &sg.signals[threadIdx.x]->end[blockIdx.x][rank],
            flag, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        while (__hip_atomic_load(
                   &self_sg->end[blockIdx.x][threadIdx.x],
                   __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < flag)
            ;
    }
    if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

#endif // GGML_USE_HIP

// ============================================================================
// Wire-type conversion helpers (BF16-on-wire support)
// ============================================================================
//
// The peer-write kernels (broadcast / twoshot) convert F32 -> T_wire at the
// wire boundary and keep the *local reduction in F32*.  The wire format only
// ever holds a single rank's contribution briefly in fine-grained staging;
// the sum is computed in F32 and written back as F32.  This means the AR is
// lossless in *range* — T_wire = nv_bfloat16 carries BF16's 8-bit exponent
// (same range as F32), so no element overflows where NCCL's BF16 ring would
// also be safe.  Precision is reduced to the wire mantissa width, matching
// NCCL's BF16-ring behaviour, but the PCIe byte count is halved vs F32-on-wire.
//
// T_wire is a compile-time template parameter selected at launch by
// GGML_TP_AR_WIRE (default f32 for unchanged behaviour).

template <typename T_wire>
static __device__ __forceinline__ float wire_to_f32(T_wire x);
template <typename T_wire>
static __device__ __forceinline__ T_wire f32_to_wire(float x);

// Portable, rank-identical F32<->BF16 conversion (round-to-nearest-even).
// Done in software so results are bit-identical on every GPU regardless of
// ROCm's __hip_bfloat16 constructor, and so no BF16 hardware is required
// (gfx906 has none).  The cost is a few ALU ops per element at the wire
// boundary, amortised over the PCIe transfer time.
static __device__ __forceinline__ uint16_t f32_to_bf16_raw(float f) {
    uint32_t x; __builtin_memcpy(&x, &f, 4);
    const uint32_t sign = x & 0x80000000u;
    const uint32_t mag  = x & 0x7FFFFFFFu;
    const uint32_t t    = mag + 0x7FFFu + ((mag >> 16) & 1u);
    return (uint16_t)((t >> 16) | (sign >> 16));
}
static __device__ __forceinline__ float bf16_raw_to_f32(uint16_t bf) {
    uint32_t y = (uint32_t)bf << 16;
    float r; __builtin_memcpy(&r, &y, 4);
    return r;
}
// Reinterpret a 16-bit wire value's bits without going through float casts.
static __device__ __forceinline__ uint16_t wraw(const nv_bfloat16 & x) { uint16_t r; __builtin_memcpy(&r, &x, 2); return r; }
static __device__ __forceinline__ nv_bfloat16 wfrom(uint16_t r) { nv_bfloat16 x; __builtin_memcpy(&x, &r, 2); return x; }

template <> __device__ __forceinline__ float wire_to_f32<float>(float x) { return x; }
template <> __device__ __forceinline__ float f32_to_wire<float>(float x) { return x; }
template <> __device__ __forceinline__ float wire_to_f32<nv_bfloat16>(nv_bfloat16 x) { return bf16_raw_to_f32(wraw(x)); }
template <> __device__ __forceinline__ nv_bfloat16 f32_to_wire<nv_bfloat16>(float x) { return wfrom(f32_to_bf16_raw(x)); }

// wire_traits: type-specific 4-element pack store/load at a wire-element
// offset, plus scalar store/load for the remainder loop.  Keeping the loop
// body identical across wire types and only varying these accessors avoids
// duplicating the kernels.
template <typename T_wire>
struct wire_traits;

template <>
struct wire_traits<float> {
    using ptr_t = float *;
    static __device__ __forceinline__ void store4(float * p, const float4 & v) {
        *reinterpret_cast<float4 *>(p) = v;
    }
    static __device__ __forceinline__ float4 load4(const float * p) {
        return *reinterpret_cast<const float4 *>(p);
    }
    static __device__ __forceinline__ void store1(float * p, float v) { *p = v; }
    static __device__ __forceinline__ float load1(const float * p) { return *p; }
};

template <>
struct wire_traits<nv_bfloat16> {
    using ptr_t = nv_bfloat16 *;
    static __device__ __forceinline__ void store4(nv_bfloat16 * p, const float4 & v) {
        const uint16_t a0 = f32_to_bf16_raw(v.x);
        const uint16_t a1 = f32_to_bf16_raw(v.y);
        const uint16_t a2 = f32_to_bf16_raw(v.z);
        const uint16_t a3 = f32_to_bf16_raw(v.w);
        uint2 u;
        u.x = (uint32_t)a0 | ((uint32_t)a1 << 16);
        u.y = (uint32_t)a2 | ((uint32_t)a3 << 16);
        *reinterpret_cast<uint2 *>(p) = u;
    }
    static __device__ __forceinline__ float4 load4(const nv_bfloat16 * p) {
        const uint2 u = *reinterpret_cast<const uint2 *>(p);
        return make_float4(
            bf16_raw_to_f32((uint16_t)(u.x & 0xFFFF)),
            bf16_raw_to_f32((uint16_t)(u.x >> 16)),
            bf16_raw_to_f32((uint16_t)(u.y & 0xFFFF)),
            bf16_raw_to_f32((uint16_t)(u.y >> 16)));
    }
    static __device__ __forceinline__ void store1(nv_bfloat16 * p, float v) { *p = f32_to_wire<nv_bfloat16>(v); }
    static __device__ __forceinline__ float load1(const nv_bfloat16 * p) { return wire_to_f32<nv_bfloat16>(*p); }
};

// ============================================================================
// One-shot AllReduce kernel
// ============================================================================
//
// Each rank reads from ALL ranks' partial buffers via peer access, sums locally,
// writes result to its own output buffer. Uses 128-bit packed loads (float4)
// for bandwidth efficiency.
//
// Data flow:
//   barrier_start  — ensure all GEMMs have written their partials
//   packed reduce  — read float4 from each peer, sum, write to local output
//   barrier_end    — ensure all reads complete before next op overwrites partials

template <int NRANKS>
__global__ void __launch_bounds__(kThreads, 1)
k_cross_device_reduce_1stage(
        RankData * __restrict__ _dp,
        RankSignals              sg,
        Signal *   __restrict__  self_sg,
        float *    __restrict__  result,
        int                      rank,
        int64_t                  n_elements) {

    // Copy RankData to registers (64 bytes — fits in register file).
    // Passing as a pointer + loading once keeps SGPR pressure low on gfx906
    // (versus pass-by-value, which bloats the kernel arg region and hurts
    // occupancy; measured ~6% TG regression on 4×MI50).
    auto dp = *_dp;

    // Start barrier: all ranks' partial data is ready
    barrier_start<NRANKS>(sg, self_sg, rank);

    // Reduce using float4 (128-bit packed loads/stores). Use non-temporal loads
    // for peer reads on HIP — peer data is one-shot (no reuse) and normal
    // cached loads can return stale L2 entries from prior ARs on MI50 without
    // XGMI cache coherence. NT loads go direct to HBM over PCIe.
    const int64_t n_float4 = n_elements / 4;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_float4;
         idx += (int64_t)gridDim.x * blockDim.x) {

        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
#if defined(GGML_USE_HIP)
            const float * base = ((const float *)dp.ptrs[r]) + idx * 4;
            const float x = __builtin_nontemporal_load(base + 0);
            const float y = __builtin_nontemporal_load(base + 1);
            const float z = __builtin_nontemporal_load(base + 2);
            const float w = __builtin_nontemporal_load(base + 3);
#else
            const float4 v = ((const float4 *)dp.ptrs[r])[idx];
            const float x = v.x, y = v.y, z = v.z, w = v.w;
#endif
            sum.x += x;
            sum.y += y;
            sum.z += z;
            sum.w += w;
        }
        ((float4 *)result)[idx] = sum;
    }

    // Handle remainder (n_elements not divisible by 4)
    const int64_t rem_start = n_float4 * 4;
    for (int64_t idx = rem_start + (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += (int64_t)gridDim.x * blockDim.x) {

        float sum = 0.0f;
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
#if defined(GGML_USE_HIP)
            sum += __builtin_nontemporal_load(((const float *)dp.ptrs[r]) + idx);
#else
            sum += ((const float *)dp.ptrs[r])[idx];
#endif
        }
        result[idx] = sum;
    }

    // End barrier: all reads complete — safe for next AllReduce to overwrite partials
    barrier_end<NRANKS>(sg, self_sg, rank);
}

// ----------------------------------------------------------------------------
// Broadcast + reduce kernel (default for small / TG-size ARs).
// Each rank writes its input into *every peer's* staging at offset
// rank*n_elements via PCIe peer access. PCIe peer writes go directly to the
// peer's HBM (bypassing both source and destination L2), so the producer's
// data is peer-visible without the usual cudaEventRecord/StreamWaitEvent
// L2-flush handshake.
//
// Staging layout (per rank): [N_ranks][n_elements] — rank R's input from peer
// R lives at offset R*n_elements. "Self slot" (R==rank) is unused; we read
// our own contribution from `input` (local L2-coherent read).
//
// Correctness contract: the in-kernel barrier's SYSTEM-scope RELEASE atomic
// flag store is itself a PCIe peer write; PCIe ordering guarantees writes
// from the same source arrive in-order, so the flag arrives after all prior
// data writes. Peer sees our flag → all our prior data writes are in peer's
// HBM.
//
// Each rank's outbound traffic: (N-1)*S bytes of PCIe writes (posted, async,
// low latency). Same aggregate BW as one-shot reads, but writes typically
// beat reads on PCIe 3.0.
// ----------------------------------------------------------------------------
template <typename T_wire, int NRANKS>
__global__ void __launch_bounds__(kThreads, 1)
k_broadcast_reduce(
        RankData * __restrict__ _dp,        // peer (and self) staging pointers
        RankSignals              sg,
        Signal *   __restrict__  self_sg,
        const float * __restrict__ input,   // our own input (local device, F32)
        float *    __restrict__  result,    // our own output (local device, F32)
        int                      rank,
        int64_t                  n_elements) {

    auto dp = *_dp;
    using W = wire_traits<T_wire>;

    // Phase 1: write our input (converted to T_wire) into each PEER's staging
    // at offset rank*n_elements.  We do NOT write the self-slot.  Writing the
    // self-slot would be a local store to fine-grained memory, which
    // empirically does NOT bypass L2 on gfx906 even under FGP=1 — phase 2's NT
    // load from HBM would then miss the fresh self-data.  Reading our own
    // contribution directly from `input` in phase 2 (local L2-coherent read)
    // avoids that path entirely.
    const int64_t n_vec4 = n_elements / 4;
    const int64_t rank_offset = (int64_t) rank * n_elements;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_vec4;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const float4 v = ((const float4 *) input)[idx];
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
            if (r == rank) continue;   // self-slot is NOT peer-written; read `input` directly in phase 2
            typename W::ptr_t dst = (typename W::ptr_t)(uintptr_t)dp.ptrs[r] + rank_offset + idx * 4;
            W::store4(dst, v);         // F32 -> T_wire, peer write (bypasses L2 via PCIe)
        }
    }
    const int64_t rem_start = n_vec4 * 4;
    for (int64_t idx = rem_start + (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const float v = input[idx];
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
            if (r == rank) continue;
            typename W::ptr_t dst = (typename W::ptr_t)(uintptr_t)dp.ptrs[r] + rank_offset + idx;
            W::store1(dst, v);
        }
    }

    // Ensure every thread in this CTA has finished its peer writes before the
    // first NRANKS threads publish the completion flag below.
    __syncthreads();

    // Force all prior peer-writes to retire before we signal via barrier.
    // The RELEASE-scope atomic inside barrier_start should do this on its own,
    // but an explicit system fence is cheap and proved necessary for a 0.18
    // PPL drift that the atomic-alone variant had at ubatch=32.
    __threadfence_system();

    // Barrier: SYSTEM-scope RELEASE flag store is itself a peer write; PCIe
    // same-source ordering guarantees our prior data writes arrive first.
    barrier_start<NRANKS>(sg, self_sg, rank);

    // Phase 2: identical reduction order on EVERY rank:
    //   sum = 0 + p0 + p1 + p2 + ... + p_{N-1}
    // For r == rank we read `input` directly (local L2-coherent, F32); for
    // peers we read T_wire from the per-peer slot in our own staging (peer-
    // written → HBM) and convert back to F32.  The `if (r == rank)` branch
    // inside a #pragma unroll + compile-time NRANKS resolves at compile time
    // per unrolled iteration, so the emitted code has no runtime branch —
    // same accumulation sequence on all ranks.
    typename W::ptr_t local_staging = (typename W::ptr_t)(uintptr_t) dp.ptrs[rank];
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_vec4;
         idx += (int64_t)gridDim.x * blockDim.x) {
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
            float4 v;
            if (r == rank) {
                v = ((const float4 *) input)[idx];   // self contribution
            } else {
                const typename W::ptr_t base = local_staging + (int64_t)r * n_elements + idx * 4;
                v = W::load4(base);
            }
            sum.x += v.x;
            sum.y += v.y;
            sum.z += v.z;
            sum.w += v.w;
        }
        ((float4 *) result)[idx] = sum;
    }
    for (int64_t idx = rem_start + (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += (int64_t)gridDim.x * blockDim.x) {
        float sum = 0.0f;
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
            float v;
            if (r == rank) {
                v = input[idx];
            } else {
                v = W::load1(local_staging + (int64_t)r * n_elements + idx);
            }
            sum += v;
        }
        result[idx] = sum;
    }

    barrier_end<NRANKS>(sg, self_sg, rank);
}

// ----------------------------------------------------------------------------
// Two-shot F32 AllReduce (reduce-scatter + allgather via peer writes).
//
// Target: large (PP-size) messages where the broadcast kernel's (N-1)*S
// per-rank outbound saturates PCIe. Two-shot brings per-rank outbound down
// to 2*(N-1)/N * S, with parallel all-to-all peer writes instead of NCCL's
// serial ring — wins on PCIe systems where every GPU has its own root-
// complex link AND the fabric can carry parallel transfers concurrently.
//
// Wire format (F32 or BF16, via GGML_TP_AR_WIRE): F32 input → T_wire → F32
// reduction (F32 accumulator) → F32 output. F32 wire is fully lossless and
// bit-deterministic (modulo the standard fp non-associativity of summation,
// which is fixed across runs because all ranks sum in the same r-order). BF16
// wire truncates to the BF16 mantissa but keeps BF16's exponent range, so it
// is lossless in range and matches NCCL's BF16-ring precision. F32 wire costs
// 2× the PCIe bytes per AR vs a BF16-on-wire variant; the trade-off vs NCCL
// BF16 ring is BW for precision.
//
// Staging layout (per rank, bytes):
//   [0, n_elements * 4)                    : scatter_buf — N F32 slots, each
//                                            of slice = N_elements/N elements.
//                                            Slot r holds rank r's contribution
//                                            to OUR slice (my_rank). Filled in
//                                            stage 1 by peer writes.
//   [n_elements * 4, 2 * n_elements * 4)   : allgather_buf — N F32 slots, same
//                                            layout. Slot r holds rank r's
//                                            reduced slice. Filled in stage 3
//                                            by peer writes from rank r.
// dp.ptrs[r] points to rank r's scatter_buf base; allgather_buf is at
// +n_elements floats from that base.
//
// Per-rank peer traffic: stage 1 (N-1) * slice F32 + stage 3 (N-1) * slice
// F32 = 2*(N-1)/N * S bytes outbound (S = n_elements * sizeof(float)).
//
// Requires n_elements % NRANKS == 0 and slice % 4 == 0 for float4 path. The
// caller (host dispatch) must gate on these before selecting this kernel.
// ----------------------------------------------------------------------------
template <typename T_wire, int NRANKS>
__global__ void __launch_bounds__(kThreads, 1)
k_twoshot(
        RankData * __restrict__ _dp,        // peer staging base pointers (T_wire*)
        RankSignals              sg,
        Signal *   __restrict__  self_sg,
        const float * __restrict__ input,   // our own input (F32)
        float *    __restrict__  result,    // our own output (F32)
        int                      rank,
        int64_t                  n_elements) {

    auto dp = *_dp;
    using W = wire_traits<T_wire>;

    const int64_t slice = n_elements / NRANKS;          // guaranteed divisible
    const int64_t slice_n_vec4 = slice / 4;
    const int64_t my_slice_start = (int64_t) rank * slice;

    // Staging pointers — each rank's staging holds scatter_buf then
    // allgather_buf (each n_elements T_wire elements).
    //   rank r's scatter_buf base   = dp.ptrs[r]
    //   rank r's allgather_buf base = dp.ptrs[r] + n_elements
    auto scat_ptr = [&](int r) -> typename W::ptr_t { return (typename W::ptr_t)(uintptr_t) dp.ptrs[r]; };
    auto ag_ptr   = [&](int r) -> typename W::ptr_t { return ((typename W::ptr_t)(uintptr_t) dp.ptrs[r]) + n_elements; };

    // ------------------------------------------------------------------------
    // Stage 1 (peer-write scatter): for each non-self target peer, all threads
    // grid-stride over the target's slice of our input, convert F32->T_wire and
    // peer-write the 4-element packs to the target's scatter_buf at OUR slot.
    // Round-robin target order (rank+t) % NRANKS spreads PCIe contention so
    // all ranks write to different targets at the same loop iteration.
    // ------------------------------------------------------------------------
    const int64_t my_slot_offset = (int64_t) rank * slice;
#pragma unroll
    for (int t = 1; t < NRANKS; t++) {
        const int target = (rank + t) % NRANKS;
        const int64_t target_slice_start_f4 = ((int64_t) target * slice) / 4;
        for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
             idx < slice_n_vec4;
             idx += (int64_t)gridDim.x * blockDim.x) {
            const float4 v = ((const float4 *) input)[target_slice_start_f4 + idx];
            typename W::ptr_t dst = scat_ptr(target) + my_slot_offset + idx * 4;
            W::store4(dst, v);   // F32 -> T_wire, 64/128-bit PCIe peer write
        }
    }

    // Ensure every thread in this CTA has finished its peer writes before the
    // first NRANKS threads publish the completion flag below.
    __syncthreads();
    __threadfence_system();
    barrier_start<NRANKS>(sg, self_sg, rank);

    // ------------------------------------------------------------------------
    // Stage 2 + 3 fused: reduce our slice from scatter_buf (T_wire -> F32) +
    // own input (F32), then convert the F32 sum to T_wire and peer-write it to
    // every rank's allgather_buf at OUR slot. Self-contribution is read
    // directly from `input` (F32); peers see our slice as T_wire via stage-1
    // and we expand it back to F32 before accumulating, so the reduction is
    // lossless in range (BF16 wire == BF16 exponent range).
    // ------------------------------------------------------------------------
    typename W::ptr_t own_scat = scat_ptr(rank);
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < slice_n_vec4;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t local_pos = idx * 4;               // position within slice
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
            float4 v;
            if (r == rank) {
                v = ((const float4 *) input)[my_slice_start / 4 + idx];
            } else {
                v = W::load4(own_scat + (int64_t) r * slice + local_pos);
            }
            sum.x += v.x;
            sum.y += v.y;
            sum.z += v.z;
            sum.w += v.w;
        }

        // Allgather peer-write: send our reduced slice (as T_wire) to every
        // PEER's allgather_buf[rank_slot]. We do NOT local-write to OWN
        // allgather_buf — on gfx906/PCIe, kernel-initiated local writes to
        // fine-grained memory dwell in L2 until kernel exit, while stage-4
        // load bypasses L2 and would see stale HBM. Instead write our own
        // slice directly to `result` (lossless, no round-trip).
        const int64_t slot_off = my_slice_start;
#pragma unroll
        for (int t = 1; t < NRANKS; t++) {
            const int target = (rank + t) % NRANKS;
            typename W::ptr_t dst = ag_ptr(target) + slot_off + local_pos;
            W::store4(dst, sum);                         // F32 -> T_wire peer write
        }
        // Own slice: write F32 sum directly to result.
        ((float4 *) result)[my_slice_start / 4 + idx] = sum;
    }

    __threadfence_system();
    barrier_end<NRANKS>(sg, self_sg, rank);
    // barrier_end has a __syncthreads() at its start but NOT at its end —
    // only the first NRANKS threads spin on peer flags. For kernels that READ
    // peer-written data after barrier_end (stage 4), the remaining threads
    // must wait for the spinning threads to confirm peer writes are visible.
    // Without this sync, threads >= NRANKS run stage-4 loads before peers'
    // stage-3 writes have arrived, returning stale HBM and corrupting output.
    __syncthreads();

    // ------------------------------------------------------------------------
    // Stage 4 (local scatter to F32 result): peer slots only (own slice was
    // written directly to `result` in stage 3). Each peer's allgather_buf
    // slot is FGP-coherent (peer writes bypass source L2 -> land in our HBM)
    // so loads return the freshly-arrived T_wire values, expanded to F32.
    // ------------------------------------------------------------------------
    typename W::ptr_t own_ag = ag_ptr(rank);
#pragma unroll
    for (int t = 1; t < NRANKS; t++) {
        const int src_slot = (rank + t) % NRANKS;
        const int64_t slot_base = (int64_t) src_slot * slice;
        for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
             idx < slice_n_vec4;
             idx += (int64_t)gridDim.x * blockDim.x) {
            const float4 v = W::load4(own_ag + slot_base + idx * 4);
            ((float4 *) result)[(slot_base / 4) + idx] = v;
        }
    }
}

// ============================================================================
// Host functions
// ============================================================================

void tp_custom_ar_init(CustomARContext * ctx, int nranks, const int * dev_ids) {
    if (ctx->initialized) return;

    ctx->nranks = nranks;
    for (int r = 0; r < nranks; r++) {
        ctx->dev_ids[r] = dev_ids ? dev_ids[r] : r;
    }

    // Enable peer access between all GPU pairs
    bool peer_access_ok = true;
    for (int i = 0; i < nranks; i++) {
        ggml_cuda_set_device(ctx->dev_ids[i]);
        for (int j = 0; j < nranks; j++) {
            if (i == j) continue;
            int can_access = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, ctx->dev_ids[i], ctx->dev_ids[j]));
            if (can_access) {
                cudaError_t err = cudaDeviceEnablePeerAccess(ctx->dev_ids[j], 0);
                if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                    CUDA_CHECK(err);
                }
                // cudaDeviceEnablePeerAccess leaves the sticky error state set when access
                // was already enabled. Clear it here so the first downstream cudaGetLastError()
                // (e.g. after a MUL_MAT kernel) doesn't trip on it.
                (void) cudaGetLastError();
            } else {
                GGML_LOG_WARN("TP custom AR: peer access not available between GPU %d and %d\n",
                              ctx->dev_ids[i], ctx->dev_ids[j]);
                peer_access_ok = false;
            }
        }
    }
    if (!peer_access_ok) {
        GGML_LOG_WARN("TP custom AR: disabled because full peer access is not available\n");
        return;
    }

    // Allocate signal buffers (one per rank) in fine-grained/uncached memory
    for (int rank = 0; rank < nranks; rank++) {
        ggml_cuda_set_device(ctx->dev_ids[rank]);

#if defined(GGML_USE_HIP)
        // AMD: hipDeviceMallocUncached for cross-device visibility (matches vLLM)
        void * ptr;
        CUDA_CHECK(hipExtMallocWithFlags(&ptr, sizeof(Signal), hipDeviceMallocUncached));
        ctx->d_signals[rank] = (Signal *)ptr;
#else
        // NVIDIA: regular device memory (peer access handles visibility)
        CUDA_CHECK(cudaMalloc(&ctx->d_signals[rank], sizeof(Signal)));
#endif
        CUDA_CHECK(cudaMemset(ctx->d_signals[rank], 0, sizeof(Signal)));

        // Allocate device-side RankData (holds pointers to all ranks' buffers)
        CUDA_CHECK(cudaMalloc(&ctx->d_rank_data[rank], sizeof(RankData)));

        ctx->rank_signals.signals[rank] = ctx->d_signals[rank];

        // Event used for cross-stream handshake before each AR
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx->events[rank], cudaEventDisableTiming));
    }

    // Determine whether the broadcast path is safe on this hardware.
    // The broadcast kernel relies on kernel-initiated PCIe peer writes being
    // visible to the destination rank before the in-kernel barrier's RELEASE
    // flag arrives, i.e. cache-coherent peer writes. This is true on:
    //   - NVIDIA (any CUDA-capable device): NVLink and PCIe P2P are HW-coherent
    //   - AMD gfx90a (MI200), gfx94x (MI300), gfx95x: XGMI is HW-coherent
    //   - any AMD PCIe GPU when HSA_FORCE_FINE_GRAIN_PCIE=1 forces all device
    //     allocations to fine-grained (write-through) memory. The fine-grain
    //     mechanism is generic to AMD, but this peer-write path is validated
    //     only on gfx906 (MI50), so other AMD archs get a warning below.
    // Without FGP (and off XGMI) we stay on the one-shot path, which uses
    // cudaMemcpyAsync staging + an event handshake and is always correct.
    bool hw_peer_write_coherent = false;
#if defined(GGML_USE_HIP)
    {
        hipDeviceProp_t prop;
        CUDA_CHECK(hipGetDeviceProperties(&prop, ctx->dev_ids[0]));
        const char * arch = prop.gcnArchName;   // e.g. "gfx906:sramecc+:xnack-"
        const bool is_gfx906 = (strncmp(arch, "gfx906", 6) == 0);
        const bool is_gfx9_xgmi =
            (strncmp(arch, "gfx90a", 6) == 0) ||
            (strncmp(arch, "gfx94",  5) == 0) ||
            (strncmp(arch, "gfx95",  5) == 0);
        const char * e_fgp = getenv("HSA_FORCE_FINE_GRAIN_PCIE");
        const bool fgp_on = e_fgp && e_fgp[0] != '\0' && e_fgp[0] != '0';
        hw_peer_write_coherent = is_gfx9_xgmi || fgp_on;
        const bool peer_write_experimental = fgp_on && !is_gfx906 && !is_gfx9_xgmi;
        if (peer_write_experimental) {
            GGML_LOG_WARN("TP custom AR: peer-write enabled on %s via HSA_FORCE_FINE_GRAIN_PCIE "
                          "(validated only on gfx906, verify output correctness)\n", arch);
        }
    }
#else
    hw_peer_write_coherent = true;   // CUDA / MUSA: HW-coherent peer access
#endif
    ctx->broadcast_ok = hw_peer_write_coherent;
    ctx->initialized  = true;

    const char * path      = ctx->broadcast_ok ? "broadcast F32 + twoshot F32 (peer-write, size-adaptive, lossless)"
                              :
#if defined(GGML_USE_HIP)
                                "one-shot (PCIe without HSA_FORCE_FINE_GRAIN_PCIE=1, set it to enable fast peer-write paths)"
#else
                                "one-shot (HW not recognised as peer-write coherent)"
#endif
                              ;
    GGML_LOG_INFO("TP custom AllReduce: initialized for %d GPUs, path = %s\n", nranks, path);
}

void tp_custom_ar_destroy(CustomARContext * ctx) {
    if (!ctx->initialized) return;

    // Dump AR-path statistics when requested, so a run can confirm which wire
    // type actually executed (and for what sizes).  oneshot is always F32.
    static const int s_dbg = []{
        const char * e = getenv("GGML_TP_AR_DEBUG");
        return (e && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }();
    if (s_dbg) {
        GGML_LOG_INFO("TP custom AllReduce stats: f32_calls=%llu bf16_calls=%llu "
                      "max_ne_bf16=%lld\n",
                      (unsigned long long)ctx->dbg_n_f32,
                      (unsigned long long)ctx->dbg_n_bf16,
                      (long long)ctx->dbg_max_ne_bf16);
    }

    for (int rank = 0; rank < ctx->nranks; rank++) {
        ggml_cuda_set_device(ctx->dev_ids[rank]);
        if (ctx->events[rank]) {
            CUDA_CHECK(cudaEventDestroy(ctx->events[rank]));
            ctx->events[rank] = nullptr;
        }
        if (ctx->d_signals[rank]) {
#if defined(GGML_USE_HIP)
            CUDA_CHECK(hipFree(ctx->d_signals[rank]));
#else
            CUDA_CHECK(cudaFree(ctx->d_signals[rank]));
#endif
            ctx->d_signals[rank] = nullptr;
        }
        if (ctx->d_rank_data[rank]) {
            CUDA_CHECK(cudaFree(ctx->d_rank_data[rank]));
            ctx->d_rank_data[rank] = nullptr;
        }
        if (ctx->d_staging[rank]) {
#if defined(GGML_USE_HIP)
            CUDA_CHECK(hipFree(ctx->d_staging[rank]));
#else
            CUDA_CHECK(cudaFree(ctx->d_staging[rank]));
#endif
            ctx->d_staging[rank] = nullptr;
        }
    }
    ctx->staging_size = 0;

    ctx->initialized = false;
}

// Per-N kernel launch + nranks dispatch, templated on the wire type T_wire.
// The compiler folds the recursive N chain into a jump table at -O2; adding
// ranks just means bumping kMaxRanks.  One-shot (peer-read) kernels stay F32
// regardless of T_wire, so they are launched whenever s_broadcast is false.
template <typename T_wire>
static void tp_custom_ar_launch(CustomARContext * ctx, int nranks, bool s_twoshot, bool s_broadcast,
                                float ** input_ptrs, float ** output_ptrs, cudaStream_t * streams,
                                int64_t n_elements, int blocks) {
    auto launch_peer = [&](auto N_CONST) {
        constexpr int N = decltype(N_CONST)::value;
        for (int rank = 0; rank < nranks; rank++) {
            ggml_cuda_set_device(ctx->dev_ids[rank]);
            if (s_twoshot) {
                k_twoshot<T_wire, N><<<blocks, kThreads, 0, streams[rank]>>>(
                    ctx->d_rank_data[rank], ctx->rank_signals, ctx->d_signals[rank],
                    input_ptrs[rank], output_ptrs[rank], rank, n_elements);
            } else {
                k_broadcast_reduce<T_wire, N><<<blocks, kThreads, 0, streams[rank]>>>(
                    ctx->d_rank_data[rank], ctx->rank_signals, ctx->d_signals[rank],
                    input_ptrs[rank], output_ptrs[rank], rank, n_elements);
            }
            CUDA_CHECK(cudaGetLastError());
        }
    };
    auto launch_oneshot = [&](auto N_CONST) {
        constexpr int N = decltype(N_CONST)::value;
        for (int rank = 0; rank < nranks; rank++) {
            ggml_cuda_set_device(ctx->dev_ids[rank]);
            k_cross_device_reduce_1stage<N><<<blocks, kThreads, 0, streams[rank]>>>(
                ctx->d_rank_data[rank], ctx->rank_signals, ctx->d_signals[rank],
                output_ptrs[rank], rank, n_elements);
            CUDA_CHECK(cudaGetLastError());
        }
    };

    auto dispatch = [&](auto self, auto N_CONST) -> void {
        constexpr int N = decltype(N_CONST)::value;
        if constexpr (N < 2) {
            GGML_ABORT("TP custom AR: unsupported nranks=%d (must be 2..%d)\n", nranks, kMaxRanks);
        } else if (nranks == N) {
            if (s_broadcast) launch_peer(N_CONST);
            else             launch_oneshot(N_CONST);
        } else {
            self(self, std::integral_constant<int, N - 1>{});
        }
    };
    dispatch(dispatch, std::integral_constant<int, kMaxRanks>{});
}

void tp_custom_ar_allreduce(CustomARContext * ctx,
                            float ** input_ptrs,
                            float ** output_ptrs,
                            int64_t  n_elements,
                            int      nranks,
                            cudaStream_t * streams) {
    GGML_ASSERT(ctx->initialized);
    GGML_ASSERT(nranks == ctx->nranks);
    GGML_ASSERT(nranks >= 2 && nranks <= kMaxRanks);

    // Path selection. broadcast_ok was set at init based on the HW peer-write
    // coherence model (see tp_custom_ar_init).
    // Two-shot (reduce-scatter + allgather) kernel selection for large msgs.
    // F32 on the wire — fully lossless, matches the broadcast kernel's
    // precision and avoids any BF16 conversion. Per-rank outbound is
    // 2*(N-1)/N * S, so 2× the BW of an equivalent BF16-on-wire variant
    // but with parallel all-to-all peer writes (vs NCCL's serial ring).
    //
    // GGML_TP_AR_TWOSHOT can force on (=1) or off (=0); default auto picks
    // twoshot for ne ≥ ~256K elements where broadcast's (N-1)*S/N per-rank
    // outbound starts to lose to twoshot's 2*(N-1)/N*S aggregate parallel BW.
    static const int s_twoshot_env = []{
        const char * e = getenv("GGML_TP_AR_TWOSHOT");
        if (!e || !e[0]) return -1;                      // default = auto
        return (e[0] != '0') ? 1 : 0;
    }();
    const bool s_broadcast = ctx->broadcast_ok;
    // Wire format for the peer-write paths. Default F32 keeps existing
    // behaviour; "bf16" halves PCIe bytes (BF16 carries BF16's 8-bit exponent,
    // so the reduction stays lossless in range) — see wire_traits above.
    static const int s_wire_env = []{
        const char * e = getenv("GGML_TP_AR_WIRE");
        if (e && (strcmp(e, "bf16") == 0 || strcmp(e, "BF16") == 0)) return 1; // bf16
        return 0; // f32
    }();
    const bool s_wire_bf16 = (s_wire_env == 1);
    // Twoshot needs slice-aligned + float4-aligned data: n_elements % (4*nranks) == 0.
    const bool twoshot_eligible =
        s_broadcast && (n_elements % (int64_t)(4 * nranks) == 0);
    const int64_t twoshot_min_ne = 262144;               // ~1 MB F32 crossover
    const bool s_twoshot =
        twoshot_eligible &&
        (s_twoshot_env == 1 || (s_twoshot_env == -1 && n_elements >= twoshot_min_ne));

    const size_t bytes_f32 = (size_t) n_elements * sizeof(float);
    const size_t wire_elem = s_wire_bf16 ? sizeof(nv_bfloat16) : sizeof(float);
    // Broadcast path: wire staging, N slots per rank (one inbox per peer + unused self slot).
    // Two-shot path: wire staging, 2 buffers (scatter_buf + allgather_buf) each
    //   sized n_elements wire elements. One-shot path always stages F32 (it
    //   reads peers' partials directly and never converts on the wire).
    const size_t per_slot   = s_broadcast ? (n_elements * wire_elem) : bytes_f32;
    const int    need_slots = s_twoshot ? 2 : (s_broadcast ? ctx->nranks : 1);
    const size_t need_bytes = per_slot * need_slots;

    if (ctx->staging_slots != need_slots || ctx->staging_size < need_bytes) {
        for (int rank = 0; rank < ctx->nranks; rank++) {
            ggml_cuda_set_device(ctx->dev_ids[rank]);
            if (ctx->d_staging[rank]) {
#if defined(GGML_USE_HIP)
                CUDA_CHECK(hipFree(ctx->d_staging[rank]));
#else
                CUDA_CHECK(cudaFree(ctx->d_staging[rank]));
#endif
                ctx->d_staging[rank] = nullptr;
            }
#if defined(GGML_USE_HIP)
            void * ptr = nullptr;
            CUDA_CHECK(hipExtMallocWithFlags(&ptr, need_bytes, hipDeviceMallocFinegrained));
            ctx->d_staging[rank] = (float *) ptr;
#else
            CUDA_CHECK(cudaMalloc(&ctx->d_staging[rank], need_bytes));
#endif
        }
        ctx->staging_size  = need_bytes;
        ctx->staging_slots = need_slots;
        // invalidate cached RankData pointers → forces re-upload below
        for (int i = 0; i < nranks; i++) ctx->cached_ptrs[i] = nullptr;
    }

    if (!s_broadcast) {
        // One-shot path: SDMA staging + event handshake. Used as a fallback
        // when peer-write coherence is unavailable.
        for (int rank = 0; rank < nranks; rank++) {
            ggml_cuda_set_device(ctx->dev_ids[rank]);
            CUDA_CHECK(cudaMemcpyAsync(ctx->d_staging[rank], input_ptrs[rank], bytes_f32,
                                       cudaMemcpyDeviceToDevice, streams[rank]));
        }
        for (int rank = 0; rank < nranks; rank++) {
            ggml_cuda_set_device(ctx->dev_ids[rank]);
            CUDA_CHECK(cudaEventRecord(ctx->events[rank], streams[rank]));
        }
        for (int rank = 0; rank < nranks; rank++) {
            for (int peer = 0; peer < nranks; peer++) {
                if (peer == rank) continue;
                CUDA_CHECK(cudaStreamWaitEvent(streams[rank], ctx->events[peer], 0));
            }
        }
    }

    // Update device-side RankData only when staging pointers change (rare —
    // staging buffers persist in ctx after first alloc, unless switching
    // between one-shot and broadcast staging layouts).
    bool ptrs_changed = false;
    for (int i = 0; i < nranks; i++) {
        if ((void *) ctx->d_staging[i] != ctx->cached_ptrs[i]) {
            ptrs_changed = true;
            break;
        }
    }
    if (ptrs_changed) {
        RankData h_data;
        for (int i = 0; i < nranks; i++) {
            h_data.ptrs[i] = ctx->d_staging[i];
            ctx->cached_ptrs[i] = ctx->d_staging[i];
        }
        for (int rank = 0; rank < nranks; rank++) {
            ggml_cuda_set_device(ctx->dev_ids[rank]);
            CUDA_CHECK(cudaMemcpy(ctx->d_rank_data[rank], &h_data,
                                  sizeof(RankData), cudaMemcpyHostToDevice));
        }
    }

    // Compute grid dimensions. Block count can be overridden via GGML_TP_AR_BLOCKS
    // for quick perf sweeps; default kDefaultBlocks is tuned per platform in .cuh.
    static const int s_blocks_override = []{
        const char * e = getenv("GGML_TP_AR_BLOCKS");
        if (!e || !e[0]) return 0;
        int v = atoi(e);
        return (v > 0 && v <= kMaxBlocks) ? v : 0;
    }();
    const int blocks_cap = s_blocks_override ? s_blocks_override : kDefaultBlocks;
    const int64_t packed_size = n_elements / 4;
    int blocks = std::min(blocks_cap, std::max(1, (int)((packed_size + kThreads - 1) / kThreads)));

    // Per-N kernel launches. Dispatch on the selected wire type; the helper
    // recursively matches `nranks` against compile-time N from kMaxRanks down
    // to 2 and instantiates the broadcast/twoshot kernels for T_wire.
    if (s_wire_bf16) {
        ctx->dbg_n_bf16++;
        if (n_elements > ctx->dbg_max_ne_bf16) ctx->dbg_max_ne_bf16 = n_elements;
        tp_custom_ar_launch<nv_bfloat16>(ctx, nranks, s_twoshot, s_broadcast,
                                         input_ptrs, output_ptrs, streams, n_elements, blocks);
    } else {
        ctx->dbg_n_f32++;
        tp_custom_ar_launch<float>(ctx, nranks, s_twoshot, s_broadcast,
                                   input_ptrs, output_ptrs, streams, n_elements, blocks);
    }
}

} // namespace ggml_cuda_tp
