#include "repack-gcn.cuh"
#include "convert.cuh"
#include "quantize.cuh"
#include "mmq.cuh"

#include "ggml-backend-impl.h"
#include "mmid.cuh"
#include "unary.cuh"

#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// =====================================================================
// REPACK SYSTEM — gfx906 (GCN / Vega20) Q8_0 weight repacking
// =====================================================================

__device__ __forceinline__ block_q8_1 rp_xq_from_mmq(
        const block_q8_1_mmq * xq, const uint32_t col, const uint32_t sb,
        const uint32_t ne11, const bool has_sum);

// ---------------------------------------------------------------------
// layout helpers
// ---------------------------------------------------------------------

static __host__ __device__ inline int64_t repack_q4k_nsp(const int64_t ne0) {
    const int64_t n_sub = ne0 / 32;
    return (n_sub & (n_sub - 1)) == 0 ? n_sub + 1 : n_sub;
}

static inline size_t repack_gcn_nbytes(const ggml_type type, const int64_t ne0, const int64_t ne1) {
    const int64_t nsp      = repack_q4k_nsp(ne0);
    switch (type) {
        case GGML_TYPE_Q8_0: return (size_t) ne1 * nsp * 34;
        default:             GGML_ABORT("unsupported repack type");
    }
}

bool ggml_cuda_repack_tensor_supported(const ggml_tensor * t) {
    // 2D weights (MUL_MAT) or 3D per-expert stacks (MUL_MAT_ID)
    if ((ggml_n_dims(t) != 2 && ggml_n_dims(t) != 3) || !ggml_is_contiguous(t)) {
        return false;
    }
    switch (t->type) {
        case GGML_TYPE_Q8_0: {
            static const bool q8 = [] {
                const char * e = getenv("GGML_CUDA_REPACK_Q8_0");
                return e != nullptr && e[0] != '0';
            }();
            return q8 && t->ne[0] % 32 == 0;
        }
        default:             return false;
    }
}

// ---------------------------------------------------------------------
// host-side repack (one-shot at weight upload)
// ---------------------------------------------------------------------

static void repack_q8_0_host(const block_q8_0 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 32;
    const int64_t nsp      = repack_q4k_nsp(ne0);
    const size_t  qs_len   = (size_t) ne1 * nsp * 32;

    memset(dst, 0, qs_len + (size_t) ne1 * nsp * 2);

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q8_0 * b = &blocks[row * n_blocks + blk];
            memcpy(dst + (size_t)(row * nsp + blk) * 32, b->qs, 32);
            memcpy(dst + qs_len + (size_t)(row * nsp + blk) * 2, &b->d, 2);
        }
    }
}

// ---------------------------------------------------------------------
// kernels (GCN only — guarded so non-HIP / non-GCN builds still compile)
// ---------------------------------------------------------------------

// Largest e with expert_bounds[e] <= a.
static __device__ __forceinline__ uint32_t repack_find_expert(
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert, const uint32_t a) {
    uint32_t lo = 0, hi = n_expert;
    while (lo + 1 < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        if ((uint32_t) expert_bounds[mid] <= a) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

template <int BN>
static __global__ void repack_tile_off(
        const int32_t * __restrict__ expert_bounds, int32_t * __restrict__ tile_off,
        const int n_expert) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    int acc = 0;
    tile_off[0] = 0;
    for (int e = 0; e < n_expert; e++) {
        const int cnt = expert_bounds[e + 1] - expert_bounds[e];
        acc += (cnt + BN - 1) / BN;
        tile_off[e + 1] = acc;
    }
}

__device__ __forceinline__ block_q8_1 rp_xq_from_mmq(
        const block_q8_1_mmq * xq, const uint32_t col, const uint32_t sb,
        const uint32_t ne11, const bool has_sum) {
    const uint64_t idx = (uint64_t)(sb >> 2) * ne11 + col;
    const block_q8_1_mmq & m = xq[idx];
    block_q8_1 out;
    if (has_sum) {
        out.ds = m.ds4[sb & 3];
    } else {
        out.ds = make_half2(__float2half(m.d4[sb & 3]), __float2half(0.0f));
    }
 
    const uint4 * qsp = reinterpret_cast<const uint4 *>(m.qs + (sb & 3) * QK8_1);
    uint4       * oqp = reinterpret_cast<uint4       *>(out.qs);
    #pragma unroll
    for (int k = 0; k < QK8_1 / 16; k++) {
        oqp[k] = qsp[k];
    }
    return out;
}

template <int ROWS, int NWAVES, bool HAS_IDS>
static __global__ void mul_mat_vec_q8_0_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs_id, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
    if constexpr (HAS_IDS) {
    
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        wbase += e * expert_stride;
        xq    += (size_t) a * xs_id;
        y     += (size_t) a * dst_s1;
        GGML_UNUSED_VARS(ids_dst, expert_bounds, n_expert);
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    }
    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t nsp = ((n_blocks & (n_blocks - 1u)) == 0u) ? (n_blocks + 1u) : n_blocks;

    const int      * qs_int  = reinterpret_cast<const int *>(wbase);
    const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 32);

    const int wave = threadIdx.x >> 6;
    const int row0 = blockIdx.x * (ROWS * NWAVES) + wave * ROWS;
    const int lane = threadIdx.x & 63;

    float acc[ROWS] = {};

    const uint32_t n_half = n_blocks * 2;
    for (uint32_t hb = lane; hb < n_half; hb += 64) {
        const uint32_t sb   = hb >> 1;
        const uint32_t half = hb & 1;
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const int * xq32 = reinterpret_cast<const int *>(xb->qs) + half * 4;

#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int) ne1) {
                continue;
            }
            const int      * w_int = qs_int + ((size_t) row * nsp + sb) * 8 + half * 4;
            const uint16_t   db    = d_plane[(size_t) row * nsp + sb];
            const float      dw    = __half2float(*reinterpret_cast<const __half *>(&db));

            int idot = 0;
#pragma unroll
            for (int g = 0; g < 4; g++) {
                idot = ggml_cuda_dp4a(w_int[g], xq32[g], idot);
            }
            acc[r] += dw * dx * (float) idot;
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const float a = warp_reduce_sum<64>(acc[r]);
        if (lane == 0 && (row0 + r) < (int) ne1) {
            y[row0 + r] = a;
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds, n_expert, expert_stride, xs_id, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}



#define MMQ_RP_Q8_BK 4
#define MMQ_RP_Q8_TN 2
#define MMQ_RP_Q8_BM 64
#define MMQ_RP_Q8_BN (64 * MMQ_RP_Q8_TN)
#define MMQ_RP_Q8_NROW_LANES 4

static __device__ __forceinline__ int sX_swizzle(int lr) {
    const int n  = lr >> 6;        // 0 or 1 (upper half)
    int       tx = lr & 63;        // 0..63 within half
    tx ^= (tx >> 5) << 4;         // XOR 16 when tx >= 32
    return (n << 6) | tx;
}

template <bool HAS_IDS, int TN_>
    static __global__ void __launch_bounds__(512, 1) mmq_gemm_q8_0_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok, const uint32_t x_stride,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(GCN)
   
    const int t  = threadIdx.x + threadIdx.y * blockDim.x;
    const int tx = threadIdx.x;          // 0..63 column lane
    const int ty = threadIdx.y;          // 0..7  row lane
    const uint32_t row0 = blockIdx.x * MMQ_RP_Q8_BM;
    uint32_t tok0 = blockIdx.y * (64 * TN_);
    uint32_t a_base = 0, a_end = 0;
    if constexpr (HAS_IDS) {
        if (blockIdx.y >= (uint32_t) tile_off[n_expert]) {
            return;
        }
        const uint32_t e = repack_find_expert(tile_off, n_expert, blockIdx.y);
        const uint32_t local_tile = blockIdx.y - (uint32_t) tile_off[e];
        a_base = (uint32_t) expert_bounds[e] + local_tile * (16 * TN_);
        a_end  = (uint32_t) expert_bounds[e + 1];
        wbase += e * expert_stride;
        tok0   = 0;
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, tile_off, n_expert, expert_stride);
    }

    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint4    * qsp = reinterpret_cast<const uint4 *>(wbase);
    const uint16_t * dp  = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 32);

    __shared__ uint4      sW_lo[MMQ_RP_Q8_BM][MMQ_RP_Q8_BK];
    __shared__ uint4      sW_hi[MMQ_RP_Q8_BM][MMQ_RP_Q8_BK];
    __shared__ float      sWd  [MMQ_RP_Q8_BM][MMQ_RP_Q8_BK];
    __shared__ block_q8_1 sX   [MMQ_RP_Q8_BN][MMQ_RP_Q8_BK + 1];

    constexpr int NROW = MMQ_RP_Q8_BM / MMQ_RP_Q8_NROW_LANES;   // 16 rows/lane
    float acc[NROW][TN_] = {};

    constexpr int NTHREADS = 64 * MMQ_RP_Q8_NROW_LANES;
    const int w_elm = MMQ_RP_Q8_BM * MMQ_RP_Q8_BK;
    const int x_elm = MMQ_RP_Q8_BN * MMQ_RP_Q8_BK;

    for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_Q8_BK) {
        for (int e = t; e < w_elm; e += NTHREADS) {
            const int lr = e / MMQ_RP_Q8_BK;
            const int lk = e % MMQ_RP_Q8_BK;
            const uint32_t sb   = sb0 + lk;
            const uint32_t wrow = row0 + lr;
            if (wrow < ne1 && sb < n_sub) {
                sW_lo[lr][lk] = qsp[(size_t)(wrow * nsp + sb) * 2];
                sW_hi[lr][lk] = qsp[(size_t)(wrow * nsp + sb) * 2 + 1];
                const uint16_t d_bits = dp[(size_t) wrow * nsp + sb];
                sWd[lr][lk] = __half2float(*reinterpret_cast<const __half *>(&d_bits));
            } else {
                sWd[lr][lk] = 0.0f;
            }
        }
        for (int e = t; e < x_elm; e += NTHREADS) {
            const int lr = e / MMQ_RP_Q8_BK;
            const int lk = e % MMQ_RP_Q8_BK;
            const uint32_t sb = sb0 + lk;
            uint32_t xcol = tok0 + lr;
            bool     xval = xcol < n_tok;
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + lr;
                xval = a < a_end;
                xcol = xval ? (uint32_t) ids_src1[a] : 0;
            }
            const int sXr = sX_swizzle(lr);
            if (xval && sb < n_sub) {
                if constexpr (!HAS_IDS) {
                    // Fast path: read from the block_q8_1_mmq buffer produced by
                    // quantize_mmq_q8_1_cuda (Q8_0 -> D4 layout, scale only).
                    sX[sXr][lk] = rp_xq_from_mmq(reinterpret_cast<const block_q8_1_mmq *>(xq),
                                                  xcol, sb, n_tok, false);
                } else {
                    sX[sXr][lk] = xq[(size_t) xcol * x_stride + sb];
                }
            } else {
                sX[sXr][lk].ds = make_half2(0.0f, 0.0f);
            }
        }
        __syncthreads();

        for (int kk = 0; kk < MMQ_RP_Q8_BK; kk++) {
           
            float   dx[TN_];
            int xq32_cached[TN_][8];
            for (int n = 0; n < TN_; n++) {
                const block_q8_1 * xb = &sX[sX_swizzle(tx + n * 64)][kk];
                dx[n] = __low2float(xb->ds);
        
                uint4 q0, q1;
                ggml_cuda_memcpy_1<16, 0>(&q0, (const void *) xb->qs);
                ggml_cuda_memcpy_1<16, 0>(&q1, (const void *) ((const char *) xb->qs + 16));
                xq32_cached[n][0] = q0.x;
                xq32_cached[n][1] = q0.y;
                xq32_cached[n][2] = q0.z;
                xq32_cached[n][3] = q0.w;
                xq32_cached[n][4] = q1.x;
                xq32_cached[n][5] = q1.y;
                xq32_cached[n][6] = q1.z;
                xq32_cached[n][7] = q1.w;
            }
            for (int r = 0; r < NROW; r++) {
                const int row = ty + r * MMQ_RP_Q8_NROW_LANES;
            
                uint4 wlo, whi;
                ggml_cuda_memcpy_1<16, 0>(&wlo, &sW_lo[row][kk]);
                ggml_cuda_memcpy_1<16, 0>(&whi, &sW_hi[row][kk]);
                const float d = sWd[row][kk];
                const uint32_t lo[4] = { wlo.x, wlo.y, wlo.z, wlo.w };
                const uint32_t hi[4] = { whi.x, whi.y, whi.z, whi.w };
                for (int n = 0; n < TN_; n++) {
                    const int * xq32 = xq32_cached[n];
                    int idot = 0;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        idot = ggml_cuda_dp4a((int) lo[j], xq32[j],     idot);
                        idot = ggml_cuda_dp4a((int) hi[j], xq32[j + 4], idot);
                    }
                    acc[r][n] += d * dx[n] * (float) idot;
                }
            }
        }
        __syncthreads();
    }

    for (int r = 0; r < NROW; r++) {
        const uint32_t row = row0 + ty + r * MMQ_RP_Q8_NROW_LANES;
        if (row >= ne1) {
            continue;
        }
        for (int n = 0; n < TN_; n++) {
            const uint32_t col = tx + n * 64;
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + col;
                if (a < a_end && col < (uint32_t)(16 * TN_)) {
                    y[(size_t) ids_dst[a] * dst_s1 + row] = acc[r][n];
                }
            } else {
                const uint32_t tok = tok0 + col;
                if (tok < n_tok) {
                    y[(size_t) tok * dst_s1 + row] = acc[r][n];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, x_stride, ids_src1, ids_dst, expert_bounds, tile_off, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

// ---------------------------------------------------------------------
// MUL_MAT dispatch
// ---------------------------------------------------------------------

static void ggml_cuda_mul_mat_repacked_slice(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const uint8_t * w, const block_q8_1 * xq,
        float * dst_d, int64_t ne00, int64_t ne01, int64_t ne11,
        int64_t x_stride, cudaStream_t stream);

void ggml_cuda_mul_mat_repacked(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->nb[0] == sizeof(float)); // rows may be strided; dim0 must be dense
    GGML_ASSERT(dst->nb[1]  == (size_t) dst->ne[0] * sizeof(float));

    const int64_t ne00 = src0->ne[0]; // K
    const int64_t ne01 = src0->ne[1]; // M
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1]; // N
    const int64_t ne12 = src1->ne[2]; // broadcast slices (2D weight repeats)
    const int64_t ne13 = src1->ne[3];
    GGML_ASSERT(ne10 == ne00);

    cudaStream_t stream = ctx.stream();
    const uint8_t * w = (const uint8_t *) src0->data;

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1;
    const uint64_t n_groups   = (uint64_t) ne10_padded / (4 * QK8_1); // mmq blocks (128 vals) per token col

    if (ne11 == 1) {
      
        ggml_cuda_pool_alloc<block_q8_1> src1_q8_1(ctx.pool(),
            ne13 * ne12 * ne11 * x_stride);
        {
            const int64_t s11 = src1->nb[1] / sizeof(float);
            const int64_t s12 = src1->nb[2] / sizeof(float);
            const int64_t s13 = src1->nb[3] / sizeof(float);
            quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(),
                src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        }
        for (int64_t i3 = 0; i3 < ne13; i3++) {
        for (int64_t i2 = 0; i2 < ne12; i2++) {
            const block_q8_1 * xq = src1_q8_1.get()
                                  + (i3 * ne12 + i2) * ne11 * x_stride;
            float * dst_d = (float *)((char *) dst->data + i3 * dst->nb[3] + i2 * dst->nb[2]);
            ggml_cuda_mul_mat_repacked_slice(ctx, src0, w, xq, dst_d,
                ne00, ne01, ne11, x_stride, stream);
        }
        }
    } else {
       
        ggml_cuda_pool_alloc<block_q8_1_mmq> src1_q8_1(ctx.pool(),
            ne13 * ne12 * ne11 * n_groups);
        {
            const int64_t s11 = src1->nb[1] / sizeof(float);
            const int64_t s12 = src1->nb[2] / sizeof(float);
            const int64_t s13 = src1->nb[3] / sizeof(float);
            quantize_mmq_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(),
                src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        }
        for (int64_t i3 = 0; i3 < ne13; i3++) {
        for (int64_t i2 = 0; i2 < ne12; i2++) {
            const block_q8_1_mmq * slice_base = src1_q8_1.get()
                                  + (i3 * ne12 + i2) * ne11 * n_groups;
            const block_q8_1 * xq = reinterpret_cast<const block_q8_1 *>(slice_base);
            float * dst_d = (float *)((char *) dst->data + i3 * dst->nb[3] + i2 * dst->nb[2]);
            ggml_cuda_mul_mat_repacked_slice(ctx, src0, w, xq, dst_d,
                ne00, ne01, ne11, x_stride, stream);
        }
        }
    }
}

static void ggml_cuda_mul_mat_repacked_slice(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const uint8_t * w, const block_q8_1 * xq,
        float * dst_d, const int64_t ne00, const int64_t ne01, const int64_t ne11,
        const int64_t x_stride, cudaStream_t stream) {
    if (ne11 == 1) {
        switch (src0->type) {
            case GGML_TYPE_Q8_0: {
             
                if (ne01 >= 4096) {
                    const dim3 grid(ne01, 1, 1);
                    mul_mat_vec_q8_0_repacked<1, 1, false><<<grid, 64, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                        nullptr, nullptr, nullptr, 0, 0, 0, 0);
                } else {
                
                    const dim3 grid((ne01 + 7) / 8, 1, 1);
                    mul_mat_vec_q8_0_repacked<2, 4, false><<<grid, 256, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                        nullptr, nullptr, nullptr, 0, 0, 0, 0);
                }
            } break;
            default: GGML_ABORT("unsupported repack type");
        }
        return;
    }

  
    const int mmq_bm = MMQ_RP_Q8_BM;
    const int mmq_bn = MMQ_RP_Q8_BN;
    const dim3 grid((ne01 + mmq_bm - 1) / mmq_bm,
                    (ne11 + mmq_bn - 1) / mmq_bn, 1);
    switch (src0->type) {
        case GGML_TYPE_Q8_0:
            mmq_gemm_q8_0_repacked<false, MMQ_RP_Q8_TN><<<grid, dim3(64, 4), 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, (uint32_t) x_stride,
                nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
            break;
        default: GGML_ABORT("unsupported repack type");
    }
    GGML_UNUSED(ctx);
}

void ggml_cuda_mul_mat_id_repacked(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ids->type  == GGML_TYPE_I32);
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(ids->nb[0]  == sizeof(int32_t));
    GGML_ASSERT(src1->ne[3] == 1 && dst->ne[3] == 1);
    GGML_ASSERT(src1->nb[2] == src1->nb[1] * src1->ne[1]);
    GGML_ASSERT(dst->nb[2]  == dst->nb[1]  * dst->ne[1]);
    GGML_ASSERT(dst->nb[1]  == (size_t) dst->ne[0] * sizeof(float));

    const int64_t ne00 = src0->ne[0]; // K
    const int64_t ne01 = src0->ne[1]; // rows per expert
    const int64_t ne02 = src0->ne[2]; // experts
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 == ne00);
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = ids->ne[1];
    const int64_t n_assign      = n_expert_used * n_tokens;

    cudaStream_t stream = ctx.stream();
    const uint8_t * w = (const uint8_t *) src0->data;
    float * dst_d = (float *) dst->data;
    const size_t expert_stride = repack_gcn_nbytes(src0->type, ne00, ne01);
    const uint32_t dst_s1 = dst->nb[1] / sizeof(float);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_assign);
    ggml_cuda_pool_alloc<int32_t> ids_dst (ctx.pool(), n_assign);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);
    if (n_tokens > 1) {
        const int si1  = ids->nb[1] / sizeof(int32_t);
        const int sis1 = src1->nb[2] / src1->nb[1];
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data,
            ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, n_tokens, n_expert_used, src1->ne[1], si1, sis1, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(),
        src1->ne[2] * src1->ne[1] * ne10_padded * sizeof(block_q8_1) / QK8_1);
    {
        const int64_t s11 = src1->nb[1] / sizeof(float);
        const int64_t s12 = src1->nb[2] / sizeof(float);
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(),
            src0->type, ne10, s11, s12, s12 * src1->ne[2], ne10_padded,
            src1->ne[1], src1->ne[2], 1, stream);
    }
    const block_q8_1 * xq = (const block_q8_1 *) src1_q8_1.get();

    if (n_tokens == 1) {
        const uint32_t xs_eff = src1->ne[1] == 1 ? 0u : (uint32_t) x_stride;
        switch (src0->type) {
            case GGML_TYPE_Q8_0: {
                if (ne01 >= 4096) {
                    const dim3 grid(ne01, n_assign, 1);
                    mul_mat_vec_q8_0_repacked<1, 1, true><<<grid, 64, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                        (const int32_t *) ids->data, nullptr, nullptr,
                        (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
                } else {
                    const dim3 grid((ne01 + 7) / 8, n_assign, 1);
                    mul_mat_vec_q8_0_repacked<2, 4, true><<<grid, 256, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                        (const int32_t *) ids->data, nullptr, nullptr,
                        (uint32_t) ne02, expert_stride, xs_eff, dst_s1);
                }
            } break;
            default: GGML_ABORT("unsupported repack type");
        }
        return;
    }

    constexpr int TN_ID = 1;
    ggml_cuda_pool_alloc<int32_t> tile_off(ctx.pool(), ne02 + 1);
    repack_tile_off<16 * TN_ID><<<1, 1, 0, stream>>>(expert_bounds.get(), tile_off.get(), ne02);
    const int64_t max_tiles = n_assign / (16 * TN_ID) + ne02;
    const int id_bm = MMQ_RP_Q8_BM;
    const dim3 grid((ne01 + id_bm - 1) / id_bm, max_tiles, 1);

    switch (src0->type) {
        case GGML_TYPE_Q8_0:
            mmq_gemm_q8_0_repacked<true, TN_ID><<<grid, dim3(64, 8), 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, 0, (uint32_t) x_stride,
                ids_src1.get(), ids_dst.get(), expert_bounds.get(), tile_off.get(),
                (uint32_t) ne02, expert_stride, dst_s1);
            break;
        default: GGML_ABORT("unsupported repack type");
    }
}

// ---------------------------------------------------------------------
// buffer type
// ---------------------------------------------------------------------

struct ggml_backend_cuda_repack_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_repack_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_repack_buffer_type_context * ctx =
        (ggml_backend_cuda_repack_buffer_type_context *) buft->context;
    return ctx->name.c_str();
}

bool ggml_backend_buft_is_cuda_repack(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_repack_buffer_type_get_name;
}

static void ggml_backend_cuda_repack_buffer_set_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor,
        const void * data, size_t offset, size_t size) {
    {
        static long long g_n = 0;
        if (g_n < 50) {
            FILE * f = fopen("/tmp/repack_dbg.log", "a");
            if (f) {
                fprintf(f, "REPACK_ENTER[%lld] dev=%d name=%s type=%d off=%zu size=%zu nbytes=%zu ne=[%lld,%lld,%lld,%lld]\n",
                    g_n, ((ggml_backend_cuda_repack_buffer_type_context *)buffer->buft->context)->device,
                    tensor->name ? tensor->name : "(null)", (int)tensor->type,
                    offset, size, ggml_nbytes(tensor),
                    (long long)tensor->ne[0], (long long)tensor->ne[1],
                    (long long)tensor->ne[2], (long long)tensor->ne[3]);
                fclose(f);
            }
        }
        g_n++;
    }

    const size_t t_nbytes = ggml_nbytes(tensor);

    if (offset != 0 || size != t_nbytes) {
        static std::map<void*, std::vector<uint8_t>> staging;
        void * key = tensor->data;
        auto & staged = staging[key];

        if (offset == 0 && staged.empty()) {
            staged.resize(t_nbytes, 0);
        }

        GGML_ASSERT(offset + size <= t_nbytes);
        memcpy(staged.data() + offset, data, size);

        if (offset + size < t_nbytes) {
            return; // more segments to come
        }

        // All segments collected; repack the full shard.
        GGML_ASSERT(ggml_cuda_repack_tensor_supported(tensor));
        const int64_t ne0 = tensor->ne[0];
        const int64_t ne1 = tensor->ne[1];
        const int64_t ne2 = tensor->ne[2];

        const size_t src_stride = t_nbytes / ne2;
        const size_t dst_stride = repack_gcn_nbytes(tensor->type, ne0, ne1);
        std::vector<uint8_t> repacked(dst_stride * ne2);
        for (int64_t e = 0; e < ne2; e++) {
            const uint8_t * src_e = staged.data() + e * src_stride;
            uint8_t       * dst_e = repacked.data() + e * dst_stride;
            switch (tensor->type) {
                case GGML_TYPE_Q8_0: repack_q8_0_host((const block_q8_0 *) src_e, dst_e, ne0, ne1); break;
                default:             GGML_ABORT("unsupported repack type");
            }
        }

        ggml_backend_cuda_repack_buffer_type_context * ctx =
            (ggml_backend_cuda_repack_buffer_type_context *) buffer->buft->context;
        ggml_cuda_set_device(ctx->device);
        CUDA_CHECK(cudaMemcpyAsync(tensor->data, repacked.data(), repacked.size(),
            cudaMemcpyHostToDevice, cudaStreamPerThread));
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));

        staging.erase(key);
        return;
    }

    // Single-segment (full-shard) upload: repack directly.
    GGML_ASSERT(ggml_cuda_repack_tensor_supported(tensor));

    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t ne2 = tensor->ne[2]; // experts (1 for plain 2D weights)

    const size_t src_stride = t_nbytes / ne2;
    const size_t dst_stride = repack_gcn_nbytes(tensor->type, ne0, ne1);
    std::vector<uint8_t> staged(dst_stride * ne2);
    for (int64_t e = 0; e < ne2; e++) {
        const uint8_t * src_e = (const uint8_t *) data + e * src_stride;
        uint8_t       * dst_e = staged.data() + e * dst_stride;
        switch (tensor->type) {
            case GGML_TYPE_Q8_0: repack_q8_0_host((const block_q8_0 *) src_e, dst_e, ne0, ne1); break;
            default:             GGML_ABORT("unsupported repack type");
        }
    }

    ggml_backend_cuda_repack_buffer_type_context * ctx =
        (ggml_backend_cuda_repack_buffer_type_context *) buffer->buft->context;
    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(tensor->data, staged.data(), staged.size(),
        cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_repack_buffer_get_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
        void * data, size_t offset, size_t size) {
    GGML_ABORT("repacked tensors cannot be read back (GGML_CUDA_REPACK)");
    GGML_UNUSED_VARS(buffer, tensor, data, offset, size);
}

static ggml_backend_buffer_t ggml_backend_cuda_repack_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_repack_buffer_type_context * ctx =
        (ggml_backend_cuda_repack_buffer_type_context *) buft->context;

    ggml_backend_buffer_t buffer =
        ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(ctx->device), size);
    if (buffer == nullptr) {
        return nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.set_tensor  = ggml_backend_cuda_repack_buffer_set_tensor;
    buffer->iface.get_tensor  = ggml_backend_cuda_repack_buffer_get_tensor;
    buffer->iface.cpy_tensor  = nullptr;
    return buffer;
}

static size_t ggml_backend_cuda_repack_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_repack_buffer_type_get_alloc_size(
        ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    if (ggml_cuda_repack_tensor_supported(tensor)) {
        return repack_gcn_nbytes(tensor->type, tensor->ne[0], tensor->ne[1]) * tensor->ne[2];
    }
    return ggml_nbytes(tensor);
    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_repack_buffer_type_interface = {
    /* .get_name       = */ ggml_backend_cuda_repack_buffer_type_get_name,
    /* .alloc_buffer   = */ ggml_backend_cuda_repack_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_cuda_repack_buffer_type_get_alignment,
    /* .get_max_size   = */ nullptr,
    /* .get_alloc_size = */ ggml_backend_cuda_repack_buffer_type_get_alloc_size,
    /* .is_host        = */ nullptr,
};

ggml_backend_buffer_type_t ggml_backend_cuda_repack_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    const char * env = getenv("GGML_CUDA_REPACK");
    if (env != nullptr && env[0] == '0') {
        return nullptr;
    }
    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }
    if (!GGML_CUDA_CC_IS_GCN(ggml_cuda_info().devices[device].cc)) {
        return nullptr;
    }

    static ggml_backend_buffer_type buft_storage[GGML_CUDA_MAX_DEVICES];
    static bool initialized[GGML_CUDA_MAX_DEVICES] = {};

    if (!initialized[device]) {
        buft_storage[device] = {
            /* .iface   = */ ggml_backend_cuda_repack_buffer_type_interface,
            /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
            /* .context = */ new ggml_backend_cuda_repack_buffer_type_context{
                                 device, GGML_CUDA_NAME + std::to_string(device) + "_Repacked"},
        };
        initialized[device] = true;
    }
    return &buft_storage[device];
}
