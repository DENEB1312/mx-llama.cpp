// Isolated test for the Q8_0 repack MMQ (prefill) kernel.
//
// Compile with:
//   clang++ -x hip --offload-arch=gfx906 -std=gnu++17 -O2 \
//     -DGGML_USE_HIP -DGCN \
//     -I$ROCM_ROOT/include \
//     -o test-repack-mmq test-repack-mmq.cu \
//     -L$ROCM_ROOT/lib -lamdhip64 -ldl

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>

// ─── Types ───
struct block_q8_0 {
    __half  d;
    int8_t  qs[32];
};
static_assert(sizeof(block_q8_0) == 34, "q8_0 size");

struct block_q8_1 {
    __half2 ds;
    int8_t  qs[32];
};
static_assert(sizeof(block_q8_1) == 36, "q8_1 size");

static __device__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

static __device__ float warp_reduce_sum(float x) {
    x += __shfl_xor(x, 16, 64);
    x += __shfl_xor(x,  8, 64);
    x += __shfl_xor(x,  4, 64);
    x += __shfl_xor(x,  2, 64);
    x += __shfl_xor(x,  1, 64);
    return x;
}

static inline int repack_nsp(int ne0) {
    int n_sub = ne0 / 32;
    return (n_sub & (n_sub - 1)) == 0 ? n_sub + 1 : n_sub;
}

static void repack_q8_0_host(const block_q8_0 * blocks, uint8_t * dst, int ne0, int ne1) {
    int n_blocks = ne0 / 32;
    int nsp = repack_nsp(ne0);
    size_t qs_len = (size_t)ne1 * nsp * 32;
    memset(dst, 0, qs_len + (size_t)ne1 * nsp * 2);
    for (int row = 0; row < ne1; row++)
        for (int blk = 0; blk < n_blocks; blk++) {
            const block_q8_0 * b = &blocks[row * n_blocks + blk];
            memcpy(dst + (size_t)(row * nsp + blk) * 32, b->qs, 32);
            memcpy(dst + qs_len + (size_t)(row * nsp + blk) * 2, &b->d, 2);
        }
}

// ─── MMQ kernel constants ───
#define MMQ_RP_BK 4
#define MMQ_RP_TM 4
#define MMQ_RP_TN 2  // reduced for simple testing
#define MMQ_RP_BM (16 * MMQ_RP_TM)  // 64
#define MMQ_RP_BN (16 * MMQ_RP_TN)  // 32

// ─── Simplified MMQ kernel ───
template <int TN_>
__global__ void mmq_gemm_q8_0_repacked(
        const uint8_t * __restrict__ wbase,
        const block_q8_1 * __restrict__ xq,
        float * __restrict__ y,
        int ne0, int ne1, int n_tok, int x_stride) {

    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const int row0 = blockIdx.x * MMQ_RP_BM;
    const int tok0 = blockIdx.y * (16 * TN_);

    const int n_sub = ne0 >> 5;
    const int nsp   = ((n_sub & (n_sub - 1)) == 0) ? (n_sub + 1) : n_sub;
    const int4    * qsp = reinterpret_cast<const int4 *>(wbase);
    const uint16_t * dp  = reinterpret_cast<const uint16_t *>(
        wbase + (size_t)ne1 * nsp * 32);

    __shared__ int4      sW_lo[MMQ_RP_BM][MMQ_RP_BK];
    __shared__ int4      sW_hi[MMQ_RP_BM][MMQ_RP_BK];
    __shared__ float     sWd  [MMQ_RP_BM][MMQ_RP_BK];
    __shared__ block_q8_1 sX[(16 * TN_)][MMQ_RP_BK + 1];

    float acc[MMQ_RP_TM][TN_] = {};

    const int lr = t >> 2;
    const int lk = t & 3;

    for (int sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_BK) {
        const int sb   = sb0 + lk;
        const int wrow = row0 + lr;
        if (wrow < ne1 && sb < n_sub) {
            sW_lo[lr][lk] = qsp[(size_t)(wrow * nsp + sb) * 2];
            sW_hi[lr][lk] = qsp[(size_t)(wrow * nsp + sb) * 2 + 1];
            const uint16_t d_bits = dp[(size_t) wrow * nsp + sb];
            sWd[lr][lk] = __half2float(*reinterpret_cast<const __half *>(&d_bits));
        } else {
            sWd[lr][lk] = 0.0f;
        }
        if (lr < (16 * TN_)) {
            int xcol = tok0 + lr;
            if (xcol < n_tok && sb < n_sub) {
                sX[lr][lk] = xq[(size_t)xcol * x_stride + sb];
            } else {
                sX[lr][lk].ds = __halves2half2(0.0f, 0.0f);
            }
            // zero-fill qs for out-of-bounds
            if (!(xcol < n_tok && sb < n_sub)) {
                memset(sX[lr][lk].qs, 0, 32);
            }
        }
        __syncthreads();

        for (int kk = 0; kk < MMQ_RP_BK; kk++) {
            int4 wq_lo[MMQ_RP_TM], wq_hi[MMQ_RP_TM];
            float dsc[MMQ_RP_TM];
            for (int r = 0; r < MMQ_RP_TM; r++) {
                wq_lo[r] = sW_lo[ty + r * 16][kk];
                wq_hi[r] = sW_hi[ty + r * 16][kk];
                dsc[r]   = sWd  [ty + r * 16][kk];
            }
            for (int n = 0; n < TN_; n++) {
                const block_q8_1 * xb = &sX[tx + n * 16][kk];
                const int * xq32 = reinterpret_cast<const int *>(xb->qs);
                const float dx = __low2float(xb->ds);
                for (int r = 0; r < MMQ_RP_TM; r++) {
                    const uint32_t lo[4] = { (uint32_t)wq_lo[r].x, (uint32_t)wq_lo[r].y, (uint32_t)wq_lo[r].z, (uint32_t)wq_lo[r].w };
                    const uint32_t hi[4] = { (uint32_t)wq_hi[r].x, (uint32_t)wq_hi[r].y, (uint32_t)wq_hi[r].z, (uint32_t)wq_hi[r].w };
                    int idot = 0;
                    for (int j = 0; j < 4; j++) {
                        idot = dp4a((int)lo[j], xq32[j],     idot);
                        idot = dp4a((int)hi[j], xq32[j + 4], idot);
                    }
                    acc[r][n] += dsc[r] * dx * (float)idot;
                }
            }
        }
        __syncthreads();
    }

    for (int r = 0; r < MMQ_RP_TM; r++) {
        const int row = row0 + ty + r * 16;
        if (row >= ne1) continue;
        for (int n = 0; n < TN_; n++) {
            const int tok = tok0 + tx + n * 16;
            if (tok < n_tok) {
                y[(size_t)tok * ne1 + row] = acc[r][n];
            }
        }
    }
}

// ─── CPU reference GEMM ───
static void cpu_gemm_q8_0(const block_q8_0 * weights, const block_q8_1 * act,
        float * result, int ne0, int ne1, int ne11, int x_stride) {
    int n_blocks = ne0 / 32;
    // result shape: [ne11, ne1] = ne11 rows of output, each with ne1 elements
    for (int col = 0; col < ne11; col++) {
        for (int row = 0; row < ne1; row++) {
            float sum = 0.0f;
            for (int blk = 0; blk < n_blocks; blk++) {
                const block_q8_0 * wb = &weights[row * n_blocks + blk];
                const block_q8_1 * xb = &act[col * x_stride + blk];
                float dw = __half2float(wb->d);
                float dx = __low2float(xb->ds);
                int dot = 0;
                for (int i = 0; i < 32; i++)
                    dot += (int)wb->qs[i] * (int)xb->qs[i];
                sum += dw * dx * (float)dot;
            }
            result[col * ne1 + row] = sum;
        }
    }
}

// ─── Q8_1 quantization ───
static void quantize_q8_1_host_batch(const float * x, block_q8_1 * y,
        int ne0, int ne11, int x_stride) {
    // ne0: K dimension (activations)
    // ne11: number of activation columns
    // x_stride: number of Q8_1 blocks per column (padded)
    // Input x: shape [ne11, ne0] or [ne11, x_stride*32] with padding
    int n_blocks = ne0 / 32;
    for (int col = 0; col < ne11; col++) {
        for (int blk = 0; blk < n_blocks; blk++) {
            float amax = 0.0f, sum_x = 0.0f;
            for (int i = 0; i < 32; i++) {
                float v = x[col * ne0 + blk * 32 + i];
                amax = fmaxf(amax, fabsf(v));
                sum_x += v;
            }
            if (amax == 0.0f) amax = 1.0f;
            float d = amax / 127.0f;
            __half d_h = (__half)d;
            __half s_h = (__half)(d * sum_x);
            for (int i = 0; i < 32; i++)
                y[col * x_stride + blk].qs[i] = (int8_t)roundf(x[col * ne0 + blk * 32 + i] / d);
            memcpy(&y[col * x_stride + blk].ds, &d_h, 2);
            memcpy((uint8_t*)&y[col * x_stride + blk].ds + 2, &s_h, 2);
        }
        // zero-fill padding blocks
        for (int blk = n_blocks; blk < x_stride; blk++) {
            memset(&y[col * x_stride + blk], 0, sizeof(block_q8_1));
        }
    }
}

// ─── Main ───
int main(int argc, char ** argv) {
    int ne0    = argc > 1 ? atoi(argv[1]) : 128;  // K dimension
    int ne1    = argc > 2 ? atoi(argv[2]) : 16;   // M (weight rows)
    int ne11   = argc > 3 ? atoi(argv[3]) : 2;    // N (activation columns / tokens)

    if (ne0 % 32 != 0) ne0 = ((ne0 + 31) / 32) * 32;
    if (ne1 < 1) ne1 = 1;
    if (ne11 < 1) ne11 = 1;

    int n_blocks = ne0 / 32;
    int nsp = repack_nsp(ne0);
    int x_stride = n_blocks;  // no padding needed for test

    printf("=== Q8_0 Repack MMQ Test ===\n");
    printf("K=%d  M=%d  N=%d  n_sub=%d  nsp=%d\n", ne0, ne1, ne11, n_blocks, nsp);
    printf("MMQ tile: BM=%d BN=%d BK=%d TM=%d TN=%d\n",
           MMQ_RP_BM, MMQ_RP_BN, MMQ_RP_BK, MMQ_RP_TM, MMQ_RP_TN);

    // ── Create weights ──
    block_q8_0 * weights = new block_q8_0[ne1 * n_blocks];
    srand(42);
    for (int i = 0; i < ne1 * n_blocks; i++) {
        weights[i].d = (__half)(0.5f + 0.1f * (rand() % 10));
        for (int j = 0; j < 32; j++)
            weights[i].qs[j] = (int8_t)((rand() % 255) - 127);
    }

    // ── Create activations ──
    float * act_f32 = new float[ne11 * ne0];
    for (int i = 0; i < ne11 * ne0; i++)
        act_f32[i] = 0.1f * (rand() % 31 - 15);
    block_q8_1 * act_q8_1 = new block_q8_1[ne11 * x_stride];
    quantize_q8_1_host_batch(act_f32, act_q8_1, ne0, ne11, x_stride);

    // ── CPU reference ──
    float * ref = new float[ne11 * ne1];
    cpu_gemm_q8_0(weights, act_q8_1, ref, ne0, ne1, ne11, x_stride);
    printf("\nCPU reference (first few):\n");
    for (int col = 0; col < ne11 && col < 2; col++) {
        printf("  col %d:", col);
        for (int row = 0; row < ne1 && row < 8; row++)
            printf(" %.4f", ref[col * ne1 + row]);
        printf("\n");
    }

    // ── Host repack ──
    size_t qs_len = (size_t)ne1 * nsp * 32;
    size_t repacked_bytes = qs_len + (size_t)ne1 * nsp * 2;
    uint8_t * repacked = new uint8_t[repacked_bytes];
    repack_q8_0_host(weights, repacked, ne0, ne1);

    // ── GPU ──
    hipSetDevice(0);
    uint8_t * w_gpu;    hipMalloc(&w_gpu, repacked_bytes);
    block_q8_1 * xq_gpu; hipMalloc(&xq_gpu, ne11 * x_stride * sizeof(block_q8_1));
    float * y_gpu;       hipMalloc(&y_gpu, ne11 * ne1 * sizeof(float));

    hipMemcpy(w_gpu, repacked, repacked_bytes, hipMemcpyHostToDevice);
    hipMemcpy(xq_gpu, act_q8_1, ne11 * x_stride * sizeof(block_q8_1), hipMemcpyHostToDevice);
    hipDeviceSynchronize();

    // ── Launch MMQ ──
    constexpr int TN = 2;
    dim3 grid((ne1 + MMQ_RP_BM - 1) / MMQ_RP_BM,
              (ne11 + MMQ_RP_BN - 1) / MMQ_RP_BN, 1);
    dim3 block(256, 1, 1);
    printf("\nGPU launch: grid(%d,%d) block(256)\n", grid.x, grid.y);

    typedef void (*KernelMMQ)(const uint8_t*, const block_q8_1*, float*, int, int, int, int);
    KernelMMQ k = mmq_gemm_q8_0_repacked<TN>;
    hipLaunchKernelGGL(k, grid, block, 0, 0, w_gpu, xq_gpu, y_gpu, ne0, ne1, ne11, x_stride);
    hipDeviceSynchronize();

    float * y_host = new float[ne11 * ne1];
    hipMemcpy(y_host, y_gpu, ne11 * ne1 * sizeof(float), hipMemcpyDeviceToHost);

    // ── Compare ──
    float max_err = 0.0f;
    int max_idx = -1;
    printf("\nGPU MMQ results:\n");
    for (int col = 0; col < ne11 && col < 2; col++) {
        printf("  col %d:", col);
        for (int row = 0; row < ne1 && row < 8; row++) {
            int idx = col * ne1 + row;
            float err = fabs(y_host[idx] - ref[idx]);
            printf(" %.4f", y_host[idx]);
            if (err > max_err) { max_err = err; max_idx = idx; }
        }
        printf("\n");
    }

    printf("\nMax error: %e (at idx %d: gpu=%f ref=%f)\n",
           max_err, max_idx, max_idx >= 0 ? y_host[max_idx] : 0.0f,
           max_idx >= 0 ? ref[max_idx] : 0.0f);

    // ── Check all ──
    int bad_count = 0;
    for (int i = 0; i < ne11 * ne1; i++) {
        float err = fabs(y_host[i] - ref[i]);
        if (err > 1e-3f) {
            bad_count++;
            if (bad_count <= 10) {
                int col = i / ne1, row = i % ne1;
                printf("MISMATCH col=%d row=%d: gpu=%f ref=%f err=%e\n",
                       col, row, y_host[i], ref[i], err);
            }
        }
    }
    printf("Bad elements: %d / %d\n", bad_count, ne11 * ne1);

    // ── Cleanup ──
    hipFree(w_gpu); hipFree(xq_gpu); hipFree(y_gpu);
    delete[] weights; delete[] act_f32; delete[] act_q8_1;
    delete[] repacked; delete[] ref; delete[] y_host;
    return 0;
}
