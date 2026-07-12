// Isolated test for the Q8_0 repack matvec kernel.
//
// Compile with HIP:
//   clang++ -x hip --offload-arch=gfx906 -std=gnu++17 -O2 \
//     -DGGML_USE_HIP -DGCN \
//     -I$ROCM_ROOT/include \
//     -o test-repack-q8_0 test-repack-q8_0.cu \
//     -L$ROCM_ROOT/lib -lamdhip64 -ldl
//
// Run: ./test-repack-q8_0 [ne0=128] [ne1=4]

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>

// ─── Type definitions ───

struct block_q8_0 {
    __half  d;
    int8_t  qs[32]; // 32 signed 8-bit quants
};
static_assert(sizeof(block_q8_0) == 34, "q8_0 size");

struct block_q8_1 {
    __half2 ds;      // x = d, y = d * sum(qs)
    int8_t  qs[32];  // 32 signed 8-bit quants
};
static_assert(sizeof(block_q8_1) == 36, "q8_1 size");

// ─── GCN dp4a ───

static __device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

// ─── Warp reduction (__shfl_xor-based, GCN-safe) ───

template <int width>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
    if constexpr (width >= 64) { x += __shfl_xor(x, 32, 64); }
    if constexpr (width >= 32) { x += __shfl_xor(x, 16, 64); }
    if constexpr (width >= 16) { x += __shfl_xor(x,  8, 64); }
    if constexpr (width >=  8) { x += __shfl_xor(x,  4, 64); }
    if constexpr (width >=  4) { x += __shfl_xor(x,  2, 64); }
    if constexpr (width >=  2) { x += __shfl_xor(x,  1, 64); }
    return x;
}

// ─── nsp helper ───

static inline int repack_nsp(int ne0) {
    int n_sub = ne0 / 32;
    return (n_sub & (n_sub - 1)) == 0 ? n_sub + 1 : n_sub;
}

// ─── Host-side repack ───

static void repack_q8_0_host(const block_q8_0 * blocks, uint8_t * dst, int ne0, int ne1) {
    int n_blocks = ne0 / 32;
    int nsp       = repack_nsp(ne0);
    size_t qs_len = (size_t)ne1 * nsp * 32;

    memset(dst, 0, qs_len + (size_t)ne1 * nsp * 2);

    for (int row = 0; row < ne1; row++) {
        for (int blk = 0; blk < n_blocks; blk++) {
            const block_q8_0 * b = &blocks[row * n_blocks + blk];
            // qs plane: qs bytes at offset (row*nsp+blk)*32
            memcpy(dst + (size_t)(row * nsp + blk) * 32, b->qs, 32);
            // d plane: 2-byte d at offset qs_len + (row*nsp+blk)*2
            memcpy(dst + qs_len + (size_t)(row * nsp + blk) * 2, &b->d, 2);
        }
    }
}

// ─── GPU kernel: exact copy of mul_mat_vec_q8_0_repacked (no ID path) ───

template <int ROWS, int NWAVES>
__global__ void mul_mat_vec_q8_0_repacked(
        const uint8_t * __restrict__ wbase,
        const block_q8_1 * __restrict__ xq,
        float * __restrict__ y,
        int ne0, int ne1) {

    const int n_blocks = ne0 >> 5;
    const int nsp = ((n_blocks & (n_blocks - 1)) == 0) ? (n_blocks + 1) : n_blocks;

    const int      * qs_int  = reinterpret_cast<const int *>(wbase);
    const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(
        wbase + (size_t)ne1 * nsp * 32);

    const int wave = threadIdx.x >> 6;
    const int row0 = blockIdx.x * (ROWS * NWAVES) + wave * ROWS;
    const int lane = threadIdx.x & 63;

    float acc[ROWS] = {};

    const int n_half = n_blocks * 2;
    for (int hb = lane; hb < n_half; hb += 64) {
        const int sb   = hb >> 1;    // sub-block index
        const int half = hb & 1;     // 0=lower 16B, 1=upper 16B

        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);           // activation scale
        const int * xq32 = reinterpret_cast<const int *>(xb->qs) + half * 4;

        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= ne1) continue;

            const int      * w_int = qs_int + ((size_t)row * nsp + sb) * 8 + half * 4;
            const uint16_t   db    = d_plane[(size_t)row * nsp + sb];
            const float      dw    = __half2float(*reinterpret_cast<const __half *>(&db));

            int idot = 0;
#pragma unroll
            for (int g = 0; g < 4; g++) {
                idot = dp4a(w_int[g], xq32[g], idot);
            }
            acc[r] += dw * dx * (float)idot;
        }
    }

    for (int r = 0; r < ROWS; r++) {
        const float a = warp_reduce_sum<64>(acc[r]);
        if (lane == 0 && (row0 + r) < ne1) {
            y[row0 + r] = a;
        }
    }
}

// ─── CPU reference matvec ───

static void cpu_matvec_q8_0(
        const uint8_t * repacked,
        const block_q8_1 * act,
        float * result,
        int ne0, int ne1) {
    const int n_blocks = ne0 / 32;
    const int nsp      = repack_nsp(ne0);
    const size_t qs_len = (size_t)ne1 * nsp * 32;

    const int      * qs_int  = reinterpret_cast<const int *>(repacked);
    const uint16_t * d_plane = reinterpret_cast<const uint16_t *>(repacked + qs_len);

    for (int row = 0; row < ne1; row++) {
        float sum = 0.0f;
        for (int blk = 0; blk < n_blocks; blk++) {
            const int    * w_int = qs_int + ((size_t)row * nsp + blk) * 8;
            const uint16_t db    = d_plane[(size_t)row * nsp + blk];
            const float    dw    = __half2float(*reinterpret_cast<const __half *>(&db));

            const block_q8_1 * xb = act + blk;
            const float dx  = __low2float(xb->ds);
            const int * xq32 = reinterpret_cast<const int *>(xb->qs);

            // Lower half (16 weights)
            int idot_lo = 0;
            for (int g = 0; g < 4; g++) {
                idot_lo = (int)idot_lo + // CPU dp4a emulation
                    (int)((int8_t)(w_int[g] & 0xFF))       * (int)((int8_t)(xq32[g] & 0xFF)) +
                    (int)((int8_t)((w_int[g] >> 8) & 0xFF))  * (int)((int8_t)((xq32[g] >> 8) & 0xFF)) +
                    (int)((int8_t)((w_int[g] >> 16) & 0xFF)) * (int)((int8_t)((xq32[g] >> 16) & 0xFF)) +
                    (int)((int8_t)((w_int[g] >> 24) & 0xFF)) * (int)((int8_t)((xq32[g] >> 24) & 0xFF));
            }

            // Upper half (16 weights)
            int idot_hi = 0;
            const int * w_int_hi = w_int + 4;
            const int * xq32_hi  = xq32 + 4;
            for (int g = 0; g < 4; g++) {
                idot_hi = (int)idot_hi +
                    (int)((int8_t)(w_int_hi[g] & 0xFF))       * (int)((int8_t)(xq32_hi[g] & 0xFF)) +
                    (int)((int8_t)((w_int_hi[g] >> 8) & 0xFF))  * (int)((int8_t)((xq32_hi[g] >> 8) & 0xFF)) +
                    (int)((int8_t)((w_int_hi[g] >> 16) & 0xFF)) * (int)((int8_t)((xq32_hi[g] >> 16) & 0xFF)) +
                    (int)((int8_t)((w_int_hi[g] >> 24) & 0xFF)) * (int)((int8_t)((xq32_hi[g] >> 24) & 0xFF));
            }

            int idot = idot_lo + idot_hi;
            sum += dw * dx * (float)idot;
        }
        result[row] = sum;
    }
}

// ─── Alternative: CPU reference from unrepacked weights ───

static void cpu_matvec_q8_0_direct(
        const block_q8_0 * weights,
        const block_q8_1 * act,
        float * result,
        int ne0, int ne1) {
    const int n_blocks = ne0 / 32;
    for (int row = 0; row < ne1; row++) {
        float sum = 0.0f;
        for (int blk = 0; blk < n_blocks; blk++) {
            const block_q8_0 * wb = &weights[row * n_blocks + blk];
            const block_q8_1 * xb = &act[blk];
            float dw = __half2float(wb->d);
            float dx = __low2float(xb->ds);
            int dot = 0;
            for (int i = 0; i < 32; i++) {
                dot += (int)wb->qs[i] * (int)xb->qs[i];
            }
            sum += dw * dx * (float)dot;
        }
        result[row] = sum;
    }
}

// ─── Host-side Q8_1 quantization ───

static void quantize_q8_1_host(const float * x, block_q8_1 * y, int ne0) {
    const int n_blocks = ne0 / 32;
    for (int blk = 0; blk < n_blocks; blk++) {
        float amax = 0.0f;
        float sum_x = 0.0f;
        for (int i = 0; i < 32; i++) {
            float v = x[blk * 32 + i];
            amax = fmaxf(amax, fabsf(v));
            sum_x += v;
        }
        if (amax == 0.0f) amax = 1.0f;
        float d = amax / 127.0f;

        __half d_h = (__half)(d);
        __half s_h = (__half)(d * sum_x);

        for (int i = 0; i < 32; i++) {
            float v = x[blk * 32 + i];
            y[blk].qs[i] = (int8_t)roundf(v / d);
        }

        __half2 ds;
        memcpy(&ds, &d_h, 2);
        memcpy(((uint8_t*)&ds) + 2, &s_h, 2);
        y[blk].ds = ds;
    }
}

// ─── Debug helpers ───

static void dump_qs(const char * label, const int8_t * qs, int n) {
    printf("%s: ", label);
    for (int i = 0; i < n && i < 16; i++) printf("%4d ", qs[i]);
    if (n > 16) printf("...");
    printf("\n");
}

static void dump_repack_planes(const uint8_t * repacked, int ne0, int ne1) {
    int n_blocks = ne0 / 32;
    int nsp = repack_nsp(ne0);
    size_t qs_len = (size_t)ne1 * nsp * 32;

    printf("--- Repack layout (ne0=%d ne1=%d n_blocks=%d nsp=%d) ---\n",
           ne0, ne1, n_blocks, nsp);
    printf("qs_plane: %zu bytes, d_plane: %zu bytes\n", qs_len, (size_t)ne1 * nsp * 2);

    for (int row = 0; row < ne1 && row < 3; row++) {
        printf("Row %d qs:\n", row);
        for (int blk = 0; blk < n_blocks && blk < 4; blk++) {
            const int8_t * qs = (const int8_t *)(repacked + (size_t)(row * nsp + blk) * 32);
            printf("  sb=%d: [", blk);
            for (int i = 0; i < 8; i++) printf("%4d", qs[i]);
            printf(" ... ");
            for (int i = 24; i < 32; i++) printf("%4d", qs[i]);
            printf("]\n");
        }
        printf("Row %d d:\n", row);
        for (int blk = 0; blk < n_blocks && blk < 4; blk++) {
            const uint16_t * dp = (const uint16_t *)(repacked + qs_len + (size_t)(row * nsp + blk) * 2);
            uint16_t db = *dp;
            float dv = __half2float(*reinterpret_cast<const __half *>(&db));
            printf("  sb=%d: d=0x%04x (%.6f)\n", blk, db, dv);
        }
    }
}

// ─── Main test ───

int main(int argc, char ** argv) {
    int ne0 = argc > 1 ? atoi(argv[1]) : 128;
    int ne1 = argc > 2 ? atoi(argv[2]) : 4;

    if (ne0 % 32 != 0) {
        printf("ne0 must be multiple of 32, adjusting to %d\n", ne0 = ((ne0 + 31) / 32) * 32);
    }
    if (ne1 < 1) ne1 = 1;

    printf("=== Q8_0 Repack Matvec Test ===\n");
    printf("ne0=%d (K dimension)  ne1=%d (rows/out_dim)\n", ne0, ne1);
    printf("n_blocks=%d  nsp=%d\n", ne0 / 32, repack_nsp(ne0));

    // ── Create synthetic weight data (Q8_0 format) ──
    int n_blocks = ne0 / 32;
    int total_blocks = ne1 * n_blocks;
    block_q8_0 * weights = new block_q8_0[total_blocks];

    srand(42);
    for (int i = 0; i < total_blocks; i++) {
        weights[i].d = (__half)(0.5f + 0.1f * (rand() % 10));
        for (int j = 0; j < 32; j++) {
            weights[i].qs[j] = (int8_t)((rand() % 255) - 127);
        }
    }

    printf("\n--- Sample weights (row 0, first 2 blocks) ---\n");
    for (int blk = 0; blk < 2 && blk < n_blocks; blk++) {
        float d = __half2float(weights[blk].d);
        printf("blk=%d d=%.6f  qs(first 8)=", blk, d);
        for (int i = 0; i < 8; i++) printf("%4d", weights[blk].qs[i]);
        printf("\n");
    }

    // ── Create synthetic activation data (float) and quantize to Q8_1 ──
    float * act_f32 = new float[ne0];
    for (int i = 0; i < ne0; i++) {
        act_f32[i] = 0.1f * (float)(rand() % 31 - 15);
    }
    block_q8_1 * act_q8_1 = new block_q8_1[n_blocks];
    quantize_q8_1_host(act_f32, act_q8_1, ne0);

    printf("\n--- Activation Q8_1 (first 2 blocks) ---\n");
    for (int blk = 0; blk < 2 && blk < n_blocks; blk++) {
        float dx = __low2float(act_q8_1[blk].ds);
        float sx = __high2float(act_q8_1[blk].ds);
        printf("blk=%d d=%.6f s=%.6f  qs(first 8)=", blk, dx, sx);
        for (int i = 0; i < 8; i++) printf("%4d", act_q8_1[blk].qs[i]);
        printf("\n");
    }

    // ── CPU reference: direct from unrepacked weights ──
    float * ref = new float[ne1];
    cpu_matvec_q8_0_direct(weights, act_q8_1, ref, ne0, ne1);
    printf("\n--- CPU reference (direct) ---\n");
    for (int r = 0; r < ne1 && r < 8; r++) printf("  row %d: %f\n", r, ref[r]);

    // ── Host-side repack ──
    int nsp = repack_nsp(ne0);
    size_t qs_len = (size_t)ne1 * nsp * 32;
    size_t repacked_bytes = qs_len + (size_t)ne1 * nsp * 2;
    uint8_t * repacked = new uint8_t[repacked_bytes];
    repack_q8_0_host(weights, repacked, ne0, ne1);
    dump_repack_planes(repacked, ne0, ne1);

    // ── CPU reference: from repacked data (should match direct) ──
    float * ref_repacked = new float[ne1];
    cpu_matvec_q8_0(repacked, act_q8_1, ref_repacked, ne0, ne1);
    printf("\n--- CPU reference (from repacked) ---\n");
    for (int r = 0; r < ne1 && r < 8; r++) {
        printf("  row %d: %f  (diff from direct: %e)\n",
               r, ref_repacked[r], fabs(ref_repacked[r] - ref[r]));
    }

    // ── GPU: upload, run kernel, download ──
    float * y_gpu, * y_host = new float[ne1];

    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    printf("\nGPU: %d device(s) found\n", deviceCount);
    hipSetDevice(0);

    uint8_t * w_gpu;
    block_q8_1 * xq_gpu;
    hipMalloc(&w_gpu, repacked_bytes);
    hipMalloc(&xq_gpu, n_blocks * sizeof(block_q8_1));
    hipMalloc(&y_gpu, ne1 * sizeof(float));

    hipMemcpy(w_gpu, repacked, repacked_bytes, hipMemcpyHostToDevice);
    hipMemcpy(xq_gpu, act_q8_1, n_blocks * sizeof(block_q8_1), hipMemcpyHostToDevice);

    hipDeviceSynchronize();

    // ── Test variant 1: ROWS=2, NWAVES=4 (256 threads, 4 waves) ──
    printf("\n=== GPU kernel: ROWS=2 NWAVES=4 (256 threads) ===\n");
    int grid_x_2x4 = (ne1 + 7) / 8;  // blocks * ROWS * NWAVES = 8 rows per block
    dim3 grid24(grid_x_2x4, 1, 1);
    dim3 block24(256, 1, 1);
    typedef void (*Kernel24)(const uint8_t*, const block_q8_1*, float*, int, int);
    Kernel24 k24 = mul_mat_vec_q8_0_repacked<2,4>;
    hipLaunchKernelGGL(k24, grid24, block24, 0, 0,
                       w_gpu, xq_gpu, y_gpu, ne0, ne1);
    hipDeviceSynchronize();
    hipMemcpy(y_host, y_gpu, ne1 * sizeof(float), hipMemcpyDeviceToHost);

    printf("GPU ROWS=2 NWAVES=4 results:\n");
    float max_err = 0.0f;
    for (int r = 0; r < ne1; r++) {
        float err = fabs(y_host[r] - ref[r]);
        if (r < 8 || err > 1e-4f)
            printf("  row %2d: gpu=%f  ref=%f  diff=%e\n", r, y_host[r], ref[r], err);
        if (err > max_err) max_err = err;
    }
    printf("Max error: %e\n", max_err);

    // ── Test variant 2: ROWS=1, NWAVES=1 (64 threads, 1 wave) ──
    printf("\n=== GPU kernel: ROWS=1 NWAVES=1 (64 threads) ===\n");
    dim3 grid11(ne1, 1, 1);
    dim3 block11(64, 1, 1);
    typedef void (*Kernel11)(const uint8_t*, const block_q8_1*, float*, int, int);
    Kernel11 k11 = mul_mat_vec_q8_0_repacked<1,1>;
    hipLaunchKernelGGL(k11, grid11, block11, 0, 0,
                       w_gpu, xq_gpu, y_gpu, ne0, ne1);
    hipDeviceSynchronize();
    hipMemcpy(y_host, y_gpu, ne1 * sizeof(float), hipMemcpyDeviceToHost);

    printf("GPU ROWS=1 NWAVES=1 results:\n");
    max_err = 0.0f;
    for (int r = 0; r < ne1; r++) {
        float err = fabs(y_host[r] - ref[r]);
        if (r < 8 || err > 1e-4f)
            printf("  row %2d: gpu=%f  ref=%f  diff=%e\n", r, y_host[r], ref[r], err);
        if (err > max_err) max_err = err;
    }
    printf("Max error: %e\n", max_err);

    // ── Verify warp reduction: dump per-lane partials ──
    // (No longer needed - error is clear enough)

    // ── Cleanup ──
    hipFree(w_gpu);
    hipFree(xq_gpu);
    hipFree(y_gpu);
    delete[] weights;
    delete[] act_f32;
    delete[] act_q8_1;
    delete[] repacked;
    delete[] ref;
    delete[] ref_repacked;
    delete[] y_host;

    return 0;
}
