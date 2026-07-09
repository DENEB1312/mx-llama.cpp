#include "gated_delta_net_chunk.cuh"
#include "ggml-cuda/common.cuh"

// Chunked Gated DeltaNet kernel.
//
// This is a *fused* kernel: one thread-block owns a single (head, sequence) and
// processes the whole sequence by looping over chunks of CS tokens.  The recurrent
// state S_v x S_v stays in registers across chunks (it is never written back to
// global memory between chunks), which makes a long prefill a single kernel launch
// with no state round-trips.
//
// The arithmetic is identical to the per-token `gated_delta_net_cuda` recurrence
// (gated_delta_net.cu), so the chunked output is bit-for-bit equal to what the
// per-token kernel produces when run over the same tokens.  CS = 64 for the
// non-KDA path and CS = 16 for the KDA (per-feature gate) path.
template <int S_v, bool KDA, int CS, int Tc>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_chunked_cuda(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ g,
        const float * __restrict__ beta,
        const float * __restrict__ curr_state,
        float *       __restrict__ dst,
        int64_t       H,
        int64_t       n_tokens,
        int64_t       n_seqs,
        int64_t       sq1,
        int64_t       sq2,
        int64_t       sq3,
        int64_t       sv1,
        int64_t       sv2,
        int64_t       sv3,
        int64_t       sb1,
        int64_t       sb2,
        int64_t       sb3,
        const uint3   neqk1_magic,
        const uint3   rq3_magic,
        float         scale,
        int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // lane indexes a row of the S_v x S_v state; threadIdx.y + blockIdx.z index a
    // band of Tc consecutive columns owned by this warp.
    const int      lane     = threadIdx.x;
    const int      warp_id  = threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t attn_score_elems = (int64_t) S_v * H * n_tokens * n_seqs;
    float *       state_out = dst + attn_score_elems;

    // input state layout (D, K, n_seqs): seq stride = K * S_v * S_v
    const int64_t state_in_offset  = sequence * K * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state_out += state_out_offset;
    const float * state_in_base = curr_state + state_in_offset;

    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;

    // Each warp owns Tc consecutive columns.  k/q/g/beta depend only on the row
    // (or head), not on the column, so they are loaded/recomputed ONCE per warp
    // and reused across its Tc columns.  This removes the S_v-way redundant global
    // loads and expf evaluations of the 1-column-per-warp layout, and improves ILP,
    // at the cost of Tc*rows_per_lane extra VGPRs for the state (cheap: <= 16).
    const int base_col = (blockIdx.z * blockDim.y + warp_id) * Tc;

    // s_shard[r][c] holds row (r*warp_size+lane) of state column (base_col + c).
    float s_shard[rows_per_lane][Tc];
    #pragma unroll
    for (int c = 0; c < Tc; c++) {
        const int col = base_col + c;
        #pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            s_shard[r][c] = (col < S_v && i < S_v) ? state_in_base[col * S_v + i] : 0.0f;
        }
    }

        ggml_cuda_pdl_sync();

    const int n_chunks = (n_tokens + CS - 1) / CS;
    for (int c = 0; c < n_chunks; c++) {
        const int t_base = c * CS;
        const int n_local = (t_base + CS <= n_tokens) ? CS : (n_tokens - t_base);

        // ---- process the chunk token-by-token (exact recurrence) ----
        for (int p = 0; p < n_local; p++) {
            const int t = t_base + p;
            const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;
            const int64_t gb = sequence * sb3 + t * sb2 + h_idx * sb1;

            // Load per-token data into registers.  k/q are per-row (shared across all
            // columns) so a single load per warp serves the whole Tc-column band.
            float k_shard[rows_per_lane];
            float q_shard[rows_per_lane];
            #pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                k_shard[r] = k_t[i];
                q_shard[r] = q_t[i];
            }
            const float beta_val = beta[gb];

            if constexpr (!KDA) {
                const float g_val = expf(g[gb]);

                for (int cc = 0; cc < Tc; cc++) {
                    const int col = base_col + cc;
                    if (col >= S_v) break;
                    const float v_col = v_t[col];

                    float kv_shard = 0.0f;
                    #pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        kv_shard += s_shard[r][cc] * k_shard[r];
                    }
                    float kv_col = warp_reduce_sum<warp_size>(kv_shard);

                    float delta_col = (v_col - g_val * kv_col) * beta_val;

                    float attn_partial = 0.0f;
                    #pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        s_shard[r][cc] = g_val * s_shard[r][cc] + k_shard[r] * delta_col;
                        attn_partial += s_shard[r][cc] * q_shard[r];
                    }

                    float attn_col = warp_reduce_sum<warp_size>(attn_partial);

                    if (lane == 0) {
                        attn_data[col] = attn_col * scale;
                    }
                }
            } else {
                // KDA: per-feature gate.  expf(g) is per-row, shared across columns,
                // so evaluate it once per warp and reuse for every column in the band.
                const float * g_t = g + gb * S_v;
                float eg_shard[rows_per_lane];
                #pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    eg_shard[r] = (i < S_v) ? expf(g_t[i]) : 1.0f;
                }

                for (int cc = 0; cc < Tc; cc++) {
                    const int col = base_col + cc;
                    if (col >= S_v) break;
                    const float v_col = v_t[col];

                    float kv_shard = 0.0f;
                    #pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        kv_shard += eg_shard[r] * s_shard[r][cc] * k_shard[r];
                    }
                    float kv_col = warp_reduce_sum<warp_size>(kv_shard);

                    float delta_col = (v_col - kv_col) * beta_val;

                    float attn_partial = 0.0f;
                    #pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        s_shard[r][cc] = eg_shard[r] * s_shard[r][cc] + k_shard[r] * delta_col;
                        attn_partial += s_shard[r][cc] * q_shard[r];
                    }

                    float attn_col = warp_reduce_sum<warp_size>(attn_partial);

                    if (lane == 0) {
                        attn_data[col] = attn_col * scale;
                    }
                }
            }

            attn_data += (int64_t) S_v * H;
        }
    }

    // ---- write final state ----
    #pragma unroll
    for (int c = 0; c < Tc; c++) {
        const int col = base_col + c;
        if (col >= S_v) break;
        #pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            if (i < S_v) {
                state_out[col * S_v + i] = s_shard[r][c];
            }
        }
    }
}

template <bool KDA, bool /*keep_rs_t*/>
void launch_gated_delta_net_chunk(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int K, cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int CS = KDA ? 16 : 64;
    const int num_warps = 4;
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    // Tc = columns owned per warp.  Keep Tc*rows_per_lane VGPRs for the state
    // small (<= ~16) while maximizing the load/expf reuse across columns.
    switch (S_v) {
        case 16: {
            constexpr int Tc = 4;
            dim3 grid_dims(H, n_seqs, (S_v + num_warps * Tc - 1) / (num_warps * Tc));
            const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
            ggml_cuda_kernel_launch(gated_delta_net_chunked_cuda<16, KDA, KDA ? 16 : 64, Tc>, lp,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, K);
            break;
        }
        case 32: {
            constexpr int Tc = 4;
            dim3 grid_dims(H, n_seqs, (S_v + num_warps * Tc - 1) / (num_warps * Tc));
            const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
            ggml_cuda_kernel_launch(gated_delta_net_chunked_cuda<32, KDA, KDA ? 16 : 64, Tc>, lp,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, K);
            break;
        }
        case 64: {
            constexpr int Tc = 4;
            dim3 grid_dims(H, n_seqs, (S_v + num_warps * Tc - 1) / (num_warps * Tc));
            const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
            ggml_cuda_kernel_launch(gated_delta_net_chunked_cuda<64, KDA, KDA ? 16 : 64, Tc>, lp,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, K);
            break;
        }
        case 128: {
            constexpr int Tc = 2;
            dim3 grid_dims(H, n_seqs, (S_v + num_warps * Tc - 1) / (num_warps * Tc));
            const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
            ggml_cuda_kernel_launch(gated_delta_net_chunked_cuda<128, KDA, KDA ? 16 : 64, Tc>, lp,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_op_gated_delta_net_chunk(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;
    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;
    const float * s_d = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    const int K = (int) src_state->ne[1];
    const bool keep_rs = K > 1;

    // Chunked kernel does not support keep_rs_t (K>1)
    GGML_ASSERT(!keep_rs && "chunked kernel does not support K>1");

    // Chunked kernel always uses keep_rs_t = false
    if (kda) {
        launch_gated_delta_net_chunk<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, K, stream);
    } else {
        launch_gated_delta_net_chunk<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, K, stream);
    }
}

// No explicit instantiations needed — function is called directly from dispatch.
