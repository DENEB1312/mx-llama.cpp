#pragma once

#include "common.cuh"

// GCN (gfx906) weight-repacking buffer type for Q8_0 weights.
//
// Q8_0 weights are repacked into a two-plane layout: 32 aligned qs bytes
// per sub-block, then the fp16 d-scales as their own stream. The planes
// are byte-identical to the on-disk block_q8_0 (modulo row-stride
// padding), so the win is purely in tiling/threading of the GEMM and
// matvec kernels, not in bandwidth.
//
// Layout (per 2D tensor [ne0 = K, ne1 = M], row-major rows of K):
//   qs plane:   M * nsp * 32 bytes — one 32-byte chunk per 32-weight
//               sub-block; q8 values packed as on-disk.
//   d plane:    M * nsp * 2 bytes — per sub-block the fp16 scale.
// nsp = K/32, plus one padding sub-block when K/32 is a power of two
// (a power-of-two row stride aliases every row onto the same HBM
// channel: ~3x matvec slowdown).
//
// Enabled with GGML_CUDA_REPACK=1 on GCN devices only. Q8_0 repack is its
// own opt-in (GGML_CUDA_REPACK_Q8_0 != 0). Weights placed in this buffer
// type cannot be read back (get_tensor aborts).

bool ggml_backend_buft_is_cuda_repack(ggml_backend_buffer_type_t buft);

// Returns the repack buffer type for `device`, or nullptr when disabled
// (env GGML_CUDA_REPACK unset/0) or the device is not GCN.
ggml_backend_buffer_type_t ggml_backend_cuda_repack_buffer_type(int device);

// True if `t` can live in the repack buffer type: 2D/3D Q8_0 with
// ne0 % 32 == 0 (when GGML_CUDA_REPACK_Q8_0 != 0).
bool ggml_cuda_repack_tensor_supported(const ggml_tensor * t);

// MUL_MAT with src0 in the repack buffer type. ne11 == 1 runs the
// repacked dp4a matvec; larger ne11 runs the repacked int8 MMQ GEMM.
// 3D/4D src1 broadcasts the 2D weight per slice.
void ggml_cuda_mul_mat_repacked(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// MUL_MAT_ID with 3D src0 (per-expert repacked slabs) in the repack
// buffer type. Decode (one token) runs one matvec per assignment;
// batches run a grouped tile GEMM over expert-sorted assignments.
void ggml_cuda_mul_mat_id_repacked(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
    ggml_tensor * dst);
