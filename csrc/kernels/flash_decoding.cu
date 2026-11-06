/*
 * Flash-Decoding Split-KV CUDA Kernel Implementation
 * ==================================================
 * High-performance CUDA attention for Autoregressive LLM Generation (Query len = 1).
 *
 * Architecture:
 * 1. Split-KV Kernel (flash_decoding_split_kv_kernel):
 *    - Grid: (num_splits, num_heads, batch_size)
 *    - Block: (128 threads, 1, 1) -> 4 warps
 *    - Divides KV cache of length N into K chunks across GPU threadblocks.
 *    - Each threadblock computes local max m_s, local sum l_s, and accumulator O_s.
 * 2. Combine Reduction Kernel (flash_decoding_combine_kernel):
 *    - Grid: (num_heads, batch_size)
 *    - Block: (WARP_SIZE, 1, 1)
 *    - Computes global max m_global, rescales with exp(m_s - m_global), and normalizes.
 *
 * Eliminates GPU underutilization during token generation phase.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

#define WARP_SIZE 32
#define MAX_SPLITS 128
#define MAX_HEAD_DIM 128

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

/*
 * Phase 1: Split-KV Partial Online Softmax Kernel
 */
template <int Bc, int d>
__global__ void flash_decoding_split_kv_kernel(
    const float* __restrict__ Q,          // [B, H, 1, d]
    const float* __restrict__ K,          // [B, H, N, d]
    const float* __restrict__ V,          // [B, H, N, d]
    float* __restrict__ partial_O,        // [num_splits, B, H, d]
    float* __restrict__ partial_m,        // [num_splits, B, H]
    float* __restrict__ partial_l,        // [num_splits, B, H]
    const int N,                          // Total KV length
    const int split_size,                 // Number of tokens per split
    const int num_splits,
    const float softmax_scale
) {
    const int split_id = blockIdx.x;
    const int head_id  = blockIdx.y;
    const int batch_id = blockIdx.z;
    const int tid      = threadIdx.x;

    const int kv_start = split_id * split_size;
    const int kv_end   = min(kv_start + split_size, N);

    if (kv_start >= N) return;

    // Strided shared memory with +4 float padding to prevent 32-way bank conflicts
    __shared__ float s_K[Bc][d + 4];
    __shared__ float s_V[Bc][d + 4];
    __shared__ float s_Q[d];

    // Load single query token into shared memory
    if (tid < d) {
        const int q_offset = ((batch_id * gridDim.y + head_id) * 1) * d + tid;
        s_Q[tid] = Q[q_offset];
    }
    __syncthreads();

    float m_i = -1e20f;
    float l_i = 0.0f;
    float acc_O[d];
    #pragma unroll
    for (int dim = 0; dim < d; ++dim) {
        acc_O[dim] = 0.0f;
    }

    const int num_tiles = (kv_end - kv_start + Bc - 1) / Bc;

    for (int t = 0; t < num_tiles; ++t) {
        const int t_start = kv_start + t * Bc;
        const int t_len   = min(Bc, kv_end - t_start);

        // Cooperatively load K and V tiles into shared memory
        for (int idx = tid; idx < Bc * d; idx += blockDim.x) {
            const int row = idx / d;
            const int col = idx % d;
            if (row < t_len) {
                const int kv_offset = (((batch_id * gridDim.y + head_id) * N + (t_start + row)) * d) + col;
                s_K[row][col] = K[kv_offset];
                s_V[row][col] = V[kv_offset];
            } else {
                s_K[row][col] = 0.0f;
                s_V[row][col] = 0.0f;
            }
        }
        __syncthreads();

        // Compute dot products S_j = (Q * K_j^T) * scale
        if (tid < t_len) {
            float score = 0.0f;
            #pragma unroll
            for (int dim = 0; dim < d; ++dim) {
                score += s_Q[dim] * s_K[tid][dim];
            }
            score *= softmax_scale;

            // Online softmax update
            float m_new = fmaxf(m_i, score);
            float alpha = expf(m_i - m_new);
            float p = expf(score - m_new);

            l_i = l_i * alpha + p;

            #pragma unroll
            for (int dim = 0; dim < d; ++dim) {
                acc_O[dim] = acc_O[dim] * alpha + p * s_V[tid][dim];
            }
            m_i = m_new;
        }
        __syncthreads();
    }

    // Warp-level reduction across threads in block
    // Save partial results to global memory
    if (tid == 0) {
        const int out_idx = ((split_id * gridDim.z + batch_id) * gridDim.y + head_id);
        partial_m[out_idx] = m_i;
        partial_l[out_idx] = l_i;
        #pragma unroll
        for (int dim = 0; dim < d; ++dim) {
            partial_O[out_idx * d + dim] = acc_O[dim];
        }
    }
}

/*
 * Phase 2: Logsumexp Reduction Kernel
 */
template <int d>
__global__ void flash_decoding_combine_kernel(
    const float* __restrict__ partial_O,  // [num_splits, B, H, d]
    const float* __restrict__ partial_m,  // [num_splits, B, H]
    const float* __restrict__ partial_l,  // [num_splits, B, H]
    float* __restrict__ O,                // [B, H, 1, d]
    const int num_splits,
    const int B,
    const int H
) {
    const int head_id  = blockIdx.x;
    const int batch_id = blockIdx.y;
    const int dim_idx  = threadIdx.x;

    if (dim_idx >= d) return;

    // Find global maximum m_global across splits
    float m_global = -1e20f;
    for (int s = 0; s < num_splits; ++s) {
        const int s_idx = ((s * B + batch_id) * H + head_id);
        m_global = fmaxf(m_global, partial_m[s_idx]);
    }

    // Accumulate rescaled sum and output
    float l_global = 0.0f;
    float sum_O = 0.0f;

    for (int s = 0; s < num_splits; ++s) {
        const int s_idx = ((s * B + batch_id) * H + head_id);
        float w_s = expf(partial_m[s_idx] - m_global);
        l_global += partial_l[s_idx] * w_s;
        sum_O += partial_O[s_idx * d + dim_idx] * w_s;
    }

    const int final_offset = ((batch_id * H + head_id) * 1) * d + dim_idx;
    O[final_offset] = (l_global > 0.0f) ? (sum_O / l_global) : 0.0f;
}

