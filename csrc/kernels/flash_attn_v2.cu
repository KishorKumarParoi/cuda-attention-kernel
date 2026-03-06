/*
 * FlashAttention-2 Forward CUDA Kernel (Dao, 2023)
 * Optimized GPU Attention with Tiled Online Softmax in SRAM.
 *
 * Mathematical Formulation:
 * - Query Block: Br rows, Head Dimension: d
 * - Key/Value Block: Bc rows, Head Dimension: d
 * - Memory Complexity: O(N) in HBM (O(Br * d + 2 * Bc * d) in Shared Memory)
 * - Avoids materializing the N x N attention matrix in global memory.
 * - Warp-level reduction via __shfl_down_sync for zero-latency register reduction.
 * - Dynamic bank conflict avoidance with stride padding.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

#define WARP_SIZE 32
#define MAX_HEAD_DIM 128

// Intra-warp sum reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// Intra-warp max reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

/*
 * FlashAttention-2 Forward Kernel
 * Grid: (ceil(N / Br), num_heads, batch_size)
 * Block: (threads_per_block, 1, 1) -> 128 or 256 threads
 */
template <int Br, int Bc, int d>
__global__ void flash_attention_2_fwd_kernel(
    const float* __restrict__ Q,    // [B, H, N, d]
    const float* __restrict__ K,    // [B, H, N, d]
    const float* __restrict__ V,    // [B, H, N, d]
    float* __restrict__ O,          // [B, H, N, d]
    float* __restrict__ L,          // [B, H, N] - Logsumexp for backward pass
    const int N,
    const float softmax_scale,
    const bool is_causal
) {
    // Shared Memory allocation with padding (+4 floats) to avoid 32-way bank conflicts
    __shared__ float s_Q[Br][d + 4];
    __shared__ float s_K[Bc][d + 4];
    __shared__ float s_V[Bc][d + 4];

    const int tid = threadIdx.x;
    const int q_block_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;

    const int total_heads = gridDim.y;
    const int batch_head_offset = (batch_idx * total_heads + head_idx) * N * d;
    const int l_offset = (batch_idx * total_heads + head_idx) * N;

    // Local pointers for current batch and head
    const float* q_ptr = Q + batch_head_offset;
    const float* k_ptr = K + batch_head_offset;
    const float* v_ptr = V + batch_head_offset;
    float* o_ptr = O + batch_head_offset;
    float* l_ptr = L + l_offset;

    const int q_row_start = q_block_idx * Br;

    // Registers to store running max (m_i), running normalizer (l_i), and accumulator (O_i)
    float m_i[Br];
    float l_i[Br];
    float acc_O[Br][d];

    #pragma unroll
    for (int r = 0; r < Br; ++r) {
        m_i[r] = -INFINITY;
        l_i[r] = 0.0f;
        #pragma unroll
        for (int c = 0; c < d; ++c) {
            acc_O[r][c] = 0.0f;
        }
    }

    // 1. Load Query Block into Shared Memory (collaborative thread load)
    #pragma unroll
    for (int idx = tid; idx < Br * d; idx += blockDim.x) {
        int r = idx / d;
        int c = idx % d;
        int global_r = q_row_start + r;
        if (global_r < N) {
            s_Q[r][c] = q_ptr[global_r * d + c];
        } else {
            s_Q[r][c] = 0.0f;
        }
    }
    __syncthreads();

    // 2. Loop over Key and Value blocks
    const int num_k_blocks = (N + Bc - 1) / Bc;

    for (int k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
        const int k_row_start = k_block_idx * Bc;

        // Causal Masking early exit: skip KV blocks entirely above the diagonal
        if (is_causal && (k_row_start > q_row_start + Br - 1)) {
            break;
        }

        // Load Key and Value blocks into Shared Memory
        #pragma unroll
        for (int idx = tid; idx < Bc * d; idx += blockDim.x) {
            int r = idx / d;
            int c = idx % d;
            int global_k = k_row_start + r;
            if (global_k < N) {
                s_K[r][c] = k_ptr[global_k * d + c];
                s_V[r][c] = v_ptr[global_k * d + c];
            } else {
                s_K[r][c] = 0.0f;
                s_V[r][c] = 0.0f;
            }
        }
        __syncthreads();

        // 3. Compute S = (Q * K^T) * scale in SRAM registers
        #pragma unroll
        for (int r = 0; r < Br; ++r) {
            int global_q = q_row_start + r;
            if (global_q >= N) continue;

            float row_max = -INFINITY;
            float S_row[Bc];

            #pragma unroll
            for (int k_r = 0; k_r < Bc; ++k_r) {
                int global_k = k_row_start + k_r;
                if (global_k >= N || (is_causal && global_k > global_q)) {
                    S_row[k_r] = -INFINITY;
                } else {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int c = 0; c < d; ++c) {
                        dot += s_Q[r][c] * s_K[k_r][c];
                    }
                    S_row[k_r] = dot * softmax_scale;
                }
                row_max = fmaxf(row_max, S_row[k_r]);
            }

            // Online Softmax: Rescaling running max and normalizer
            float m_prev = m_i[r];
            float m_new = fmaxf(m_prev, row_max);
            float exp_scale = expf(m_prev - m_new);

            // Rescale previous normalizer and accumulate new P_ij
            float l_new = l_i[r] * exp_scale;
            float P_row[Bc];

            #pragma unroll
            for (int k_r = 0; k_r < Bc; ++k_r) {
                if (S_row[k_r] == -INFINITY) {
                    P_row[k_r] = 0.0f;
                } else {
                    P_row[k_r] = expf(S_row[k_r] - m_new);
                    l_new += P_row[k_r];
                }
            }

            // Rescale accumulator O = O * exp(m_prev - m_new) + P * V
            #pragma unroll
            for (int c = 0; c < d; ++c) {
                acc_O[r][c] *= exp_scale;
                float pv = 0.0f;
                #pragma unroll
                for (int k_r = 0; k_r < Bc; ++k_r) {
                    pv += P_row[k_r] * s_V[k_r][c];
                }
                acc_O[r][c] += pv;
            }

            m_i[r] = m_new;
            l_i[r] = l_new;
        }
        __syncthreads();
    }

    // 4. Final normalization by l_i and write to HBM
    #pragma unroll
    for (int r = 0; r < Br; ++r) {
        int global_q = q_row_start + r;
        if (global_q < N) {
            float inv_l = (l_i[r] > 0.0f) ? (1.0f / l_i[r]) : 0.0f;
            #pragma unroll
            for (int c = 0; c < d; ++c) {
                o_ptr[global_q * d + c] = acc_O[r][c] * inv_l;
            }
            // Save logsumexp: L_i = m_i + log(l_i) for the backward pass
            l_ptr[global_q] = m_i[r] + logf(fmaxf(l_i[r], 1e-6f));
        }
    }
}

