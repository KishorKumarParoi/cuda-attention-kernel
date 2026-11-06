/*
 * Paged Attention CUDA Kernel Implementation
 * ==========================================
 * Non-contiguous physical memory lookup for autoregressive KV caches.
 *
 * Architecture:
 * - kv_cache: [num_blocks, 2, num_heads, block_size, head_dim]
 * - block_tables: [batch_size, max_blocks_per_seq]
 * - context_lens: [batch_size]
 *
 * Eliminates GPU memory fragmentation (reducing waste from 60-80% to <4%).
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

#define WARP_SIZE 32

template <int block_size, int d>
__global__ void paged_attention_kernel(
    const float* __restrict__ Q,              // [B, H, 1, d]
    const float* __restrict__ kv_cache,       // [num_blocks, 2, H, block_size, d]
    const int*   __restrict__ block_tables,   // [B, max_blocks_per_seq]
    const int*   __restrict__ context_lens,   // [B]
    float*       __restrict__ O,              // [B, H, 1, d]
    const int max_blocks_per_seq,
    const int H,
    const float softmax_scale
) {
    const int head_id  = blockIdx.x;
    const int batch_id = blockIdx.y;
    const int tid      = threadIdx.x;

    const int total_tokens = context_lens[batch_id];
    if (total_tokens <= 0) return;

    __shared__ float s_Q[d];
    if (tid < d) {
        const int q_offset = ((batch_id * H + head_id) * 1) * d + tid;
        s_Q[tid] = Q[q_offset];
    }
    __syncthreads();

    const int num_blocks = (total_tokens + block_size - 1) / block_size;

    float m_i = -1e20f;
    float l_i = 0.0f;
    float acc_O[d];
    #pragma unroll
    for (int dim = 0; dim < d; ++dim) {
        acc_O[dim] = 0.0f;
    }

    for (int b_idx = 0; b_idx < num_blocks; ++b_idx) {
        const int phys_block = block_tables[batch_id * max_blocks_per_seq + b_idx];
        const int tokens_in_block = min(block_size, total_tokens - b_idx * block_size);

        for (int tok = 0; tok < tokens_in_block; ++tok) {
            // Compute dot product with Key
            float score = 0.0f;
            #pragma unroll
            for (int dim = 0; dim < d; ++dim) {
                // kv_cache stride: phys_block * 2 * H * block_size * d + 0 * H * block_size * d + head_id * block_size * d + tok * d + dim
                const int k_idx = (((phys_block * 2 + 0) * H + head_id) * block_size + tok) * d + dim;
                score += s_Q[dim] * kv_cache[k_idx];
            }
            score *= softmax_scale;

            float m_new = fmaxf(m_i, score);
            float alpha = expf(m_i - m_new);
            float p = expf(score - m_new);

            l_i = l_i * alpha + p;

            #pragma unroll
            for (int dim = 0; dim < d; ++dim) {
                const int v_idx = (((phys_block * 2 + 1) * H + head_id) * block_size + tok) * d + dim;
                acc_O[dim] = acc_O[dim] * alpha + p * kv_cache[v_idx];
            }
            m_i = m_new;
        }
    }

    if (tid < d) {
        const int out_idx = ((batch_id * H + head_id) * 1) * d + tid;
        O[out_idx] = (l_i > 0.0f) ? (acc_O[tid] / l_i) : 0.0f;
    }
}
