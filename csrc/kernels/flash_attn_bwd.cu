/*
 * FlashAttention-2 Backward CUDA Kernel (Dao, 2023)
 * Recomputes Attention Matrix P in SRAM to achieve O(N) memory backward pass.
 *
 * Gradients computed:
 * - dV = P^T * dO
 * - dP = dO * V^T
 * - D_i = sum_j (dO_ij * O_ij)
 * - dS_ij = P_ij * (dP_ij - D_i)
 * - dQ = (dS * K) * scale
 * - dK = (dS^T * Q) * scale
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

template <int Br, int Bc, int d>
__global__ void flash_attention_2_bwd_kernel(
    const float* __restrict__ Q,    // [B, H, N, d]
    const float* __restrict__ K,    // [B, H, N, d]
    const float* __restrict__ V,    // [B, H, N, d]
    const float* __restrict__ O,    // [B, H, N, d]
    const float* __restrict__ dO,   // [B, H, N, d]
    const float* __restrict__ L,    // [B, H, N]
    float* __restrict__ dQ,         // [B, H, N, d]
    float* __restrict__ dK,         // [B, H, N, d]
    float* __restrict__ dV,         // [B, H, N, d]
    const int N,
    const float softmax_scale,
    const bool is_causal
) {
    __shared__ float s_Q[Br][d + 4];
    __shared__ float s_K[Bc][d + 4];
    __shared__ float s_V[Bc][d + 4];
    __shared__ float s_dO[Br][d + 4];

    const int tid = threadIdx.x;
    const int kv_block_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;

    const int total_heads = gridDim.y;
    const int offset = (batch_idx * total_heads + head_idx) * N * d;
    const int l_offset = (batch_idx * total_heads + head_idx) * N;

    const float* q_ptr = Q + offset;
    const float* k_ptr = K + offset;
    const float* v_ptr = V + offset;
    const float* o_ptr = O + offset;
    const float* do_ptr = dO + offset;
    const float* l_ptr = L + l_offset;

    float* dq_ptr = dQ + offset;
    float* dk_ptr = dK + offset;
    float* dv_ptr = dV + offset;

    const int k_row_start = kv_block_idx * Bc;

    // Load Key and Value blocks into SRAM
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

    // Accumulators for dK and dV
    float acc_dK[Bc][d];
    float acc_dV[Bc][d];
    #pragma unroll
    for (int r = 0; r < Bc; ++r) {
        #pragma unroll
        for (int c = 0; c < d; ++c) {
            acc_dK[r][c] = 0.0f;
            acc_dV[r][c] = 0.0f;
        }
    }

    const int num_q_blocks = (N + Br - 1) / Br;

    for (int q_block_idx = 0; q_block_idx < num_q_blocks; ++q_block_idx) {
        const int q_row_start = q_block_idx * Br;

        if (is_causal && (q_row_start + Br - 1 < k_row_start)) {
            continue;
        }

        // Load Query and dO blocks into SRAM
        #pragma unroll
        for (int idx = tid; idx < Br * d; idx += blockDim.x) {
            int r = idx / d;
            int c = idx % d;
            int global_q = q_row_start + r;
            if (global_q < N) {
                s_Q[r][c] = q_ptr[global_q * d + c];
                s_dO[r][c] = do_ptr[global_q * d + c];
            } else {
                s_Q[r][c] = 0.0f;
                s_dO[r][c] = 0.0f;
            }
        }
        __syncthreads();

        // Recompute P_ij and accumulate dV, dK
        #pragma unroll
        for (int r = 0; r < Br; ++r) {
            int global_q = q_row_start + r;
            if (global_q >= N) continue;

            float l_val = l_ptr[global_q];
            float D_i = 0.0f;
            #pragma unroll
            for (int c = 0; c < d; ++c) {
                D_i += s_dO[r][c] * o_ptr[global_q * d + c];
            }

            #pragma unroll
            for (int k_r = 0; k_r < Bc; ++k_r) {
                int global_k = k_row_start + k_r;
                if (global_k >= N || (is_causal && global_k > global_q)) continue;

                float dot = 0.0f;
                #pragma unroll
                for (int c = 0; c < d; ++c) {
                    dot += s_Q[r][c] * s_K[k_r][c];
                }
                float S_val = dot * softmax_scale;
                float P_val = expf(S_val - l_val);

                // Accumulate dV: dV = P^T * dO
                #pragma unroll
                for (int c = 0; c < d; ++c) {
                    acc_dV[k_r][c] += P_val * s_dO[r][c];
                }

                // Compute dP = dO * V^T
                float dP_val = 0.0f;
                #pragma unroll
                for (int c = 0; c < d; ++c) {
                    dP_val += s_dO[r][c] * s_V[k_r][c];
                }

                // dS = P * (dP - D_i)
                float dS_val = P_val * (dP_val - D_i) * softmax_scale;

                // Accumulate dK: dK = dS^T * Q
                #pragma unroll
                for (int c = 0; c < d; ++c) {
                    acc_dK[k_r][c] += dS_val * s_Q[r][c];
                }

                // Atomic add to dQ: dQ = dS * K
                #pragma unroll
                for (int c = 0; c < d; ++c) {
                    atomicAdd(&dq_ptr[global_q * d + c], dS_val * s_K[k_r][c]);
                }
            }
        }
        __syncthreads();
    }

    // Write dK and dV back to global HBM
    #pragma unroll
    for (int idx = tid; idx < Bc * d; idx += blockDim.x) {
        int r = idx / d;
        int c = idx % d;
        int global_k = k_row_start + r;
        if (global_k < N) {
            atomicAdd(&dk_ptr[global_k * d + c], acc_dK[r][c]);
            atomicAdd(&dv_ptr[global_k * d + c], acc_dV[r][c]);
        }
    }
}

