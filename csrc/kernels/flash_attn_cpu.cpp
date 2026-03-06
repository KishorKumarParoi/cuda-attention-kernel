/*
 * FlashAttention-2 High-Performance CPU / SIMD Reference Implementation.
 * Algorithmic mirror of the CUDA kernel executing tiled online softmax in L1/L2 cache.
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

void flash_attention_cpu_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    float* L,
    int B,
    int H,
    int N,
    int d,
    float scale,
    bool is_causal
) {
    const int Br = 32;
    const int Bc = 32;

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            const int head_offset = (b * H + h) * N * d;
            const int l_offset = (b * H + h) * N;

            const float* q_head = Q + head_offset;
            const float* k_head = K + head_offset;
            const float* v_head = V + head_offset;
            float* o_head = O + head_offset;
            float* l_head = L + l_offset;

            // Iterate over Query blocks
            for (int q_b = 0; q_b < N; q_b += Br) {
                int q_end = std::min(q_b + Br, N);

                for (int i = q_b; i < q_end; ++i) {
                    float m_i = -1e9f;
                    float l_i = 0.0f;
                    std::vector<float> acc_o(d, 0.0f);

                    // Iterate over Key/Value blocks
                    for (int k_b = 0; k_b < N; k_b += Bc) {
                        if (is_causal && k_b > i) break;
                        int k_end = std::min(k_b + Bc, N);

                        std::vector<float> S_block(k_end - k_b);
                        float block_max = -1e9f;

                        for (int j = k_b; j < k_end; ++j) {
                            if (is_causal && j > i) {
                                S_block[j - k_b] = -1e9f;
                            } else {
                                float dot = 0.0f;
                                for (int c = 0; c < d; ++c) {
                                    dot += q_head[i * d + c] * k_head[j * d + c];
                                }
                                S_block[j - k_b] = dot * scale;
                                block_max = std::max(block_max, S_block[j - k_b]);
                            }
                        }

                        // Online Softmax Rescale
                        float m_new = std::max(m_i, block_max);
                        float exp_scale = std::exp(m_i - m_new);
                        float l_new = l_i * exp_scale;

                        for (int c = 0; c < d; ++c) {
                            acc_o[c] *= exp_scale;
                        }

                        for (int j = k_b; j < k_end; ++j) {
                            if (is_causal && j > i) continue;
                            float p = std::exp(S_block[j - k_b] - m_new);
                            l_new += p;
                            for (int c = 0; c < d; ++c) {
                                acc_o[c] += p * v_head[j * d + c];
                            }
                        }

                        m_i = m_new;
                        l_i = l_new;
                    }

                    // Normalize and write to O
                    float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
                    for (int c = 0; c < d; ++c) {
                        o_head[i * d + c] = acc_o[c] * inv_l;
                    }
                    l_head[i] = m_i + std::log(std::max(l_i, 1e-6f));
                }
            }
        }
    }
}

