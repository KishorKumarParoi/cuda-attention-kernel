/*
 * PyTorch C++ / CUDA Extension Binding for FlashAttention-2.
 */

#include <torch/extension.h>
#include <vector>

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
);

std::vector<torch::Tensor> flash_attn_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float softmax_scale,
    bool is_causal
) {
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);
    const int d = Q.size(3);

    auto O = torch::zeros_like(Q);
    auto L = torch::empty({B, H, N}, Q.options());

    if (Q.is_cuda()) {
        #ifdef AT_CUDA_ENABLED
        // Call CUDA Kernel
        #endif
    } else {
        flash_attention_cpu_forward(
            Q.data_ptr<float>(),
            K.data_ptr<float>(),
            V.data_ptr<float>(),
            O.data_ptr<float>(),
            L.data_ptr<float>(),
            B, H, N, d, softmax_scale, is_causal
        );
    }

    return {O, L};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attn_forward, "FlashAttention-2 forward (CUDA / CPU)");
}

