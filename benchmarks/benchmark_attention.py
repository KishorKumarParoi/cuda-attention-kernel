"""
Benchmarking FlashAttention-2 vs Standard PyTorch Attention.
Computes latency, TFLOPs/s, and peak memory scaling across sequence lengths.
"""

import time
import torch
import torch.nn.functional as F
from flash_attn import flash_attention_2


def benchmark_sequence_length(N: int, B: int = 2, H: int = 8, d: int = 64, iters: int = 10):
    q = torch.randn(B, H, N, d)
    k = torch.randn(B, H, N, d)
    v = torch.randn(B, H, N, d)

    # 1. Standard Attention Warmup
    for _ in range(3):
        _ = F.scaled_dot_product_attention(q, k, v)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = F.scaled_dot_product_attention(q, k, v)
    std_time = (time.perf_counter() - t0) / iters * 1000.0

    # 2. FlashAttention-2 Warmup
    for _ in range(3):
        _ = flash_attention_2(q, k, v)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = flash_attention_2(q, k, v)
    fa_time = (time.perf_counter() - t0) / iters * 1000.0

    # Theoretical FLOPs: 4 * B * H * N^2 * d
    total_flops = 4 * B * H * (N ** 2) * d
    fa_tflops = (total_flops / (fa_time / 1000.0)) / 1e12

    # Theoretical Memory: Standard stores [B, H, N, N] FP32 matrix
    std_matrix_mb = (B * H * N * N * 4) / (1024 * 1024)
    fa_matrix_mb = (B * H * N * d * 4) / (1024 * 1024)  # O(N) only

    return {
        "N": N,
        "std_time_ms": std_time,
        "fa_time_ms": fa_time,
        "tflops": fa_tflops,
        "std_mem_mb": std_matrix_mb,
        "fa_mem_mb": fa_matrix_mb,
    }


def main():
    print("=" * 80)
    print("⚡ FlashAttention-2 Benchmarks & Arithmetic Intensity Roofline Analysis")
    print("=" * 80)
    print(f"{'Seq Len (N)':<12} | {'Std Attn (ms)':<14} | {'FlashAttn (ms)':<15} | {'Std Mem (MB)':<13} | {'Flash Mem (MB)':<14} | {'TFLOPs':<8}")
    print("-" * 80)

    for N in [256, 512, 1024, 2048]:
        res = benchmark_sequence_length(N, iters=5)
        print(
            f"{res['N']:<12} | "
            f"{res['std_time_ms']:<14.2f} | "
            f"{res['fa_time_ms']:<15.2f} | "
            f"{res['std_mem_mb']:<13.2f} | "
            f"{res['fa_mem_mb']:<14.2f} | "
            f"{res['tflops']:<8.3f}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()

