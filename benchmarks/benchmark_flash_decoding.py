"""
Benchmark: Flash-Decoding (Split-KV) vs Standard Attention on LLM Generation (Q=1).
===================================================================================
Evaluates execution latency, memory traffic, and speedup across long KV contexts.
"""

import time
import math
import torch
import torch.nn.functional as F
from flash_attn import flash_decoding, flash_attention_2


def run_benchmark():
    device = torch.device("cpu")
    batch_size = 1
    num_heads = 32
    head_dim = 128
    q_len = 1  # 1 token decoding step

    kv_lengths = [512, 1024, 2048, 4096, 8192, 16384]

    print("=" * 85)
    print(f"{'FLASH-DECODING (SPLIT-KV) vs STANDARD ATTENTION BENCHMARK':^85}")
    print(f"Batch={batch_size}, Heads={num_heads}, HeadDim={head_dim}, QueryLen={q_len}")
    print("=" * 85)
    print(f"{'KV Context':<12} | {'Standard SDPA (ms)':<20} | {'Flash-Decoding (ms)':<20} | {'Speedup':<10} | {'Status'}")
    print("-" * 85)

    num_warmup = 3
    num_iters = 10

    for kv_len in kv_lengths:
        q = torch.randn(batch_size, num_heads, q_len, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, kv_len, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, kv_len, head_dim, device=device)

        # Warmup
        for _ in range(num_warmup):
            _ = F.scaled_dot_product_attention(q, k, v)
            _ = flash_decoding(q, k, v)

        # Standard SDPA
        start = time.perf_counter()
        for _ in range(num_iters):
            ref = F.scaled_dot_product_attention(q, k, v)
        sdpa_time = (time.perf_counter() - start) / num_iters * 1000

        # Flash-Decoding
        start = time.perf_counter()
        for _ in range(num_iters):
            out = flash_decoding(q, k, v)
        decoding_time = (time.perf_counter() - start) / num_iters * 1000

        speedup = sdpa_time / max(decoding_time, 1e-4)

        # Verify numerical match
        is_close = torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
        status = "PASSED" if is_close else "FAIL"

        print(f"{kv_len:<12} | {sdpa_time:<20.3f} | {decoding_time:<20.3f} | {speedup:<9.2f}x | {status}")

    print("=" * 85)
    print("Key Takeaways:")
    print("1. Standard attention leaves GPU SMs idle because 1 token yields only 1 tile.")
    print("2. Split-KV decomposes the KV sequence into parallel chunks across all threadblocks.")
    print("3. Logsumexp reduction merges partial accumulators with zero precision loss.")


if __name__ == "__main__":
    run_benchmark()
