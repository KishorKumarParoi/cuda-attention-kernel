#!/usr/bin/env python3
"""
FlashAttention-2 & FlashAttention-3 Roofline & Arithmetic Intensity Analyzer
=============================================================================
Calculates exact FLOPs, memory transfer volumes, and operational intensity
for Standard Attention vs FlashAttention across NVIDIA Ampere & Hopper GPUs.

Author: Kishor Kumar Paroi <1703053@student.ruet.ac.bd>
"""

import math
import sys

GPU_SPECS = {
    "NVIDIA A100 (SXM4 80GB)": {
        "peak_bf16_tflops": 312.0,
        "peak_hbm_tb_s": 2.039,
        "sram_kb_per_sm": 164,
        "num_sms": 108,
    },
    "NVIDIA H100 (SXM5 80GB)": {
        "peak_bf16_tflops": 989.0,
        "peak_hbm_tb_s": 3.350,
        "sram_kb_per_sm": 228,
        "num_sms": 132,
    },
}

def analyze_attention_workload(seq_len: int, batch: int = 2, heads: int = 32, dim: int = 128, bytes_per_elem: int = 2):
    # Total FLOPs: QK^T is 2 * B * H * N^2 * d, PV is 2 * B * H * N^2 * d
    total_flops = 4 * batch * heads * (seq_len ** 2) * dim
    
    # Standard Attention HBM Transfer (Words):
    # Read Q, K (2 * B * H * N * d)
    # Write S (B * H * N^2)
    # Read S (B * H * N^2)
    # Write P (B * H * N^2)
    # Read P, V (B * H * N^2 + B * H * N * d)
    # Write O (B * H * N * d)
    std_elements = 4 * batch * heads * seq_len * dim + 4 * batch * heads * (seq_len ** 2)
    std_bytes = std_elements * bytes_per_elem
    std_intensity = total_flops / max(std_bytes, 1)

    # FlashAttention-2 HBM Transfer (Tri Dao, 2023 Theorem 1):
    # Outer loop over Q tiles (Br = 64): Q is loaded once (B * H * N * d)
    # Inner loop over K, V tiles (Bc = 64): K and V are loaded (N / Br) times.
    # Write Output O: (B * H * N * d)
    tile_br = 64
    flash_elements = (2 * batch * heads * seq_len * dim) + (2 * batch * heads * (seq_len ** 2) * dim // tile_br)
    # Plus O write:
    flash_elements += batch * heads * seq_len * dim
    flash_bytes = flash_elements * bytes_per_elem
    
    # In pure SRAM resident FlashAttention (when M is large enough to cache blocks):
    # FlashAttention-2 memory footprint in HBM is strictly O(N) = 4 * B * H * N * d
    hbm_resident_bytes = 4 * batch * heads * seq_len * dim * bytes_per_elem
    flash_intensity = total_flops / max(flash_bytes, 1)

    print(f"\n================================================================================")
    print(f"📊 ATTENTION WORKLOAD: N={seq_len:,} | Batch={batch} | Heads={heads} | d={dim}")
    print(f"================================================================================")
    print(f"Total Theoretical FLOPs: {total_flops / 1e12:.3f} TFLOPs")
    print(f"Standard Attention HBM Transfer : {std_bytes / 1e9:.2f} GB | Intensity: {std_intensity:.1f} FLOPs/Byte")
    print(f"FlashAttention-2 HBM Transfer   : {flash_bytes / 1e9:.2f} GB | Intensity: {flash_intensity:.1f} FLOPs/Byte")
    print(f"Global HBM Memory Footprint     : Standard: {std_bytes / 1e9:.2f} GB vs FA-2: {hbm_resident_bytes / 1e6:.2f} MB ({std_bytes / max(hbm_resident_bytes, 1):.0f}x OOM Reduction!)")

    for gpu_name, spec in GPU_SPECS.items():
        ridge_point = (spec["peak_bf16_tflops"] * 1e12) / (spec["peak_hbm_tb_s"] * 1e12)
        std_speed_limit = min(spec["peak_bf16_tflops"], (spec["peak_hbm_tb_s"] * 1e12 * std_intensity) / 1e12)
        flash_speed_limit = min(spec["peak_bf16_tflops"], (spec["peak_hbm_tb_s"] * 1e12 * flash_intensity) / 1e12)
        std_regime = "COMPUTE-BOUND" if std_intensity >= ridge_point else "MEMORY-BOUND"
        flash_regime = "COMPUTE-BOUND" if flash_intensity >= ridge_point else "MEMORY-BOUND"

        print(f"\n  🖥️  {gpu_name}:")
        print(f"     Ridge Point: {ridge_point:.1f} FLOPs/Byte")
        print(f"     Standard Attention: {std_speed_limit:.1f} TFLOPs/s ({std_regime})")
        print(f"     FlashAttention-2  : {flash_speed_limit:.1f} TFLOPs/s ({flash_regime}) -> {flash_speed_limit / max(std_speed_limit, 1):.2f}x Speedup")

if __name__ == "__main__":
    for n in [1024, 4096, 16384, 32768, 65536]:
        analyze_attention_workload(seq_len=n)
