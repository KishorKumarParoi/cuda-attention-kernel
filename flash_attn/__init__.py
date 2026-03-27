"""
CUDA Attention Kernel — High-Performance FlashAttention-2 Implementation.
"""

__version__ = "1.0.0"
__author__ = "Kishor Kumar Paroi <1703053@student.ruet.ac.bd>"

from flash_attn.core import flash_attention_2, FlashAttentionV2Function

__all__ = [
    "flash_attention_2",
    "FlashAttentionV2Function",
]


# Commit 1: math(attention): derive online softmax running maximum and normalizer recurrences

# Commit 2: feat(kernel): scaffold basic CUDA attention kernel with naive global memory access

# Commit 3: arch(memory): analyze SRAM vs HBM memory hierarchy traffic bottlenecks

# Commit 4: feat(tiling): implement Br x Bc shared memory block tiling for query, key, and value

# Commit 5: perf(cuda): pad shared memory allocations [d + 4] to eliminate 32-way bank conflicts

# Commit 6: feat(warp): implement __shfl_down_sync intra-warp sum and max reduction intrinsics

# Commit 7: math(online_softmax): derive rescale factor alpha = exp(m_prev - m_new) for running accumulator

# Commit 8: feat(kernel): fuse scale factor into QK dot product to maximize register occupancy

# Commit 9: feat(causal): implement causal block skipping to prune upper-triangular computations

# Commit 10: feat(cpu_ref): implement high-performance CPU reference kernel with identical tiling

# Commit 11: test(forward): add unit tests verifying bitwise numerical equivalence against PyTorch SDPA

# Commit 12: feat(bwd_math): derive backward gradients dQ, dK, dV without saving attention matrix P

# Commit 13: feat(bwd_kernel): implement SRAM recomputation of attention weights in backward pass

# Commit 14: feat(bwd_kernel): implement atomicAdd accumulation for dQ in shared memory tiles

# Commit 15: test(backward): verify backward gradient precision against PyTorch autograd

# Commit 16: feat(binding): create pybind11 C++ and CUDA extension bindings for PyTorch

# Commit 17: feat(autograd): implement FlashAttentionV2Function with custom forward and backward passes

# Commit 18: feat(python): expose clean high-level flash_attention_2 Python API

# Commit 19: test(causal): add unit tests verifying causal masking across varying sequence lengths

# Commit 20: test(memory): add automated test asserting O(N) linear memory footprint scaling
