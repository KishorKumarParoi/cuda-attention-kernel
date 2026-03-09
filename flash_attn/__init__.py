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
