"""
PagedAttention & Physical KV-Cache Memory Management.
=====================================================
Models vLLM-style virtual memory paging for Key-Value caches in LLM serving.

Key Innovations:
1. PagedKVCache: Dynamic allocator that partitions GPU HBM into fixed-size physical
   blocks (e.g. block_size=16 or 32 tokens), eliminating internal & external fragmentation.
2. Block Table Mapping: Maps sequence logical token positions to non-contiguous physical
   block IDs: logical_token_idx -> (block_number, block_offset).
3. Paged FlashAttention Kernel: Performs tiled online softmax reading directly from
   non-contiguous physical block memory via indirection table without copying or defragmenting.

Author: Kishor Kumar Paroi <1703053@student.ruet.ac.bd>
"""

import math
from typing import List, Optional, Tuple, Dict
import torch


class PagedKVCacheManager:
    """
    Manages non-contiguous physical memory blocks for Key and Value tensors.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Physical storage: [num_blocks, 2, num_heads, block_size, head_dim]
        # Dim 1: 0 for Key, 1 for Value
        self.kv_cache = torch.zeros(
            (num_blocks, 2, num_heads, block_size, head_dim),
            device=device,
            dtype=dtype,
        )

        # Free block pool
        self.free_blocks: List[int] = list(range(num_blocks))
        # Sequence allocation map: seq_id -> list of physical block indices
        self.seq_block_map: Dict[int, List[int]] = {}

    def allocate_sequence(self, seq_id: int, initial_len: int = 0) -> List[int]:
        """Allocate initial physical blocks for a new sequence."""
        num_needed = max(1, (initial_len + self.block_size - 1) // self.block_size)
        if len(self.free_blocks) < num_needed:
            raise RuntimeError(
                f"Out of physical KV blocks! Needed {num_needed}, available {len(self.free_blocks)}"
            )

        allocated = [self.free_blocks.pop() for _ in range(num_needed)]
        self.seq_block_map[seq_id] = allocated
        return allocated

    def append_token_kv(
        self,
        seq_id: int,
        token_pos: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """
        Append a single token's K and V tensors to the sequence's allocated paged cache.
        Args:
            seq_id: Sequence identifier
            token_pos: 0-indexed position of this token in sequence
            k: [num_heads, head_dim]
            v: [num_heads, head_dim]
        """
        logical_block_idx = token_pos // self.block_size
        block_offset = token_pos % self.block_size

        blocks = self.seq_block_map.get(seq_id)
        if blocks is None:
            raise KeyError(f"Sequence {seq_id} not initialized in cache.")

        # If token exceeds currently allocated blocks, allocate a new physical block
        if logical_block_idx >= len(blocks):
            if not self.free_blocks:
                raise RuntimeError("KV Cache full: cannot allocate new physical block.")
            new_block = self.free_blocks.pop()
            blocks.append(new_block)

        physical_block_id = blocks[logical_block_idx]

        self.kv_cache[physical_block_id, 0, :, block_offset, :] = k
        self.kv_cache[physical_block_id, 1, :, block_offset, :] = v

    def free_sequence(self, seq_id: int) -> None:
        """Release allocated physical blocks back to the free pool."""
        if seq_id in self.seq_block_map:
            freed = self.seq_block_map.pop(seq_id)
            self.free_blocks.extend(freed)

    def get_block_table(self, seq_id: int) -> List[int]:
        """Return the physical block indices allocated to seq_id."""
        return self.seq_block_map.get(seq_id, [])

    def get_memory_utilization(self) -> float:
        """Return ratio of allocated blocks to total physical blocks."""
        used = self.num_blocks - len(self.free_blocks)
        return used / self.num_blocks


def paged_flash_attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int = 16,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    PagedAttention execution kernel over non-contiguous physical blocks.

    Args:
        q: Query tensor [batch_size, num_heads, 1, head_dim] (decoding token)
        kv_cache: Physical pool [num_blocks, 2, num_heads, block_size, head_dim]
        block_tables: [batch_size, max_blocks_per_seq] int tensor of physical block IDs
        context_lens: [batch_size] int tensor containing total past tokens per sequence
        block_size: Number of tokens per physical block (default: 16)
        softmax_scale: Optional scale factor (default: 1 / sqrt(head_dim))

    Returns:
        O: Output tensor [batch_size, num_heads, 1, head_dim]
    """
    batch_size, num_heads, q_len, head_dim = q.shape
    assert q_len == 1, "PagedAttention typically operates during autoregressive decode (q_len=1)"

    scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(head_dim))
    O = torch.zeros_like(q)

    # Process each batch item independently
    for b in range(batch_size):
        total_len = int(context_lens[b].item())
        if total_len == 0:
            continue

        num_blocks_for_seq = (total_len + block_size - 1) // block_size
        q_b = q[b : b + 1]  # [1, H, 1, d]

        m_i = torch.full((1, num_heads, 1, 1), float("-inf"), device=q.device, dtype=q.dtype)
        l_i = torch.zeros((1, num_heads, 1, 1), device=q.device, dtype=q.dtype)
        acc_O = torch.zeros((1, num_heads, 1, head_dim), device=q.device, dtype=q.dtype)

        for l_idx in range(num_blocks_for_seq):
            phys_block_id = int(block_tables[b, l_idx].item())

            # Tokens in this physical block
            tokens_in_block = min(block_size, total_len - l_idx * block_size)

            # Key block: [1, H, tokens_in_block, d]
            k_block = kv_cache[phys_block_id, 0, :, :tokens_in_block, :].unsqueeze(0)
            # Value block: [1, H, tokens_in_block, d]
            v_block = kv_cache[phys_block_id, 1, :, :tokens_in_block, :].unsqueeze(0)

            # S_ij = (Q * K^T) * scale: [1, H, 1, tokens_in_block]
            S_ij = torch.matmul(q_b, k_block.transpose(-2, -1)) * scale

            # Online Softmax updates
            m_block, _ = torch.max(S_ij, dim=-1, keepdim=True)
            m_new = torch.maximum(m_i, m_block)
            alpha = torch.exp(m_i - m_new)
            alpha = torch.nan_to_num(alpha, nan=0.0)

            P_ij = torch.exp(S_ij - m_new)
            P_ij = torch.nan_to_num(P_ij, nan=0.0)

            l_new = l_i * alpha + torch.sum(P_ij, dim=-1, keepdim=True)
            acc_O = acc_O * alpha + torch.matmul(P_ij, v_block)

            m_i = m_new
            l_i = l_new

        inv_l = torch.where(l_i > 0, 1.0 / l_i, torch.zeros_like(l_i))
        O[b : b + 1] = acc_O * inv_l

    return O

