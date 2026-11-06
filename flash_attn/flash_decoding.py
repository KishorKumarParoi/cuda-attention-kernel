"""
Flash-Decoding (Split-KV) Attention Implementation.
===================================================
Optimized for LLM Generation / Decoding Phase where Query length S_q = 1 (or small chunk)
and Key/Value length S_kv is very large (e.g., 2K, 8K, 32K, 128K).

In standard FlashAttention, parallelization is across (Batch, Heads, ceil(S_q / Br)).
When S_q = 1, there is only 1 query tile per head, which leaves the majority of GPU
Streaming Multiprocessors (SMs) completely idle.

Flash-Decoding solves this by splitting the KV dimension across threadblocks (Split-KV):
1. Phase 1 (Split-KV): The KV sequence is partitioned into K chunks (splits).
   Each threadblock computes partial online softmax:
       m_k = max_{j in split k} (S_ij)
       l_k = sum_{j in split k} exp(S_ij - m_k)
       O_tilde_k = sum_{j in split k} exp(S_ij - m_k) * V_j
2. Phase 2 (Logsumexp Reduction): A lightweight reduction kernel merges the K partial
   accumulators using global logsumexp:
       m_global = max_k (m_k)
       w_k = exp(m_k - m_global)
       l_global = sum_k (l_k * w_k)
       O_final = (sum_k O_tilde_k * w_k) / l_global

Author: Kishor Kumar Paroi <1703053@student.ruet.ac.bd>
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def flash_decoding(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_splits: Optional[int] = None,
    is_causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Flash-Decoding Split-KV Attention.

    Args:
        q: Query tensor of shape [batch_size, num_heads, q_len, head_dim] (typically q_len=1)
        k: Key tensor of shape [batch_size, num_heads, kv_len, head_dim]
        v: Value tensor of shape [batch_size, num_heads, kv_len, head_dim]
        num_splits: Number of KV splits along sequence dimension (auto-computed if None)
        is_causal: Whether causal masking applies
        softmax_scale: Scale factor (default: 1 / sqrt(head_dim))

    Returns:
        O: Output tensor of shape [batch_size, num_heads, q_len, head_dim]
    """
    batch_size, num_heads, q_len, head_dim = q.shape
    _, _, kv_len, _ = k.shape

    scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(head_dim))

    # For prefill or very short KV, standard attention or 1 split is optimal
    if kv_len <= 128:
        num_splits = 1
    elif num_splits is None:
        # Auto-heuristic: distribute work across typical GPU SM counts (e.g. 64-128)
        # Aim for chunk sizes of ~128-256 tokens per split
        target_chunk_size = 128
        estimated_splits = (kv_len + target_chunk_size - 1) // target_chunk_size
        num_splits = max(1, min(64, estimated_splits))

    # Fast path: Single split degenerates to standard tiled online softmax
    if num_splits == 1:
        return _tiled_attention_fwd(q, k, v, scale, is_causal)

    # -------------------------------------------------------------
    # PHASE 1: Split-KV Partial Online Softmax Accumulation
    # -------------------------------------------------------------
    # Each split handles a contiguous slice of KV of size split_size
    split_size = (kv_len + num_splits - 1) // num_splits

    # Partial outputs buffer: [num_splits, batch_size, num_heads, q_len, head_dim]
    partial_O = torch.zeros(
        (num_splits, batch_size, num_heads, q_len, head_dim),
        device=q.device,
        dtype=q.dtype,
    )
    # Partial running max: [num_splits, batch_size, num_heads, q_len, 1]
    partial_m = torch.full(
        (num_splits, batch_size, num_heads, q_len, 1),
        float("-inf"),
        device=q.device,
        dtype=q.dtype,
    )
    # Partial running sum: [num_splits, batch_size, num_heads, q_len, 1]
    partial_l = torch.zeros(
        (num_splits, batch_size, num_heads, q_len, 1),
        device=q.device,
        dtype=q.dtype,
    )

    Bc = 32  # Inner tile size within each split

    for s in range(num_splits):
        kv_start = s * split_size
        kv_end = min(kv_start + split_size, kv_len)

        if kv_start >= kv_len:
            continue

        k_split = k[:, :, kv_start:kv_end, :]
        v_split = v[:, :, kv_start:kv_end, :]
        curr_split_len = kv_end - kv_start

        m_i = torch.full((batch_size, num_heads, q_len, 1), float("-inf"), device=q.device, dtype=q.dtype)
        l_i = torch.zeros((batch_size, num_heads, q_len, 1), device=q.device, dtype=q.dtype)
        acc_O = torch.zeros((batch_size, num_heads, q_len, head_dim), device=q.device, dtype=q.dtype)

        num_inner_tiles = (curr_split_len + Bc - 1) // Bc

        for tile_idx in range(num_inner_tiles):
            t_start = tile_idx * Bc
            t_end = min(t_start + Bc, curr_split_len)
            global_k_start = kv_start + t_start
            global_k_end = kv_start + t_end

            k_tile = k_split[:, :, t_start:t_end, :]
            v_tile = v_split[:, :, t_start:t_end, :]

            # S_ij = (Q * K_tile^T) * scale
            S_ij = torch.matmul(q, k_tile.transpose(-2, -1)) * scale

            if is_causal:
                # Token at position (kv_len - q_len + q_idx) attends to k_idx
                # For decoding with kv_len and q_len=1, the new token position is kv_len - 1
                q_positions = torch.arange(kv_len - q_len, kv_len, device=q.device).unsqueeze(1)
                k_positions = torch.arange(global_k_start, global_k_end, device=q.device).unsqueeze(0)
                mask = q_positions < k_positions
                S_ij.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            m_tile, _ = torch.max(S_ij, dim=-1, keepdim=True)
            m_new = torch.maximum(m_i, m_tile)
            alpha = torch.exp(m_i - m_new)
            alpha = torch.nan_to_num(alpha, nan=0.0)

            P_ij = torch.exp(S_ij - m_new)
            P_ij = torch.nan_to_num(P_ij, nan=0.0)

            l_new = l_i * alpha + torch.sum(P_ij, dim=-1, keepdim=True)
            acc_O = acc_O * alpha + torch.matmul(P_ij, v_tile)

            m_i = m_new
            l_i = l_new

        partial_O[s] = acc_O
        partial_m[s] = m_i
        partial_l[s] = l_i

    # -------------------------------------------------------------
    # PHASE 2: Logsumexp Reduction Across Splits
    # -------------------------------------------------------------
    # Global maximum across all K splits
    # partial_m: [num_splits, B, H, q_len, 1]
    m_global, _ = torch.max(partial_m, dim=0, keepdim=True)  # [1, B, H, q_len, 1]

    # Scaling factor for each split: w_s = exp(m_s - m_global)
    rescale_weights = torch.exp(partial_m - m_global)
    rescale_weights = torch.nan_to_num(rescale_weights, nan=0.0)  # [num_splits, B, H, q_len, 1]

    # Global normalizer: sum_s (l_s * w_s)
    rescaled_l = partial_l * rescale_weights  # [num_splits, B, H, q_len, 1]
    l_global = torch.sum(rescaled_l, dim=0)   # [B, H, q_len, 1]

    # Rescaled accumulated output: sum_s (O_s * w_s)
    rescaled_O = partial_O * rescale_weights  # [num_splits, B, H, q_len, d]
    sum_O = torch.sum(rescaled_O, dim=0)      # [B, H, q_len, d]

    # Final normalization
    inv_l = torch.where(l_global > 0, 1.0 / l_global, torch.zeros_like(l_global))
    O_final = sum_O * inv_l

    return O_final


def _tiled_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    is_causal: bool,
) -> torch.Tensor:
    """Tiled online softmax single-split fallback."""
    batch_size, num_heads, q_len, head_dim = q.shape
    _, _, kv_len, _ = k.shape

    Bc = 32
    num_k_blocks = (kv_len + Bc - 1) // Bc

    m_i = torch.full((batch_size, num_heads, q_len, 1), float("-inf"), device=q.device, dtype=q.dtype)
    l_i = torch.zeros((batch_size, num_heads, q_len, 1), device=q.device, dtype=q.dtype)
    acc_O = torch.zeros((batch_size, num_heads, q_len, head_dim), device=q.device, dtype=q.dtype)

    for j in range(num_k_blocks):
        k_start = j * Bc
        k_end = min(k_start + Bc, kv_len)

        k_block = k[:, :, k_start:k_end, :]
        v_block = v[:, :, k_start:k_end, :]

        S_ij = torch.matmul(q, k_block.transpose(-2, -1)) * scale

        if is_causal:
            q_positions = torch.arange(kv_len - q_len, kv_len, device=q.device).unsqueeze(1)
            k_positions = torch.arange(k_start, k_end, device=q.device).unsqueeze(0)
            causal_mask = q_positions < k_positions
            S_ij.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

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
    return acc_O * inv_l
