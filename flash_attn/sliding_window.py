"""
Sliding Window Attention & Fused ALiBi (Attention with Linear Biases).
=====================================================================
Optimized for long-context LLMs (e.g., Mistral, Longformer, MPT).

Key Innovations:
1. Sliding Window Attention (Local Attention): Restricts attention to a local band
   of width W (window_size): token i only attends to tokens in [max(0, i - W + 1), i].
   Tile-Level Pruning: Skips entire tiles where k_end < q_start - W or k_start > q_end,
   slashing compute from O(N^2) to O(N * W).
2. Fused ALiBi: Implements Press et al. linear positional bias fused directly into
   online softmax accumulator without intermediate memory traffic:
       S_ij = (Q_i * K_j^T) * scale - m_h * (i - j)
   where m_h = 2^(-8 * h / H) is the head-specific slope.

Author: Kishor Kumar Paroi <1703053@student.ruet.ac.bd>
"""

import math
from typing import Optional
import torch


def sliding_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int = 512,
    is_causal: bool = True,
    alibi: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Sliding Window Attention with tile-level block pruning and optional ALiBi bias.

    Args:
        q: [Batch, Heads, SeqLen_q, HeadDim]
        k: [Batch, Heads, SeqLen_kv, HeadDim]
        v: [Batch, Heads, SeqLen_kv, HeadDim]
        window_size: Maximum backward lookback distance W
        is_causal: Causal attention (cannot look into future tokens)
        alibi: Whether to fuse ALiBi positional slope biases
        softmax_scale: Optional scale factor (default: 1 / sqrt(HeadDim))

    Returns:
        O: [Batch, Heads, SeqLen_q, HeadDim]
    """
    batch_size, num_heads, seq_len_q, head_dim = q.shape
    _, _, seq_len_kv, _ = k.shape

    scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(head_dim))

    # Precompute ALiBi slopes per head if requested
    # m_h = 2^(-8 * h / H)
    alibi_slopes = None
    if alibi:
        def get_slopes(n: int) -> torch.Tensor:
            def get_slopes_power_of_2(n_heads: int) -> list:
                start = 2 ** (-(2 ** -(math.log2(n_heads) - 3)))
                ratio = start
                return [start * (ratio ** i) for i in range(n_heads)]

            if math.log2(n).is_integer():
                return torch.tensor(get_slopes_power_of_2(n), device=q.device, dtype=q.dtype)
            else:
                closest_pow2 = 2 ** math.floor(math.log2(n))
                base_slopes = get_slopes_power_of_2(closest_pow2)
                extra_slopes = get_slopes_power_of_2(2 * closest_pow2)[0::2][: n - closest_pow2]
                return torch.tensor(base_slopes + extra_slopes, device=q.device, dtype=q.dtype)

        alibi_slopes = get_slopes(num_heads).view(1, num_heads, 1, 1)

    Br = 32
    Bc = 32
    num_q_blocks = (seq_len_q + Br - 1) // Br
    num_k_blocks = (seq_len_kv + Bc - 1) // Bc

    O = torch.zeros_like(q)

    for i in range(num_q_blocks):
        q_start = i * Br
        q_end = min(q_start + Br, seq_len_q)
        q_block = q[:, :, q_start:q_end, :]

        m_i = torch.full((batch_size, num_heads, q_end - q_start, 1), float("-inf"), device=q.device, dtype=q.dtype)
        l_i = torch.zeros((batch_size, num_heads, q_end - q_start, 1), device=q.device, dtype=q.dtype)
        acc_O = torch.zeros((batch_size, num_heads, q_end - q_start, head_dim), device=q.device, dtype=q.dtype)

        for j in range(num_k_blocks):
            k_start = j * Bc
            k_end = min(k_start + Bc, seq_len_kv)

            # --- TILE-LEVEL PRUNING ---
            # 1. Causal pruning: if k_start > q_end - 1, entire tile is strictly in the future
            if is_causal and k_start > q_end - 1:
                break

            # 2. Sliding window pruning: if k_end - 1 < q_start - window_size,
            # entire tile is strictly older than the local window W
            if k_end - 1 < q_start - window_size:
                continue

            k_block = k[:, :, k_start:k_end, :]
            v_block = v[:, :, k_start:k_end, :]

            # S_ij = (Q * K^T) * scale
            S_ij = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

            # Masking within block
            q_idx = torch.arange(q_start, q_end, device=q.device).unsqueeze(1)
            k_idx = torch.arange(k_start, k_end, device=q.device).unsqueeze(0)

            # Causal mask: q_idx < k_idx
            if is_causal:
                causal_mask = q_idx < k_idx
                S_ij.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # Window mask: token distance (q_idx - k_idx) > window_size
            window_mask = (q_idx - k_idx) >= window_size
            S_ij.masked_fill_(window_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # Fused ALiBi bias
            if alibi and alibi_slopes is not None:
                # distance = max(0, q_idx - k_idx)
                dist = torch.clamp(q_idx - k_idx, min=0).float()
                alibi_bias = -alibi_slopes * dist.unsqueeze(0).unsqueeze(0)
                S_ij = S_ij + alibi_bias

            # Online Softmax update
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
        O[:, :, q_start:q_end, :] = acc_O * inv_l

    return O
