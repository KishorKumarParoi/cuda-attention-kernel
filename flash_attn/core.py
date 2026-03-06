"""
FlashAttention-2 Python Core Engine.
Provides custom PyTorch Autograd Function implementing Tiled Online Softmax in O(N) memory.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F


class FlashAttentionV2Function(torch.autograd.Function):
    """
    FlashAttention-2 custom Autograd Function.
    Executes Tiled Online Softmax to bypass O(N^2) HBM memory traffic.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = q.shape
        scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(head_dim))

        # Tiling block sizes
        Br = 32
        Bc = 32

        O = torch.zeros_like(q)
        # Logsumexp storage for backward pass: [B, H, N]
        L = torch.empty((batch_size, num_heads, seq_len), device=q.device, dtype=q.dtype)

        num_q_blocks = (seq_len + Br - 1) // Br
        num_k_blocks = (seq_len + Bc - 1) // Bc

        # Outer loop over Query blocks, inner loop over KV blocks (Tiled Online Softmax)
        for i in range(num_q_blocks):
            q_start = i * Br
            q_end = min(q_start + Br, seq_len)
            q_block = q[:, :, q_start:q_end, :]  # [B, H, Br, d]

            m_i = torch.full((batch_size, num_heads, q_end - q_start, 1), float("-inf"), device=q.device)
            l_i = torch.zeros((batch_size, num_heads, q_end - q_start, 1), device=q.device)
            acc_O = torch.zeros((batch_size, num_heads, q_end - q_start, head_dim), device=q.device)

            for j in range(num_k_blocks):
                k_start = j * Bc
                k_end = min(k_start + Bc, seq_len)

                if is_causal and k_start > q_end - 1:
                    break

                k_block = k[:, :, k_start:k_end, :]  # [B, H, Bc, d]
                v_block = v[:, :, k_start:k_end, :]  # [B, H, Bc, d]

                # S_ij = (Q_i * K_j^T) * scale
                S_ij = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

                if is_causal:
                    # Apply causal mask within block
                    q_indices = torch.arange(q_start, q_end, device=q.device).unsqueeze(1)
                    k_indices = torch.arange(k_start, k_end, device=q.device).unsqueeze(0)
                    causal_mask = q_indices < k_indices
                    S_ij.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

                # Online Softmax update
                m_block, _ = torch.max(S_ij, dim=-1, keepdim=True)
                m_new = torch.maximum(m_i, m_block)
                alpha = torch.exp(m_i - m_new)

                P_ij = torch.exp(S_ij - m_new)
                P_ij = torch.nan_to_num(P_ij, nan=0.0)

                l_new = l_i * alpha + torch.sum(P_ij, dim=-1, keepdim=True)

                # Accumulate O: O = O * alpha + P_ij * V_j
                acc_O = acc_O * alpha + torch.matmul(P_ij, v_block)

                m_i = m_new
                l_i = l_new

            # Normalize output
            inv_l = torch.where(l_i > 0, 1.0 / l_i, torch.zeros_like(l_i))
            O[:, :, q_start:q_end, :] = acc_O * inv_l
            L[:, :, q_start:q_end] = (m_i + torch.log(torch.clamp(l_i, min=1e-6))).squeeze(-1)

        ctx.save_for_backward(q, k, v, O, L)
        ctx.scale = scale
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        q, k, v, O, L = ctx.saved_tensors
        scale = ctx.scale
        is_causal = ctx.is_causal

        batch_size, num_heads, seq_len, head_dim = q.shape

        dQ = torch.zeros_like(q)
        dK = torch.zeros_like(k)
        dV = torch.zeros_like(v)

        # D_i = sum_c (dO_ic * O_ic)
        D = torch.sum(dO * O, dim=-1, keepdim=True)  # [B, H, N, 1]

        Br = 32
        Bc = 32
        num_q_blocks = (seq_len + Br - 1) // Br
        num_k_blocks = (seq_len + Bc - 1) // Bc

        for j in range(num_k_blocks):
            k_start = j * Bc
            k_end = min(k_start + Bc, seq_len)
            k_block = k[:, :, k_start:k_end, :]
            v_block = v[:, :, k_start:k_end, :]

            for i in range(num_q_blocks):
                q_start = i * Br
                q_end = min(q_start + Br, seq_len)

                if is_causal and q_end - 1 < k_start:
                    continue

                q_block = q[:, :, q_start:q_end, :]
                do_block = dO[:, :, q_start:q_end, :]
                l_block = L[:, :, q_start:q_end].unsqueeze(-1)
                d_block = D[:, :, q_start:q_end, :]

                # Recompute S_ij and P_ij
                S_ij = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

                if is_causal:
                    q_indices = torch.arange(q_start, q_end, device=q.device).unsqueeze(1)
                    k_indices = torch.arange(k_start, k_end, device=q.device).unsqueeze(0)
                    causal_mask = q_indices < k_indices
                    S_ij.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

                P_ij = torch.exp(S_ij - l_block)
                P_ij = torch.nan_to_num(P_ij, nan=0.0)

                # dV = P^T * dO
                dV[:, :, k_start:k_end, :] += torch.matmul(P_ij.transpose(-2, -1), do_block)

                # dP = dO * V^T
                dP_ij = torch.matmul(do_block, v_block.transpose(-2, -1))

                # dS = P * (dP - D)
                dS_ij = P_ij * (dP_ij - d_block) * scale

                # dQ += dS * K
                dQ[:, :, q_start:q_end, :] += torch.matmul(dS_ij, k_block)

                # dK += dS^T * Q
                dK[:, :, k_start:k_end, :] += torch.matmul(dS_ij.transpose(-2, -1), q_block)

        return dQ, dK, dV, None, None


def flash_attention_2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    High-level FlashAttention-2 interface.
    Args:
        q: [Batch, Heads, SeqLen, HeadDim]
        k: [Batch, Heads, SeqLen, HeadDim]
        v: [Batch, Heads, SeqLen, HeadDim]
        causal: bool, whether to apply lower-triangular causal mask
        softmax_scale: float, scale factor (default: 1 / sqrt(HeadDim))
    Returns:
        O: [Batch, Heads, SeqLen, HeadDim]
    """
    return FlashAttentionV2Function.apply(q, k, v, causal, softmax_scale)

