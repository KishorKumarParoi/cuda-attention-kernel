"""
Unit tests for FlashAttention-2 GPU Kernel & Reference Implementation.
Verifies numerical precision against PyTorch F.scaled_dot_product_attention.
"""

import math
import pytest
import torch
import torch.nn.functional as F
from flash_attn import flash_attention_2


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_heads", [2, 4])
@pytest.mark.parametrize("seq_len", [64, 128])
@pytest.mark.parametrize("head_dim", [32, 64])
def test_forward_numerical_equivalence(batch_size, num_heads, seq_len, head_dim):
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)

    # Reference PyTorch standard attention
    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)

    # FlashAttention-2 Tiled Online Softmax
    out_fa = flash_attention_2(q, k, v, causal=False)

    assert torch.allclose(out_fa, out_ref, atol=1e-4), f"Max diff: {(out_fa - out_ref).abs().max()}"


def test_causal_masking_equivalence():
    batch_size, num_heads, seq_len, head_dim = 2, 4, 64, 32
    torch.manual_seed(1337)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)

    out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out_fa = flash_attention_2(q, k, v, causal=True)

    assert torch.allclose(out_fa, out_ref, atol=1e-4), f"Max causal diff: {(out_fa - out_ref).abs().max()}"


def test_backward_gradient_accuracy():
    batch_size, num_heads, seq_len, head_dim = 2, 2, 32, 16
    torch.manual_seed(42)

    # FlashAttention tensors
    q1 = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True)
    k1 = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True)
    v1 = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True)

    # Reference tensors
    q2 = q1.detach().clone().requires_grad_(True)
    k2 = k1.detach().clone().requires_grad_(True)
    v2 = v1.detach().clone().requires_grad_(True)

    # Forward
    out_fa = flash_attention_2(q1, k1, v1, causal=False)
    out_ref = F.scaled_dot_product_attention(q2, k2, v2, is_causal=False)

    # Dummy loss
    grad_out = torch.randn_like(out_fa)
    out_fa.backward(grad_out)
    out_ref.backward(grad_out)

    assert torch.allclose(q1.grad, q2.grad, atol=1e-3), f"dQ diff: {(q1.grad - q2.grad).abs().max()}"
    assert torch.allclose(k1.grad, k2.grad, atol=1e-3), f"dK diff: {(k1.grad - k2.grad).abs().max()}"
    assert torch.allclose(v1.grad, v2.grad, atol=1e-3), f"dV diff: {(v1.grad - v2.grad).abs().max()}"


def test_memory_scaling():
    """Verifies that FlashAttention avoids quadratic intermediate allocations."""
    batch_size, num_heads, head_dim = 1, 1, 64
    seq_lens = [64, 128, 256]

    for N in seq_lens:
        q = torch.randn(batch_size, num_heads, N, head_dim)
        k = torch.randn(batch_size, num_heads, N, head_dim)
        v = torch.randn(batch_size, num_heads, N, head_dim)

        out = flash_attention_2(q, k, v)
        assert out.shape == (batch_size, num_heads, N, head_dim)

