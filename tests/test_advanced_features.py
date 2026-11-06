"""
Unit Tests for Advanced CUDA Attention Features:
- Flash-Decoding (Split-KV Attention)
- PagedAttention (Non-contiguous Block Tables)
- Sliding Window Attention & ALiBi
- FP8 Scaled Attention Simulation
"""

import math
import pytest
import torch
import torch.nn.functional as F

from flash_attn import (
    flash_decoding,
    PagedKVCacheManager,
    paged_flash_attention,
    sliding_window_attention,
    fp8_scaled_attention,
    quantize_to_fp8_simulated,
)


@pytest.mark.parametrize("kv_len", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize("head_dim", [32, 64])
def test_flash_decoding_equivalence(kv_len, head_dim):
    torch.manual_seed(42)
    batch_size = 2
    num_heads = 4
    q_len = 1  # Standard generation decode

    q = torch.randn(batch_size, num_heads, q_len, head_dim)
    k = torch.randn(batch_size, num_heads, kv_len, head_dim)
    v = torch.randn(batch_size, num_heads, kv_len, head_dim)

    # Reference SDPA
    ref_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

    # Flash-Decoding
    out = flash_decoding(q, k, v, is_causal=False)

    assert torch.allclose(out, ref_out, atol=1e-4, rtol=1e-4), (
        f"Max diff: {torch.max(torch.abs(out - ref_out))}"
    )


@pytest.mark.parametrize("num_splits", [1, 2, 4, 8])
def test_flash_decoding_split_invariance(num_splits):
    """Verify that changing num_splits yields mathematically identical results."""
    torch.manual_seed(123)
    batch_size = 1
    num_heads = 2
    q_len = 1
    kv_len = 512
    head_dim = 64

    q = torch.randn(batch_size, num_heads, q_len, head_dim)
    k = torch.randn(batch_size, num_heads, kv_len, head_dim)
    v = torch.randn(batch_size, num_heads, kv_len, head_dim)

    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    out = flash_decoding(q, k, v, num_splits=num_splits, is_causal=False)

    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_flash_decoding_causal():
    torch.manual_seed(999)
    batch_size = 2
    num_heads = 2
    q_len = 1
    kv_len = 256
    head_dim = 32

    q = torch.randn(batch_size, num_heads, q_len, head_dim)
    k = torch.randn(batch_size, num_heads, kv_len, head_dim)
    v = torch.randn(batch_size, num_heads, kv_len, head_dim)

    # When q_len = 1 and represents token at index kv_len - 1, causal attention
    # allows attending to all previous tokens [0 .. kv_len - 1]
    out = flash_decoding(q, k, v, is_causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)

    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_paged_kv_cache_manager():
    num_blocks = 32
    block_size = 16
    num_heads = 4
    head_dim = 64
    device = torch.device("cpu")

    manager = PagedKVCacheManager(
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        device=device,
    )

    # Allocate sequence 0 with 30 tokens -> needs 2 blocks
    blocks = manager.allocate_sequence(seq_id=0, initial_len=30)
    assert len(blocks) == 2
    assert manager.get_block_table(seq_id=0) == blocks
    assert manager.get_memory_utilization() == 2 / 32

    # Append 30 tokens
    for pos in range(30):
        k_tok = torch.randn(num_heads, head_dim)
        v_tok = torch.randn(num_heads, head_dim)
        manager.append_token_kv(seq_id=0, token_pos=pos, k=k_tok, v=v_tok)

    # Free sequence
    manager.free_sequence(seq_id=0)
    assert manager.get_memory_utilization() == 0.0
    assert len(manager.free_blocks) == 32


def test_paged_flash_attention_correctness():
    torch.manual_seed(777)
    batch_size = 2
    num_heads = 2
    head_dim = 32
    block_size = 16
    device = torch.device("cpu")

    manager = PagedKVCacheManager(
        num_blocks=16,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        device=device,
    )

    # Seq 0: 20 tokens (2 blocks)
    # Seq 1: 35 tokens (3 blocks)
    context_lens_list = [20, 35]
    for seq_id, c_len in enumerate(context_lens_list):
        manager.allocate_sequence(seq_id, initial_len=c_len)
        for pos in range(c_len):
            k_tok = torch.randn(num_heads, head_dim)
            v_tok = torch.randn(num_heads, head_dim)
            manager.append_token_kv(seq_id, token_pos=pos, k=k_tok, v=v_tok)

    # Build block_tables tensor
    max_blocks = 3
    block_tables = torch.zeros((batch_size, max_blocks), dtype=torch.long)
    for b in range(batch_size):
        b_list = manager.get_block_table(b)
        block_tables[b, : len(b_list)] = torch.tensor(b_list, dtype=torch.long)

    context_lens = torch.tensor(context_lens_list, dtype=torch.long)
    q = torch.randn(batch_size, num_heads, 1, head_dim)

    # Execute PagedAttention
    out = paged_flash_attention(
        q=q,
        kv_cache=manager.kv_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        block_size=block_size,
    )

    # Verify against manual contiguous reconstruction
    for b in range(batch_size):
        c_len = context_lens_list[b]
        b_list = manager.get_block_table(b)
        recon_k = []
        recon_v = []
        for l_idx, p_id in enumerate(b_list):
            toks = min(block_size, c_len - l_idx * block_size)
            recon_k.append(manager.kv_cache[p_id, 0, :, :toks, :])
            recon_v.append(manager.kv_cache[p_id, 1, :, :toks, :])
        full_k = torch.cat(recon_k, dim=1).unsqueeze(0)  # [1, H, c_len, d]
        full_v = torch.cat(recon_v, dim=1).unsqueeze(0)  # [1, H, c_len, d]

        q_b = q[b : b + 1]
        ref_b = F.scaled_dot_product_attention(q_b, full_k, full_v)
        assert torch.allclose(out[b : b + 1], ref_b, atol=1e-4, rtol=1e-4)


def test_sliding_window_attention():
    torch.manual_seed(44)
    batch_size = 1
    num_heads = 2
    seq_len = 128
    head_dim = 32
    window_size = 32

    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Run sliding window attention
    out = sliding_window_attention(q, k, v, window_size=window_size, is_causal=True)

    # Reference with explicit band matrix
    scale = 1.0 / math.sqrt(head_dim)
    S = torch.matmul(q, k.transpose(-2, -1)) * scale
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    S.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    window_mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=-window_size).bool()
    S.masked_fill_(window_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    ref_P = torch.softmax(S, dim=-1)
    ref_O = torch.matmul(ref_P, v)

    assert torch.allclose(out, ref_O, atol=1e-4, rtol=1e-4)


def test_sliding_window_with_alibi():
    torch.manual_seed(55)
    batch_size = 1
    num_heads = 4
    seq_len = 64
    head_dim = 32

    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)

    out = sliding_window_attention(q, k, v, window_size=32, is_causal=True, alibi=True)
    assert out.shape == q.shape
    assert not torch.isnan(out).any()


def test_fp8_scaled_attention():
    torch.manual_seed(888)
    batch_size = 2
    num_heads = 2
    seq_len = 64
    head_dim = 32

    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)

    out, metrics = fp8_scaled_attention(q, k, v, is_causal=True)

    assert out.shape == q.shape
    assert not torch.isnan(out).any()
    assert metrics["snr_db"] > 20.0, f"SNR too low: {metrics['snr_db']} dB"
    assert metrics["max_abs_error"] < 0.2
