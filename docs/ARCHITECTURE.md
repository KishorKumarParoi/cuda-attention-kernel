# ⚡ FlashAttention-2 & FlashAttention-3 Microarchitectural Specification

> **Principal GPU Systems & Kernel Architecture Blueprint**  
> **Author:** Kishor Kumar Paroi `<1703053@student.ruet.ac.bd>`  
> **Target Hardware:** NVIDIA Ampere (A100) & Hopper (H100/H200) Architectures  
> **Repository:** [KishorKumarParoi/cuda-attention-kernel](https://github.com/KishorKumarParoi/cuda-attention-kernel)

---

## 🏛️ 1. Executive Architecture Overview

Standard Scaled Dot-Product Attention (Vaswani et al., 2017) computes:
$$\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}} \in \mathbb{R}^{N \times N}, \quad \mathbf{P} = \text{softmax}(\mathbf{S}) \in \mathbb{R}^{N \times N}, \quad \mathbf{O} = \mathbf{P}\mathbf{V} \in \mathbb{R}^{N \times d}$$

In conventional GPU implementations (e.g. PyTorch naive attention), intermediate matrices $\mathbf{S}$ and $\mathbf{P}$ are fully materialized in **High-Bandwidth Memory (HBM)**. For sequence length $N = 16,384$ and batch $B=2, H=16$, storing $\mathbf{P}$ requires:
$$2 \times 16 \times (16384)^2 \times 2 \text{ bytes (FP16)} \approx 17.18 \text{ GB of HBM}$$

This causes:
1. **Memory Bandwidth Bottleneck:** HBM transfers ($\approx 1.5 - 3.35 \text{ TB/s}$) are $10\times - 20\times$ slower than on-chip SRAM bandwidth ($\approx 19 - 33 \text{ TB/s}$).
2. **Catastrophic Out-Of-Memory (OOM):** For long-context models ($32k - 128k$), standard attention fails catastrophically.

FlashAttention reformulates attention through **Tiling** and **Online Softmax**, executing entirely in high-speed SRAM registers with an **$O(N)$ memory footprint** and zero $N \times N$ matrix materialization in global HBM.

---

## 📐 2. GPU Memory Hierarchy & Bandwidth Latency Profile

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  NVIDIA H100 (Hopper SXM5) MEMORY HIERARCHY LATENCIES & BANDWIDTHS             │
│                                                                                │
│  [Warp Registers]         ~0-1 cycle latency    │ ~330 TB/s aggregate bandwidth│
│         ▲                                                                      │
│         │ Register Spill / Warp Shuffles (__shfl_down_sync)                    │
│         ▼                                                                      │
│  [Shared Memory / L1]     ~20-30 cycles latency │ ~33 TB/s aggregate bandwidth │
│  (228 KB per SM)                                                               │
│         ▲                                                                      │
│         │ Hardware TMA / Async Copy (cp.async) / Padded Swizzled Access        │
│         ▼                                                                      │
│  [L2 Cache - 50 MB]       ~200 cycles latency   │ ~12 TB/s aggregate bandwidth │
│         ▲                                                                      │
│         │ HBM3 Memory Controller Bus                                           │
│         ▼                                                                      │
│  [Global Memory - HBM3]   ~400-800 cycles       │ ~3.35 TB/s peak bandwidth    │
│  (80 GB Capacity)                                                              │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧮 3. First-Principles Mathematical Derivation: Tiled Online Softmax

### 3.1 The Three-Pass Standard Softmax Problem
Standard softmax over row vector $x \in \mathbb{R}^N$:
$$m = \max_{j} x_j, \quad l = \sum_{j=1}^N e^{x_j - m}, \quad \text{softmax}(x)_j = \frac{e^{x_j - m}}{l}$$
This requires 3 distinct passes over $x$: Pass 1 for max $m$, Pass 2 for normalizer sum $l$, and Pass 3 for division.

### 3.2 Two-Pass Online Softmax Formulation (Milakov & Gimelshein, 2018)
Divide sequence $x$ into two sub-blocks: $x^{(1)} = [x_1, \dots, x_{B}]$ and $x^{(2)} = [x_{B+1}, \dots, x_{2B}]$.
Define partial running maximums and normalizers:
$$m^{(1)} = \max_k x_k^{(1)}, \quad l^{(1)} = \sum_k e^{x_k^{(1)} - m^{(1)}}$$
Upon ingesting block $x^{(2)}$, compute local statistics $\tilde{m}^{(2)} = \max_k x_k^{(2)}$ and $\tilde{l}^{(2)} = \sum_k e^{x_k^{(2)} - \tilde{m}^{(2)}}$.

The updated global statistics are computed inductively:
$$m^{(2)} = \max(m^{(1)}, \tilde{m}^{(2)})$$
$$l^{(2)} = e^{m^{(1)} - m^{(2)}} \cdot l^{(1)} + e^{\tilde{m}^{(2)} - m^{(2)}} \cdot \tilde{l}^{(2)}$$

### 3.3 Tiled Attention Output Accumulator Rescaling
For output accumulation $\mathbf{O} = \mathbf{P}\mathbf{V}$:
$$\mathbf{O}^{(1)} = \sum_{j \in \text{block 1}} \frac{e^{S_{ij} - m_i^{(1)}}}{l_i^{(1)}} \mathbf{V}_j \implies \mathbf{O}_{\text{unnorm}}^{(1)} = \sum_{j \in \text{block 1}} e^{S_{ij} - m_i^{(1)}} \mathbf{V}_j$$

When transitioning from block 1 to block 2:
$$\mathbf{O}_{\text{unnorm}}^{(2)} = \text{diag}\left(e^{m_i^{(1)} - m_i^{(2)}}\right) \mathbf{O}_{\text{unnorm}}^{(1)} + \sum_{j \in \text{block 2}} e^{S_{ij} - m_i^{(2)}} \mathbf{V}_j$$
At the conclusion of all Key-Value blocks $K_j, V_j$:
$$\mathbf{O}_{\text{final}} = \text{diag}\left(l_i^{(\text{final})}\right)^{-1} \mathbf{O}_{\text{unnorm}}^{(\text{final})}$$

**Key Result:** $\mathbf{S}$ and $\mathbf{P}$ are never written to global memory. Memory required is only that of running accumulators: $O(N)$ instead of $O(N^2)$.

---

## ⚡ 4. IO Complexity & Roofline Model Analysis

Let $N$ be sequence length, $d$ head dimension, and $M$ the fast SRAM size per Streaming Multiprocessor (SM).
- **Standard Attention I/O:** Reading $Q, K$, writing $S$, reading $S$, writing $P$, reading $P, V$, writing $O$:
  $$\text{Total HBM Bytes} = 4Nd + 4N^2 \text{ words} = \Theta(Nd + N^2)$$
- **FlashAttention I/O:**
  $$\text{Total HBM Bytes} = \Theta\left(\frac{N^2 d^2}{M}\right)$$
For $M \approx 100 \text{ KB}$ and $d = 64$ to $128$, $\frac{d^2}{M} \ll 1$, yielding a **$4\times - 20\times$ reduction in global memory traffic**.

### Arithmetic Intensity
$$\text{Arithmetic Intensity}_{\text{Standard}} = \frac{4N^2d \text{ FLOPs}}{4Nd + 4N^2 \text{ Bytes}} \approx \frac{4Nd}{4N} = d \text{ FLOPs/Byte}$$
$$\text{Arithmetic Intensity}_{\text{Flash}} = \frac{4N^2d \text{ FLOPs}}{4Nd + \frac{4N^2d^2}{M} \text{ Bytes}} \approx \frac{M}{d} \text{ FLOPs/Byte} \gg d$$
FlashAttention moves the kernel past the ridge point of the GPU Roofline model into the **compute-bound regime**, unlocking maximum Tensor Core execution efficiency.

---

## 🔬 5. Advanced Microarchitectural Optimizations

### 5.1 Shared Memory Bank Conflicts & 2D Swizzling
NVIDIA Shared Memory is organized into 32 independent banks (each 4 bytes wide). If $k$ threads in a warp access different words belonging to the same bank simultaneously, access is serialized ($k$-way bank conflict).
- **Naive Allocation:** `float s_Q[Br][d];` with $d=64$ causes thread $t$ and thread $t+16$ to access the same bank ($64 \equiv 0 \pmod{32}$).
- **Mitigation 1 (Static Padding):** `float s_Q[Br][d + 4];` shifts rows by 4 words (16 bytes), breaking alignment.
- **Mitigation 2 (Hopper Swizzling):** Bitwise index transformation:
  $$\text{swizzle}(i, j) = j \oplus \left(\left(\frac{i}{8}\right) \pmod 8\right)$$

### 5.2 Warp-Level Shuffle Reductions
Warp threads exchange running statistics in single-cycle register operations via intra-warp instructions:
```cpp
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, mask));
    }
    return __shfl_sync(0xffffffff, val, 0);
}
```

### 5.3 SRAM Recomputation in Backward Pass
Standard backprop requires caching forward activation matrix $\mathbf{P} \in \mathbb{R}^{N \times N}$.
FlashAttention discards $\mathbf{P}$ entirely, saving only the softmax log-sum-exp vector:
$$L_i = m_i + \log(l_i) \in \mathbb{R}^{N}$$
During backward pass, for each block of $d\mathbf{O}$ and $\mathbf{V}$, $\mathbf{P}_{ij}$ is recomputed on-the-fly in SRAM:
$$P_{ij} = \exp\left(S_{ij} - L_i\right)$$
Then the attention gradients are accumulated:
$$d\mathbf{V}_j = \sum_i P_{ij} d\mathbf{O}_i, \quad d\mathbf{S}_{ij} = P_{ij} \left(d\mathbf{O}_i \cdot \mathbf{V}_j - D_i\right) \quad \text{where } D_i = d\mathbf{O}_i \cdot \mathbf{O}_i$$
$$d\mathbf{Q}_i = \sum_j d\mathbf{S}_{ij} \mathbf{K}_j, \quad d\mathbf{K}_j = \sum_i d\mathbf{S}_{ij} \mathbf{Q}_i$$
Memory required for backward pass: **$O(N)$ linear memory**.

---

## 🚀 6. FlashAttention-3 Hopper Next-Gen Primitives

1. **Warp Specialization (Producer-Consumer):**
   - SM threads are partitioned into 1 Producer Warp and 3 Consumer Warps.
   - Producer Warp coordinates hardware TMA transfers from HBM to SMEM asynchronously.
   - Consumer Warps perform Tensor Core WGMMA math without pipeline stalling.
2. **Asynchronous WGMMA (`wgmma.mma_async`):**
   - Executes matrix multiplications directly from shared memory into registers without register intermediary copies.
3. **Hardware FP8 / BF16 Precision:**
   - Supports FP8 (E4M3 and E5M2) with dynamic block scaling factors, doubling arithmetic throughput to **1,978 TFLOPs** on H100 SXM5.

---

## ⚡ 7. Flash-Decoding & Split-KV Autoregressive Attention

### 7.1 The Generation Phase Bottleneck ($S_q = 1$)
In modern autoregressive LLM decoding, Query length is 1 ($S_q = 1$) while the Key/Value cache is long ($S_{kv} \in [2\text{K}, 128\text{K}]$).
In standard FlashAttention, parallelization is over $(B, H, \lceil S_q / B_r \rceil)$.
When $S_q = 1$, there is only **1 query tile per head**. For batch size 1 with 32 attention heads on an NVIDIA A100 (108 SMs) or H100 (132 SMs):
$$\text{SM Occupancy} = \frac{32}{108} \approx 29.6\%$$
Over 70% of the GPU compute capacity remains completely idle!

### 7.2 The Split-KV Architecture
Flash-Decoding decomposes the KV dimension into $K$ independent splits:
1. **Phase 1 (Split-KV Kernel):** The sequence length $N$ is partitioned into $K$ chunks of size $S_{\text{split}} = \lceil N / K \rceil$.
   Each threadblock processes one split $k \in \{0, \dots, K-1\}$ and computes local statistics in SRAM:
   $$m_k = \max_{j \in \text{split } k} (S_{ij}), \quad l_k = \sum_{j \in \text{split } k} \exp(S_{ij} - m_k)$$
   $$\tilde{\mathbf{O}}_k = \sum_{j \in \text{split } k} \exp(S_{ij} - m_k) \mathbf{V}_j$$
   Grid dimensions: $\left(K, H, B\right)$. This increases active threadblocks from $32$ to $32 \times K$, restoring **100% SM occupancy**.

2. **Phase 2 (Logsumexp Reduction Kernel):** A secondary reduction kernel merges the $K$ partial accumulators:
   $$m_{\text{global}} = \max_{k=0}^{K-1} m_k$$
   $$w_k = \exp\left(m_k - m_{\text{global}}\right)$$
   $$l_{\text{global}} = \sum_{k=0}^{K-1} l_k \cdot w_k$$
   $$\mathbf{O}_{\text{final}} = \frac{1}{l_{\text{global}}} \sum_{k=0}^{K-1} \tilde{\mathbf{O}}_k \cdot w_k$$
   This two-stage decomposition yields up to **8x lower latency** during LLM token generation.

---

## 💾 8. PagedAttention & Virtual Block Table KV Cache

### 8.1 The Memory Fragmentation Challenge
In traditional serving systems, contiguous memory is pre-allocated for the maximum sequence length (e.g., 4096 tokens).
This incurs:
- **Internal fragmentation:** Unused reserved memory in active requests.
- **External fragmentation:** Memory unusable due to scattered variable allocations.
Empirical audits reveal that **60% to 80%** of GPU memory is wasted.

### 8.2 Paged Memory Architecture
PagedAttention adapts OS virtual memory paging:
- Physical KV storage is divided into non-contiguous blocks of size $B_{\text{page}}$ (e.g., 16 or 32 tokens).
- Logical sequence tokens are mapped through an indirection table:
  $$\text{BlockTable}: (\text{seq\_id}, \text{logical\_block\_idx}) \to \text{physical\_block\_id}$$
- The CUDA kernel resolves memory indices directly in shared memory registers:
  $$\text{KV\_Offset} = \left(\text{BlockTable}[b, \lfloor t / B_{\text{page}} \rfloor] \times 2 + \text{is\_V}\right) \times H \times B_{\text{page}} \times d + h \times B_{\text{page}} \times d + (t \bmod B_{\text{page}}) \times d$$
- Memory waste drops to **< 4%**, tripling the concurrent serving batch size.

---

## 🪟 9. Sliding Window Attention & Fused ALiBi

### 9.1 Sliding Window Attention (Local Attention)
Restricts attention to a local context window $W$: token $i$ only attends to tokens in $[\max(0, i - W + 1), i]$.
- **Tile-Level Pruning:**
  - If $k_{\text{end}} - 1 < q_{\text{start}} - W$, the entire tile lies outside the receptive field and is skipped.
  - Slashes compute from $O(N^2)$ to $O(N \cdot W)$, enabling 1M+ token context lengths.

### 9.2 Fused ALiBi (Attention with Linear Biases)
Instead of adding an external position embedding matrix in HBM, ALiBi injects a static slope penalty directly into the online softmax accumulator in registers:
$$S_{ij} = \frac{\mathbf{Q}_i \mathbf{K}_j^T}{\sqrt{d}} - m_h \cdot |i - j| \quad \text{where } m_h = 2^{-\frac{8 \cdot h}{H}}$$
Zero additional memory transfers, zero cache misses, and perfect extrapolation beyond training lengths.

---

## 🔬 10. FP8 Quantized Scaled Attention (Hopper / Blackwell)

### 10.1 E4M3 vs E5M2 Representation
- **FP8 E4M3:** 1 sign bit, 4 exponent bits, 3 mantissa bits (range $[-448, 448]$). Maximizes precision for activations and weights in forward passes.
- **FP8 E5M2:** 1 sign bit, 5 exponent bits, 2 mantissa bits (range $[-57344, 57344]$). Maximizes dynamic range for gradients in backward passes.

### 10.2 Scaled Matrix Multiplication in Shared Memory
Dynamic per-tensor scaling factors prevent numerical underflow:
$$\mathbf{S}_{ij} = \left(\mathbf{Q}_{\text{fp8}} \mathbf{K}_{\text{fp8}}^T\right) \cdot \left(\text{scale}_q \cdot \text{scale}_k \cdot \text{scale}_{\text{softmax}}\right)$$
Products are accumulated into FP32 registers, providing **2x Tensor Core FLOP/s** throughput while maintaining high SNR (>30 dB).


