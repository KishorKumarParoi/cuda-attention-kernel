# ⚡ FlashAttention-2 Custom CUDA Attention Kernel

A bare-metal, high-performance CUDA Attention Kernel engineered from first principles based on **FlashAttention-2** (Tri Dao, 2023). Eliminates the $O(N^2)$ High-Bandwidth Memory (HBM) traffic bottleneck by computing scaled dot-product attention in fast on-chip SRAM via **Tiled Online Softmax**.

```
Author: Kishor Kumar Paroi <1703053@student.ruet.ac.bd>
Production Release: v1.0.0
Hardware Target: NVIDIA Ampere / Hopper / Ada Lovelace / Reference SIMD
Memory Complexity: O(N) linear memory footprint (Zero N x N matrix materialization)
```

---

## 🏛️ GPU Memory Hierarchy & Tiling Architecture

```mermaid
flowchart TD
    subgraph GPU_Global_Memory ["High-Bandwidth Memory (HBM ~1.5 - 3.0 TB/s)"]
        Q_HBM["Query Tensor Q\n[B, H, N, d]"]
        K_HBM["Key Tensor K\n[B, H, N, d]"]
        V_HBM["Value Tensor V\n[B, H, N, d]"]
        O_HBM["Output Tensor O\n[B, H, N, d]"]
    end

    subgraph GPU_Streaming_Multiprocessor ["SM On-Chip SRAM (Shared Memory ~19 TB/s)"]
        s_Q["Tiled Query Block s_Q\n[Br, d + 4] (Padded)"]
        s_K["Tiled Key Block s_K\n[Bc, d + 4] (Padded)"]
        s_V["Tiled Value Block s_V\n[Bc, d + 4] (Padded)"]
        
        subgraph Warp_Registers ["Warp Registers (Zero-Latency)"]
            m_i["Running Max m_i\n__shfl_down_sync"]
            l_i["Normalizer l_i\n__shfl_down_sync"]
            acc_O["Accumulator acc_O\n(Scaled & Rescaled)"]
        end
    end

    Q_HBM -->|Tiled Load Br| s_Q
    K_HBM -->|Tiled Load Bc| s_K
    V_HBM -->|Tiled Load Bc| s_V
    s_Q & s_K -->|S = Q · K^T| m_i
    m_i -->|P = exp(S - m)| l_i
    l_i & s_V -->|O = O · α + P · V| acc_O
    acc_O -->|Write Back (Inv-L)| O_HBM
```

---

## 🧮 Mathematical Foundations: Tiled Online Softmax

Standard multi-head attention computes:
$$S = \frac{Q K^T}{\sqrt{d}} \in \mathbb{R}^{N \times N}, \quad P = \text{softmax}(S) \in \mathbb{R}^{N \times N}, \quad O = P V \in \mathbb{R}^{N \times d}$$

In standard attention, writing $S$ and $P$ to global HBM and reading them back costs $\Theta(N^2)$ memory reads/writes. For long contexts ($N = 8192$ or $32k$), this dominates GPU latency and triggers out-of-memory (OOM) failures.

### The Online Softmax Algorithm
Let $x \in \mathbb{R}^N$ split into two blocks $x^{(1)}$ and $x^{(2)}$.
1. Compute local maximums:
   $$m^{(1)} = \max_k x_k^{(1)}, \quad m^{(2)} = \max_k x_k^{(2)}$$
2. Global maximum:
   $$m = \max(m^{(1)}, m^{(2)})$$
3. Rescale previous partial sum and add current block:
   $$l = e^{m^{(1)} - m} l^{(1)} + \sum_k e^{x_k^{(2)} - m}$$
4. Softmax vector update without saving unnormalized scores:
   $$O = \text{diag}(e^{m^{(1)} - m}) O^{(1)} + P^{(2)} V^{(2)}$$
5. Final normalization:
   $$O_{\text{final}} = \text{diag}(l)^{-1} O$$

---

## ⚡ CUDA Microarchitectural Optimizations

1. **Shared Memory Bank Conflict Avoidance:**
   - Standard 32-bank SRAM architectures trigger bank conflicts when strides align with 32 floats.
   - Padded declaration: `__shared__ float s_Q[Br][d + 4];` ensures zero serialization.
2. **Warp-Level Intra-Register Shuffle Reductions:**
   - Implements `__shfl_xor_sync` and `__shfl_down_sync` across 32 threads in each warp to compute $m_i$ and $l_i$ in register space with zero shared memory latency.
3. **Causal Block Skipping:**
   - In causal language modeling, blocks where $j \cdot B_c > i \cdot B_r + B_r - 1$ are strictly upper-triangular.
   - The kernel conditionally terminates the inner KV loop early, slashing FLOPs by **$50\%$**.
4. **SRAM Recomputation in Backward Pass:**
   - Rather than storing the $N \times N$ attention matrix $P$, the backward kernel recomputes $P_{ij}$ in SRAM using saved vector logsumexp $L_i = m_i + \log(l_i)$, achieving an **$O(N)$ memory backward pass**.

---

## 📊 Benchmark & Memory Scaling Comparison

Memory consumption of attention matrix vs sequence length ($B=2, H=8, d=64$):

| Sequence Length ($N$) | Standard Attention HBM | FlashAttention-2 HBM | Memory Reduction | TFLOPs/s |
| :--- | :--- | :--- | :--- | :--- |
| **256** | 4.0 MB | 1.0 MB | **$4\times$** | 0.024 |
| **512** | 16.0 MB | 2.0 MB | **$8\times$** | 0.025 |
| **1024** | 64.0 MB | 4.0 MB | **$16\times$** | 0.018 |
| **2048** | 256.0 MB | 8.0 MB | **$32\times$** | 0.017 |
| **8192** | 4,096.0 MB (4.1 GB) | 32.0 MB | **$128\times$** | 0.016 |

---

## 🧪 Verification & Execution

### 1. Run Unit Tests (Forward & Backward Parity)
```bash
make test
# or
PYTHONPATH=. ./.venv/bin/pytest tests/ -v
```

### 2. Run Arithmetic Intensity Benchmark
```bash
make bench
# or
PYTHONPATH=. ./.venv/bin/python benchmarks/benchmark_attention.py
```
