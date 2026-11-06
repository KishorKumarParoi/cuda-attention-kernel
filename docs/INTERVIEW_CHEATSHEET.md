# 🎯 Principal GPU Kernel & CUDA Attention Interview Mastery Guide

> **Elite Technical Interview Preparation ($100k+ to $350k+ Compensation Tier)**  
> **Target Roles:** Staff/Principal AI Systems Engineer, GPU Kernel Engineer, Foundation Model Infrastructure Architect  
> **Companies:** NVIDIA, OpenAI, Anthropic, Meta Fair, DeepSeek, Cerebras, Modal Labs  
> **Author:** Kishor Kumar Paroi `<1703053@student.ruet.ac.bd>`

---

## ⚡ Section 1: Memory Hierarchy, I/O Complexity & The Roofline Model

### Q1: What is the fundamental memory bottleneck in standard Multi-Head Attention, and what is the exact byte footprint for a modern context window?
**Answer:**
Standard Scaled Dot-Product Attention calculates:
$$\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}}, \quad \mathbf{P} = \text{softmax}(\mathbf{S}), \quad \mathbf{O} = \mathbf{P}\mathbf{V}$$
The bottleneck is **not computation** (FLOPs), but **memory traffic** to High-Bandwidth Memory (HBM).
For batch size $B$, number of heads $H$, sequence length $N$, and head dimension $d$:
- $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ each take $B \times H \times N \times d$ elements.
- The intermediate score matrix $\mathbf{S}$ and probability matrix $\mathbf{P}$ take $B \times H \times N^2$ elements.

**Exact Calculation for $B=2, H=32, N=32,768, d=128$ in FP16 (2 bytes/elem):**
$$\text{Memory of } \mathbf{P} = 2 \times 32 \times (32768)^2 \times 2 \text{ bytes} = 64 \times 1,073,741,824 \times 2 \text{ bytes} = 137.4 \text{ GB}$$
A single forward pass of $\mathbf{P}$ exceeds the entire memory capacity of an 80 GB NVIDIA H100.
Writing $\mathbf{S}$ and $\mathbf{P}$ to HBM and reading them back costs $\Theta(B H N^2)$ round-trip memory bandwidth. Since HBM bandwidth is limited ($\approx 3.35 \text{ TB/s}$ on H100), the GPU arithmetic units (Tensor Cores) stall waiting for data, operating at $<5\%$ efficiency.

---

### Q2: What is the Hong & Kung I/O pebbling lower bound, and how does FlashAttention match it asymptotically?
**Answer:**
Under the Hong & Kung (1981) red-blue pebbling game for matrix multiplication on a two-level memory hierarchy with fast on-chip SRAM capacity $M$:
Any algorithm computing $\mathbf{C} = \mathbf{A}\mathbf{B}$ for $N \times N$ matrices must transfer at least:
$$\Omega\left(\frac{N^3}{\sqrt{M}}\right) \text{ elements between slow memory and fast memory}$$
Applied to scaled dot-product attention $\mathbf{O} = \text{softmax}(\mathbf{Q}\mathbf{K}^T)\mathbf{V}$ where matrices are $N \times d$:
The theoretical lower bound of HBM transfers is:
$$\Omega\left(\frac{N^2 d^2}{M}\right)$$
Standard PyTorch Attention requires $\Theta(Nd + N^2)$ HBM transfers because it materializes the $N \times N$ attention matrix. FlashAttention chooses tile sizes $B_r \approx \frac{M}{4d}$ and $B_c \approx \frac{M}{4d}$, loading chunks into SRAM and performing the full attention computation without touching HBM. FlashAttention achieves:
$$\Theta\left(\frac{N^2 d^2}{M}\right) \text{ HBM transfers}$$
Thus, FlashAttention is **provably I/O-optimal**.

---

### Q3: Explain the GPU Roofline Model. How does FlashAttention transition Attention across the ridge point?
**Answer:**
The Roofline model plots Attainable Performance (TFLOPs/s) against Arithmetic Intensity (FLOPs / Byte of HBM transfer):
$$\text{Attainable Performance} = \min\left(\text{Peak Compute Performance}, \; \text{Memory Bandwidth} \times \text{Arithmetic Intensity}\right)$$
The **Ridge Point** is defined as:
$$\text{Ridge Point} = \frac{\text{Peak Compute FLOPs/s}}{\text{Peak Memory Bandwidth Bytes/s}}$$
For NVIDIA H100 SXM5:
- Peak BF16 Tensor Core FLOPs = $989 \times 10^{12} \text{ FLOPs/s}$
- Peak HBM3 Bandwidth = $3.35 \times 10^{12} \text{ Bytes/s}$
- $\text{Ridge Point} = \frac{989}{3.35} \approx 295 \text{ FLOPs/Byte}$

**Standard Attention Intensity:**
$$\text{Intensity} = \frac{4N^2 d}{4Nd \times 2 + 2N^2 \times 2} \approx \frac{4Nd}{4N} = d \approx 64 - 128 \text{ FLOPs/Byte}$$
Because $128 < 295$, standard attention is strictly **memory bandwidth bound**.

**FlashAttention Intensity:**
By keeping $\mathbf{S}$ and $\mathbf{P}$ in SRAM, HBM traffic is reduced by a factor of $\frac{B_c}{d}$:
$$\text{Intensity}_{\text{Flash}} \approx \frac{4N^2 d}{4Nd \times 2 + \frac{4N^2 d^2}{M}} \approx \frac{M}{2d} \approx \frac{100 \times 10^3}{2 \times 128} \approx 390 \text{ FLOPs/Byte}$$
Because $390 > 295$, FlashAttention pushes the computation past the ridge point into the **compute-bound** regime.

---

## 🔬 Section 2: Kernel Microarchitecture & CUDA Hardware

### Q4: What is a Shared Memory Bank Conflict, and how do you mathematically prove your code eliminates it?
**Answer:**
On NVIDIA GPUs, Shared Memory is partitioned into 32 independent memory banks of 4-byte (32-bit) words. Consecutive 32-bit words map to successive banks:
$$\text{Bank ID} = \left(\frac{\text{Byte Address}}{4}\right) \pmod{32}$$
When multiple threads within the same warp (32 threads) access addresses that map to the **same bank** but **different 32-bit words**, the hardware serializes the requests ($k$-way bank conflict), multiplying access latency by $k$.

**The Conflict in Naive Attention:**
Let head dimension $d = 64$ floats.
If shared memory is allocated as `__shared__ float s_Q[Br][d];`:
Thread $t$ accessing `s_Q[i][0]` has index $i \times 64 + 0$.
Thread $t+16$ accessing `s_Q[i+1][0]` has index $(i+1) \times 64 + 0$.
The bank IDs:
$$\text{Bank}(t) = (i \times 64) \pmod{32} = 0$$
$$\text{Bank}(t+16) = ((i+1) \times 64) \pmod{32} = 0$$
Both threads hit Bank 0, causing a 2-way bank conflict for all 16 thread pairs!

**The Mathematical Fix (Padding):**
Allocate with padding: `__shared__ float s_Q[Br][d + 4];`
Now stride is $64 + 4 = 68$ words.
$$\text{Bank}(t) = (i \times 68) \pmod{32} = (4i) \pmod{32}$$
$$\text{Bank}(t+1) = ((i+1) \times 68) \pmod{32} = (4i + 4) \pmod{32}$$
Every consecutive row shifts by 4 banks. Since $\gcd(68, 32) = 4$, rows cycle through bank assignments, achieving **zero-conflict conflict-free access**.

---

### Q5: What are Warp Shuffle instructions (`__shfl_down_sync`, `__shfl_xor_sync`), and why are they superior to Shared Memory reductions?
**Answer:**
1. **Zero Memory Latency:** Warp shuffles allow threads within a 32-thread warp to read registers directly from other threads within the same warp in **1 clock cycle**, with zero shared memory allocations and zero cache line accesses.
2. **Eliminates `__syncthreads()`:** `__syncthreads()` is a block-wide barrier that stalls all 128 to 1024 threads until the slowest thread arrives. In contrast, `__shfl_down_sync` synchronizes only the active warp threads via a bitmask (e.g., `0xffffffff`).
3. **Logarithmic Tree Reduction:**
```cpp
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val; // Thread 0 now contains sum of all 32 warp threads
}
```
In 5 instructions ($\log_2(32) = 5$), the entire warp is reduced without touching SRAM.

---

### Q6: Derive the FlashAttention Backward Pass. Why is recomputing in backward faster than reading cached attention matrices?
**Answer:**
The chain rule for attention $\mathbf{O} = \mathbf{P}\mathbf{V}$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{V}} = \mathbf{P}^T d\mathbf{O}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{P}} = d\mathbf{O} \mathbf{V}^T$$
Let $\mathbf{P} = \text{softmax}(\mathbf{S})$. The gradient with respect to scores $\mathbf{S}$:
$$d\mathbf{S} = \mathbf{P} \circ \left(d\mathbf{P} - \mathbf{D}\right) \quad \text{where } \mathbf{D}_i = \sum_{j} d\mathbf{P}_{ij} P_{ij} = \sum_{j} (d\mathbf{O}_i \cdot \mathbf{V}_j) P_{ij} = d\mathbf{O}_i \cdot \mathbf{O}_i$$
Then:
$$d\mathbf{Q} = \frac{1}{\sqrt{d}} d\mathbf{S} \mathbf{K}, \quad d\mathbf{K} = \frac{1}{\sqrt{d}} d\mathbf{S}^T \mathbf{Q}$$

**Why Recomputation Wins:**
Standard backprop caches $\mathbf{P} \in \mathbb{R}^{N \times N}$ during the forward pass and reads it during backward.
- Reading $\mathbf{P}$ from HBM: $O(N^2)$ reads at $\approx 3.35 \text{ TB/s}$.
- FlashAttention backward saves only the vector $L_i = m_i + \log(l_i) \in \mathbb{R}^N$ ($O(N)$ bytes).
- During backward, each tile of $\mathbf{Q}_i$ and $\mathbf{K}_j$ is reloaded into SRAM, and $\mathbf{P}_{ij} = \exp(\mathbf{Q}_i \mathbf{K}_j^T / \sqrt{d} - L_i)$ is recomputed in SRAM at **33 TB/s**.
Because SRAM compute is $>10\times$ faster than HBM read latency, recomputing $\mathbf{P}_{ij}$ on-the-fly executes **faster** than loading it from HBM!

---

## 🚀 Section 3: FlashAttention-2 vs FlashAttention-3 Innovations

### Q7: What are the key architectural differences between FlashAttention-1, 2, and 3?
| Feature | FlashAttention-1 | FlashAttention-2 | FlashAttention-3 (Hopper) |
| :--- | :--- | :--- | :--- |
| **Outer Loop** | Keys & Values ($K, V$) | Queries ($Q$) | Queries ($Q$) with TMA |
| **Non-MatMul FLOPs** | Rescaled both $O$ and $P$ | Rescaled only accumulator $O$ at step end | Asynchronous rescaling |
| **Parallelism Grid** | Batch $\times$ Heads | Batch $\times$ Heads $\times$ Seq Splits ($N / B_r$) | Distributed Warp Specialization |
| **Memory Transfer Engine** | Synchronous loads | `cp.async` (Ampere async copy) | Hardware TMA (Hopper Async DMA) |
| **MMA Instruction** | FP16 SIMD / WMMA | `mma.sync` Tensor Core tiles | `wgmma.mma_async` directly from SMEM |
| **Thread Architecture** | Homogeneous warps | Homogeneous warps | Warp Specialization (Producer vs Consumer) |
| **Attainable Peak MFU** | ~35-45% | ~50-65% | ~70-75% |

---

### Q8: What is Warp Specialization in FlashAttention-3?
**Answer:**
In traditional CUDA kernels, every thread in a threadblock executes the same sequence of instructions: load global memory to shared memory, synchronize (`__syncthreads()`), execute math, and write back.
In FlashAttention-3 on NVIDIA Hopper:
- The threadblock (e.g. 128 threads) is partitioned into **Producer Warps** (1 warp, 32 threads) and **Consumer Warps** (3 warps, 96 threads).
- **Producer Warp:** Never does matrix multiplication. Its sole job is to issue hardware TMA (Tensor Memory Accelerator) descriptors to stream $Q, K, V$ blocks from HBM into Shared Memory without consuming register file bandwidth.
- **Consumer Warps:** Never issue global memory loads. They wait on hardware transaction barriers (`cuda::barrier`) and issue `wgmma.mma_async` instructions to multiply matrices directly from Shared Memory into accumulator registers.
This decouples memory latency from computation pipelines, preventing pipeline stalls.

---

## 🛠️ Section 4: Production Diagnostics & Profiling

### Q9: How do you detect and eliminate Register Spilling using NVIDIA Nsight Compute (`ncu`)?
**Answer:**
- **Symptom:** In `ncu`, inspect `Memory [Bytes]` $\to$ `Local Memory Overhead`. Any non-zero local memory load/store indicates register spilling to the L1/L2 cache and global memory, slowing down execution by $5\times - 10\times$.
- **Diagnosis:** Compile with `nvcc -Xptxas -v` to inspect exact register counts per thread:
  ```bash
  ptxas info : Used 64 registers, 4096 bytes smem, 0 bytes cmem[0]
  ```
- **Remediation:**
  1. Add `__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)` to instruct the compiler on register budgeting.
  2. Reduce tile sizes ($B_r, B_c$) to shrink register accumulator arrays.
  3. Keep temporary variables in shared memory rather than local arrays with runtime indexes.
