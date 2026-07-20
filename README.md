# CUDA GEMM: From Naive to 72% of cuBLAS

A step-by-step journey optimizing SGEMM (single-precision GEMM) and HGEMM (half-precision tensor-core GEMM) on an NVIDIA T4 (sm_75), progressing from a naive 567 GFLOPS kernel to a WMMA-based tensor-core kernel reaching 72.4% of cuBLAS FP16 throughput.

## Background

General Matrix Multiplication (GEMM) - `C = A × B + C` - is the computational foundation of deep learning and scientific computing. cuBLAS represents the highly optimized reference. This project systematically re-discovers the major CUDA optimization techniques by implementing them one phase at a time, measuring each against cuBLAS on real hardware (Tesla T4, 40 SMs, Compute Capability 7.5).

## Hardware & Software

| Item               | Detail                                                                   |
| ------------------ | ------------------------------------------------------------------------ |
| GPU                | NVIDIA Tesla T4 (40 SMs, 64 KB shared mem/block, 1024 max threads/block) |
| Compute Capability | 7.5 (Turing)                                                             |
| FP16 Tensor Core   | 64 HMMA per SM per cycle                                                 |
| CUDA Version       | 12.x                                                                     |
| cuBLAS             | cuBLAS 12.x (FP32: ~3,945 GFLOPS, FP16: ~22,069 GFLOPS)                  |
| Problem Size       | M = N = K = 4096                                                         |

## Optimization Phases

### FP32 Kernels (Levels 0–8)

| Level | Technique              | Peak GFLOPS | % cuBLAS FP32 | Key Idea                                                     |
| ----- | ---------------------- | ----------- | ------------- | ------------------------------------------------------------ |
| 0     | Naive triple-loop      | 567         | 14.4%         | Straightforward `C[i][j] += A[i][k] * B[k][j]`               |
| 2     | Shared-memory tiling   | 882         | 22.4%         | Tile A/B in shared memory to reuse data across warps         |
| 3     | 1D register blocking   | 1,564       | 39.6%         | Each thread accumulates multiple C elements in registers     |
| 4     | 2D register blocking   | 3,083       | 78.1%         | Threads cover 2D output tiles → better register reuse        |
| 5     | Vectorized (float4)    | 3,195       | 81.0%         | `float4` global loads → fewer instructions, higher bandwidth |
| 6     | Bank-conflict-free     | 3,118       | 79.0%         | Padding in shared memory to avoid 16-way bank conflicts      |
| 7     | Double buffering       | 3,076       | 78.0%         | Overlap global→shared with compute via ping-pong buffers     |
| 8     | Fully vectorized A+B+C | 3,282       | 83.2%         | `float4` stores for C too                                    |

**Takeaway:** FP32 peaks at ~83% of cuBLAS FP32. The remaining gap is likely a combination of instruction-level scheduling, register pressure, and prefetching that hand-tuned assembly or CUTLASS exploits beyond what simple CUDA C++ can reach.

### WMMA / Tensor Core Kernels (Levels 9–20)

| Level  | Technique                       | GFLOPS     | % cuBLAS FP16 | Key Idea                                                        |
| ------ | ------------------------------- | ---------- | ------------- | --------------------------------------------------------------- |
| 9      | WMMA naive                      | 3,296      | 14.9%         | First tensor-core kernel using `wmma::mma_sync`                 |
| 10     | Multi-tile (2×2 per warp)       | 5,670      | 25.7%         | Each warp computes 4 output tiles → more work per load          |
| 11     | Direct B load                   | 3,021      | 13.7%         | Load B in col-major directly - regressed (bank conflicts)       |
| 12     | Optimal (128-block, 512-thread) | 7,729      | 35.0%         | B transpose via shared memory, named fragments                  |
| 13     | Combined (regression)           | 4,906      | 22.2%         | Merged direct-Load + multi-tile - bank conflicts                |
| 14     | Padded (bank-conflict-free)     | 8,802      | 39.9%         | `BK+8`/`BN+8` padding + `alignas(16)` for `load_matrix_sync`    |
| 15     | Pipelined (2-stage)             | 8,620      | 39.1%         | Prefetch next tile into regs while computing current tile       |
| **16** | **No B transpose**              | **12,389** | **56.1%**     | Use `row_major` B fragment (B is already row-major in global)   |
| **17** | **Deep-K (BK=32)**              | **13,693** | **62.1%**     | Each SMEM tile feeds 2 WMMA passes → halved sync overhead       |
| **18** | **Threadblock swizzle**         | **15,982** | **72.4%**     | Reorder block-to-tile mapping for L2 reuse (GROUP_M=4)          |
| 19     | Scalar SMEM stores              | 15,331     | 69.5%         | Scalar stores fix 4-way bank conflicts - but killed L2 benefit? |
| 20     | Wide warp (1024 threads)        | 13,637     | 61.8%         | Full SM occupancy (1×2 per warp) - more threads ≠ faster        |

## Key Insights

### 1. The B-transpose elimination (Level 16) was the biggest single jump

Phase 16 removed the shared-memory B transpose entirely - B in global memory is row-major K×N, which is the _native_ layout for a `row_major` WMMA fragment. No transpose needed. This removed one full SMEM array, one scalar-copy pass per tile, and one `__syncthreads()`. **Result: +17 percentage points** (39% → 56%).

### 2. Deeper K-tiling (Level 17) compounds the savings

Doubling BK from 16 to 32 halves the outer-loop count. Each SMEM tile now feeds two inner `WMMA_K=16` sub-passes. **Result: +6 points** (56% → 62%).

### 3. Threadblock swizzle (Level 18) exploits the L2 cache

The CUTLASS-inspired `remap_block()` reorders block-to-output-tile assignment so consecutively launched blocks share B column-tile ranges. **Result: +10 points** (62% → 72%).

### 4. Scalar stores (Level 19) showed a puzzling regression

Replacing uint4 SMEM stores with scalar half stores eliminated 4-way bank conflicts but _dropped_ performance. This suggests the L2 benefit from the swizzle is fragile - the extra scalar-store instructions may have altered scheduling enough to hurt cache reuse.

### 5. Maximum occupancy (Level 20) didn't help

Doubling threads to 1024 (full SM) while halving per-warp output (1×2 tiles) should have hidden memory latency, but performance dropped. Register pressure from 1024 threads/block likely forced spilling or reduced per-warp efficiency.

### 6. Alignment matters for WMMA

`wmma::load_matrix_sync` emits `ldmatrix` PTX, which does 128-bit vectorized loads per lane. The leading dimension must be a multiple of 8 \_\_half elements (16 bytes). Using `BK+8` padding (not `BK+1`) and `alignas(16)` on shared memory arrays is mandatory.

## Profiling Results (Best Kernel: Level 18 - Swizzled)

Profiled with Nsight Compute on Tesla T4, M=N=K=4096.

| Metric                      | Value     | Notes                                                        |
| --------------------------- | --------- | ------------------------------------------------------------ |
| Compute Throughput          | 44.05%    | Below 60% - latency-bound, not compute-bound                 |
| DRAM Throughput             | 16.49%    | Far from bandwidth-limited                                   |
| L1/TEX Throughput           | 78.98%    | Heavy shared-memory traffic                                  |
| SM Busy                     | 44.78%    |                                                              |
| Achieved Occupancy          | 49.80%    | 15.94 warps/SM (cap: 50% from 94 regs/thread + 37.9 KB SMEM) |
| Registers Per Thread        | 94        | Primary occupancy limiter (1 block/SM)                       |
| Static Shared Memory        | 37.89 KB  | Secondary occupancy limiter                                  |
| IPC (Active)                | 0.63      | Instructions per cycle                                       |
| Warp Cycles Per Instr.      | 25.11     | Cycles between warp issues                                   |
| No Eligible Cycles          | 84.13%    | Only 0.39 eligible warps/scheduler out of 4 active           |
| MIO Stall (% of issue)      | 45.13%    | Top stall reason - shared memory pipeline pressure           |
| SMEM Bank Conflicts (load)  | 209,715   | Negligible                                                   |
| SMEM Bank Conflicts (store) | 8,388,608 | **6.0-way conflicts** - 33% of store wavefronts affected     |
| L2 Hit Rate                 | 81.82%    | High - swizzle is working                                    |
| Tensor Core Pipe Util.      | 44.12%    | HMMA pipeline active % of peak sustained elapsed             |
| Tensor Instruction Issue    | 22.29%    | % of peak sustained active                                   |
| Duration                    | 13.10 ms  |                                                              |

### Bottleneck Analysis

The single largest bottleneck is the **MIO pipeline stall (45.1%)** caused by shared-memory store pressure. The uint4 vectorized stores to SMEM with `BK+SKEW_HALF=40` stride create systematic **6-way bank conflicts** on stores, wasting 33% of store wavefronts. This explains why:

1. **Swizzle works** - 81.82% L2 hit rate confirms consecutive blocks share B data
2. **Occupancy isn't the problem** - only 50% theoretical, but 84% of cycles have no eligible warp anyway; more warps wouldn't help without fixing the stall
3. **Scalar stores (Level 19) regressed** - fixing bank conflicts increased instruction count enough that the MIO pipeline became even more saturated, negating the benefit
4. **The real lever is `ldmatrix` + XOR-swizzle** - CUTLASS's approach avoids scalar SMEM stores entirely by using `ldmatrix` PTX directly with a swizzled shared-memory layout, eliminating both the bank conflicts and the MIO pipeline pressure

## How to Run

### On Colab (recommended)

1. Upload the project zip
2. Open `gemm_colab.ipynb`
3. Run cells sequentially to compile and benchmark individual phases
4. Run the "Combined Benchmark" cell for the full comparison

### Standalone compilation

```bash
# Individual phase
nvcc -arch=sm_75 -std=c++17 -O3 -o kernel kernels/XX_phase.cu -lcublas -lcudart
./kernel

# Combined benchmark
nvcc -arch=sm_75 -std=c++17 -O3 -DBENCHMARK_ALL -o bench/benchmark_all \
    bench/benchmark_all.cu kernels/*.cu -lcublas -lcudart
./bench/benchmark_all
```

### Profiling with Nsight Compute

```bash
# Run all profiling passes (builds standalone + installs ncu if needed)
bash profile_best.sh

# Extract all text reports into one file
cat profile/*.txt > full_profile_report.txt
```

## Project Structure

```
GEMM/
├── common.cuh           # Shared utilities: timer, verification, cuBLAS wrappers
├── kernels/
│   ├── 00_naive.cu      # FP32 naive
│   ├── 02_shared_tiled.cu
│   ├── 03_1d_blocktiling.cu
│   ├── ...
│   ├── 18_wmma_swizzled.cu  # Best WMMA kernel (72.4% cuBLAS)
│   ├── 19_wmma_scalarstore.cu
│   └── 20_wmma_widewarp.cu
├── bench/
│   └── benchmark_all.cu # Unified benchmark driver
├── gemm_colab.ipynb     # Colab notebook
├── autotune.py          # Autotuning script
├── Makefile / CMakeLists.txt
└── README.md
```

## References

- [CUTLASS](https://github.com/NVIDIA/cutlass) - NVIDIA's template library for GEMM
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [wmma::mma_sync documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma-mma-sync)
- [Programming Tensor Cores in CUDA 9](https://devblogs.nvidia.com/programming-tensor-cores-cuda-9/) (Mark Harris, NVIDIA Developer Blog)
- T4 Tensor Core benchmarks and roofline analysis from various NVIDIA GTC presentations
