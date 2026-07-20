# Profiling Analysis - Best Kernel (Level 18, Swizzled)

M=N=K=4096, Tesla T4, CUDA 12.x, `ncu --set full`

## Section 1: Speed of Light

| Metric                         | Value       |
| ------------------------------ | ----------- |
| Duration                       | 13.10 ms    |
| Compute (SM) Throughput        | 44.05%      |
| Memory Throughput              | 39.49%      |
| DRAM Throughput                | 16.49%      |
| L1/TEX Cache Throughput        | 78.98%      |
| L2 Cache Throughput            | 19.32%      |
| Registers Per Thread           | 94          |
| Static Shared Memory Per Block | 37.89 Kbyte |
| Theoretical Occupancy          | 50.00%      |
| Achieved Occupancy             | 49.80%      |
| Achieved Active Warps Per SM   | 15.94       |

The kernel is **latency-bound, not bandwidth-bound**. DRAM throughput is only 16.5% of peak, ruling out memory as the primary bottleneck. L1/TEX throughput is high (79%) because all data passes through shared memory. Compute throughput at 44% confirms the SM is idle more than half the time (stalls).

## Section 2: Occupancy + Warp Stall

| Metric                             | Value                    |
| ---------------------------------- | ------------------------ |
| Achieved Occupancy                 | 49.76%                   |
| Executed Ipc Active                | 0.63 inst/cycle          |
| SM Busy                            | 44.78%                   |
| Memory Throughput                  | 52.77 Gbyte/s            |
| Mem Pipes Busy                     | 28.24%                   |
| L2 Hit Rate                        | 81.82%                   |
| No Eligible Cycles                 | 84.13%                   |
| Eligible Warps Per Scheduler       | 0.39                     |
| Warp Cycles Per Issued Instruction | 25.11                    |
| Long Scoreboard Stall              | 0.0%                     |
| Short Scoreboard Stall             | 0.0%                     |
| **MIO Pipeline Stall**             | **45.13%** (11.3 cycles) |
| Wait Stall                         | 0.0%                     |
| Not Predicated Off                 | 99.79%                   |

### Stall Breakdown

- **45% MIO stall**: The warp is waiting for the MIO (memory I/O) pipeline to drain. This handles shared memory loads/stores, special math, and other "slow" operations. Our uint4 vectorized SMEM stores are saturating the MIO pipeline. Each warp issues stores through LSU → MIO, and 84% of cycles have no warp eligible because all warps are stuck waiting.
- **0% scoreboard stalls**: No outstanding global memory loads - the pipeline hides global latency perfectly (81.8% L2 hit rate helps).
- **84% No Eligible**: Only 0.39 eligible warps per scheduler on average (4 active). This is the direct consequence of the MIO pipeline being the serialization point.

The MIO stall is the single biggest performance problem. In CUTLASS, `ldmatrix` PTX goes through the tensor memory pipeline (separate from MIO), avoiding this bottleneck entirely.

## Section 3: Bank Conflicts

| Metric                                    | Value                            |
| ----------------------------------------- | -------------------------------- |
| Load bank conflicts                       | 209,715                          |
| Store bank conflicts                      | 8,388,608                        |
| Load wavefronts                           | 629,145                          |
| Store wavefronts                          | 25,165,824                       |
| Store conflicts as % of wavefronts        | 33.33%                           |
| Average way conflicts (stores)            | 6.0-way                          |
| Load conflicts as % of wavefronts         | ~33% (negligible absolute count) |
| Excessive wavefronts (uncoalesced shared) | 8.9%                             |

### Why 6-way bank conflicts?

Shared memory has 32 banks (4-byte each). The A fragment SMEM array (`As[2][BK+SKEW_HALF][BK]`) has stride `BK+SKEW_HALF = 32+8 = 40` \_\_half elements = 80 bytes per row. 80 bytes spans banks `[0..19]` with 4 bytes per bank → the stride modulo 128 (32 × 4 bytes) = 80 mod 128 = 80 bytes = 20 banks. Since each half is 2 bytes, 40 halves × 2 bytes = 80 bytes = stride 80. The modulo-128 of stride 80 means each successive row starts at bank `(prev + 20) mod 32`. With 8 rows per fragment and 2 fragments per register-tile, the 16 active rows during a store pattern interleave across banks, creating 6-way conflicts on average.

### Store Pattern

The store in the kernel is:

```cuda
// From Phase 18 kernel
((uint4*)As[curr])[row + warp_row * (BK+SKEW_HALF)/4] =
    make_uint4(a.x, a.y, a.z, a.w);
```

where each warp stores a row of 16 halves (= 8 uint4 elements) to SMEM. The stride of 40 halves (80 bytes) across banks creates the conflict pattern.

## Section 4: L2 Cache Behavior

| Metric                   | Value      |
| ------------------------ | ---------- |
| L2 Hit Rate              | 81.81%     |
| L2 Sectors Read (tex)    | 67,108,864 |
| DRAM Throughput (% peak) | 16.46%     |
| L2 Throughput (% peak)   | 19.32%     |

The 81.82% L2 hit rate confirms the **threadblock swizzle (GROUP_M=4)** is working. Consecutively launched threadblocks process tiles that share B column ranges, so B's data remains hot in L2 across blocks. Without the swizzle, each block's B data would evict the previous block's before the next block in the same output column launches.

## Section 5: Tensor Core Utilization

| Metric                                   | Value  |
| ---------------------------------------- | ------ |
| Tensor Core Pipe Cycles Active (% peak)  | 44.12% |
| Tensor Instruction Issue (% peak active) | 22.29% |

The HMMA pipeline is active only 44% of the time. The remaining 56% of cycles, the tensor cores are idle because warps are stalled on the MIO pipeline (loading SMEM→regs or storing regs→SMEM for the next tile). This directly quantifies the gap to cuBLAS: if we could keep the tensor cores busy 100% of the time, we'd approach 2.27× speedup - roughly 36 TFLOPS (cuBLAS territory).

## Summary of Bottlenecks (Ranked)

| #   | Bottleneck                       | Impact                            | Root Cause                                              |
| --- | -------------------------------- | --------------------------------- | ------------------------------------------------------- |
| 1   | **MIO Pipeline Stall**           | **45% of warp cycles**            | uint4 SMEM stores + shared load/store through MIO pipe  |
| 2   | **Bank Conflicts (Store)**       | **6-way avg, 33% wavefronts**     | Stride-80-byte SMEM layout with 32 banks                |
| 3   | **Low Tensor Core Utilization**  | **44% active**                    | Consequence of #1 - tensor cores idle while warps stall |
| 4   | **Register Pressure**            | **94 regs/thread, 50% occupancy** | 2 WMMA accumulator fragments + SMEM load/store temps    |
| 5   | **L2 (not really a bottleneck)** | **81.8% hit rate**                | Working well - swizzle effective                        |

## What cuBLAS Does Differently

Based on CUTLASS architecture (which cuBLAS is built on):

1. **`ldmatrix` PTX instead of SMEM → WMMA load**: CUTLASS uses `ldmatrix` to load data directly from shared memory into warp registers through the _tensor memory accelerator_ pipeline (not MIO). This bypasses the MIO pipeline entirely.

2. **XOR-swizzled SMEM layout**: Instead of padding to avoid bank conflicts, CUTLASS reorders bytes within each 128-byte SMEM row using XOR-based swizzle so that `ldmatrix` accesses hit unique banks. This eliminates bank conflicts at the source.

3. **Software pipelining with `cp.async`**: CUTLASS uses `cp.async` (Turing async copy) to move global→SMEM without consuming registers, completely decoupling the global-load pipeline from the compute pipeline.

4. **Higher register budgets**: CUTLASS kernels often use 128+ registers/thread to keep more state in registers, reducing shared-memory traffic.

These differences collectively explain the remaining ~28% gap.
