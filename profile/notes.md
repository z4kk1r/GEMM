# Nsight Compute Profiling Notes

Use this file to record profiling observations for each kernel level.

## Device

- GPU: Tesla T4
- Compute Capability: 7.5

## Profiling Commands

```bash
# Profile a specific kernel
sudo ncu --set full -o profile_naive kernels/00_naive.exe

# Key metrics to check:
# - l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld
# - sm__throughput.avg.pct_of_peak_sustained_elapsed
# - l1tex__t_request_access_sum
# - l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
```

## Level 0: Naive

- Date:
- Duration:
- Observations:

## Level 2: Shared Memory Tiling

- Date:
- Duration:
- Observations:

## Level 3: 1D Register Blocking

- Date:
- Duration:
- Observations:

## Level 4: 2D Register Blocking

- Date:
- Duration:
- Observations:

## Level 5: Vectorized

- Date:
- Duration:
- Observations:

## Level 6: Bank-Conflict-Free

- Date:
- GPUDump or Nsight Compute output filename:
- Bank conflicts before (Level 5):
- Bank conflicts after:
- Speedup observed:
- Observations:

## Level 7: Double Buffered

- Date:
- Duration:
- Observations:

## Level 9: WMMA Tensor Core

- Date:
- Duration:
- Observations:

## Summary Table

| Level | Technique | Duration (ms) | GFLOPS | % of Peak | Bank Conflicts | Notes |
|-------|-----------|---------------|--------|-----------|----------------|-------|
| 0 | Naive | | | | | |
| 2 | Shared Mem | | | | | |
| 3 | 1D Reg Block | | | | | |
| 4 | 2D Reg Block | | | | | |
| 5 | Vectorized | | | | | |
| 6 | Bank-Free | | | | | |
| 7 | Double Buffer | | | | | |
| 9 | Tensor Core | | | | | |
