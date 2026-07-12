#!/usr/bin/env python3
import subprocess
import re
import itertools
import os
import sys

KERNEL_SOURCE = r'''
#include "common.cuh"

#define BM {BM}
#define BN {BN}
#define BK {BK}
#define TM {TM}
#define TN {TN}

__global__ void gemm_autotune(int M, int N, int K,
                              const float* A, const float* B, float* C) {{
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN];

    int threadCol = threadIdx.x % (BN / TN);
    int threadRow = threadIdx.x / (BN / TN);

    float threadResults[TM][TN] = {{0.0f}};

    int numTiles = K / BK;
    int tid = threadIdx.x;
    int numThreads = blockDim.x;

    for (int t = 0; t < numTiles; t++) {{
        for (int i = tid; i < BM * BK; i += numThreads) {{
            int r = i / BK;
            int c = i % BK;
            As[r][c] = A[(blockIdx.y * BM + r) * K + t * BK + c];
        }}
        for (int i = tid; i < BK * BN / 4; i += numThreads) {{
            int idx = i * 4;
            int r = idx / BN;
            int c = idx % BN;
            reinterpret_cast<float4*>(&Bs[r][c])[0] =
                reinterpret_cast<const float4*>(&B[(t * BK + r) * N + blockIdx.x * BN + c])[0];
        }}
        __syncthreads();

        for (int k = 0; k < BK; k++) {{
            float regA[TM];
            float regB[TN];
            for (int m = 0; m < TM; m++) regA[m] = As[threadRow * TM + m][k];
            for (int n = 0; n < TN; n += 4)
                reinterpret_cast<float4*>(&regB[n])[0] =
                    reinterpret_cast<const float4*>(&Bs[k][threadCol * TN + n])[0];
            for (int m = 0; m < TM; m++)
                for (int n = 0; n < TN; n++)
                    threadResults[m][n] += regA[m] * regB[n];
        }}
        __syncthreads();
    }}

    for (int m = 0; m < TM; m++) {{
        for (int n = 0; n < TN; n++) {{
            int row = blockIdx.y * BM + threadRow * TM + m;
            int col = blockIdx.x * BN + threadCol * TN + n;
            if (row < 4096 && col < 4096)
                C[row * 4096 + col] = threadResults[m][n];
        }}
    }}
}}

float run_autotune(int M, int N, int K,
                   const float* d_A, const float* d_B, float* d_C,
                   int warmup_iters, int bench_iters) {{
    int threadsPerBlock = (BM * BN) / (TM * TN);
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    for (int i = 0; i < warmup_iters; i++) {{
        gemm_autotune<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    }}
    cudaDeviceSynchronize();

    CudaTimer timer;
    timer.begin();
    for (int i = 0; i < bench_iters; i++) {{
        gemm_autotune<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    }}
    cudaDeviceSynchronize();
    return timer.end() / bench_iters;
}}

int main() {{
    int M = 4096, N = 4096, K = 4096;
    srand(42);

    size_t size_a = (size_t)M * K * sizeof(float);
    size_t size_b = (size_t)K * N * sizeof(float);
    size_t size_c = (size_t)M * N * sizeof(float);

    float *h_A = (float*)malloc(size_a);
    float *h_B = (float*)malloc(size_b);
    srand(42);
    for (size_t i = 0; i < (size_t)M * K; i++) h_A[i] = ((float)rand()/RAND_MAX)*2-1;
    srand(42);
    for (size_t i = 0; i < (size_t)K * N; i++) h_B[i] = ((float)rand()/RAND_MAX)*2-1;

    float *d_A, *d_B, *d_C, *d_C_ref;
    cudaMalloc(&d_A, size_a);
    cudaMalloc(&d_B, size_b);
    cudaMalloc(&d_C, size_c);
    cudaMalloc(&d_C_ref, size_c);
    cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float ms_ref = cublas_gemm_f32(handle, 4096, 4096, 4096, d_A, d_B, d_C_ref, 3, 10);
    cudaDeviceSynchronize();
    float ms = run_autotune(4096, 4096, 4096, d_A, d_B, d_C, 3, 10);

    float* h_C = (float*)malloc(size_c);
    float* h_C_ref = (float*)malloc(size_c);
    cudaMemcpy(h_C, d_C, size_c, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_ref, d_C_ref, size_c, cudaMemcpyDeviceToHost);

    bool ok = true;
    for (size_t i = 0; i < (size_t)M * N; i++) {{
        float diff = fabs(h_C_ref[i] - h_C[i]);
        float ref_abs = fabs(h_C_ref[i]);
        float rel_diff = (ref_abs > 1.0f) ? diff / ref_abs : diff;
        if (rel_diff > 1e-3f) {{ ok = false; break; }}
    }}

    printf("CONFIG BM=%d BN=%d BK=%d TM=%d TN=%d | MS=%.3f | GFLOPS=%.1f | PASS=%s\\n",
           BM, BN, BK, TM, TN,
           ms, 2.0*4096*4096*4096/(ms*1e6), ok ? "YES" : "NO");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    cublasDestroy(handle);
    return ok ? 0 : 1;
}}
'''


def generate_configs():
    bm_vals = [64, 128, 256]
    bn_vals = [64, 128, 256]
    bk_vals = [8, 16, 32]
    tm_vals = [4, 8, 16]
    tn_vals = [4, 8]

    configs = []
    for BM, BN, BK, TM, TN in itertools.product(bm_vals, bn_vals, bk_vals, tm_vals, tn_vals):
        if TM * TN > 128:
            continue
        nthreads = (BM * BN) // (TM * TN)
        if nthreads < 64 or nthreads > 1024:
            continue
        smem_a = BM * (BK + 1) * 4
        smem_b = BK * BN * 4
        if smem_a + smem_b > 48 * 1024:
            continue
        if BM % TM != 0 or BN % TN != 0:
            continue
        configs.append((BM, BN, BK, TM, TN))
    return configs


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = "/tmp/autotune"
    os.makedirs(tmp_dir, exist_ok=True)
    results = []
    errors = 0
    skipped = 0

    configs = generate_configs()
    print(f"Testing {len(configs)} configurations...\n")

    for i, (BM, BN, BK, TM, TN) in enumerate(configs):
        code = KERNEL_SOURCE.format(
            BM=BM, BN=BN, BK=BK, TM=TM, TN=TN
        )

        base = f"{tmp_dir}/cfg_{i}"
        src_path = base + ".cu"
        exe_path = base

        with open(src_path, "w") as f:
            f.write(code)

        cmd = f"nvcc -arch=sm_75 -std=c++17 -O3 -I{script_dir} -o {exe_path} {src_path} -lcublas 2>{tmp_dir}/err.txt"
        ret = os.system(cmd)
        if ret != 0:
            skipped += 1
            with open(f"{tmp_dir}/err.txt") as f:
                err = f.read().strip()
            if err:
                print(f"  SKIP BM={BM:>3} BN={BN:>3} BK={BK:>2} TM={TM:>2} TN={TN:>2}: compile error")
            try:
                os.remove(src_path)
            except:
                pass
            continue

        try:
            raw = subprocess.check_output([exe_path], timeout=120, stderr=subprocess.STDOUT)
            output = raw.decode()
        except subprocess.CalledProcessError as e:
            output = e.output.decode() if e.output else ""
            errors += 1
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT BM={BM:>3} BN={BN:>3} BK={BK:>2} TM={TM:>2} TN={TN:>2}")
            errors += 1
            try:
                os.remove(src_path)
                os.remove(exe_path)
            except:
                pass
            continue

        match = re.search(r"MS=([\d.]+).*GFLOPS=([\d.]+).*PASS=(\w+)", output)
        if match:
            ms = float(match.group(1))
            gflops = float(match.group(2))
            passed = match.group(3) == "YES"
            if passed:
                results.append((BM, BN, BK, TM, TN, ms, gflops, passed))
                print(f"  OK   BM={BM:>3} BN={BN:>3} BK={BK:>2} TM={TM:>2} TN={TN:>2} -> {gflops:>8.0f} GFLOPS")
            else:
                print(f"  FAIL BM={BM:>3} BN={BN:>3} BK={BK:>2} TM={TM:>2} TN={TN:>2}: verification failed")
        else:
            first_line = output.strip().split('\n')[0] if output else "(no output)"
            print(f"  BAD  BM={BM:>3} BN={BN:>3} BK={BK:>2} TM={TM:>2} TN={TN:>2}: {first_line[:80]}")

        try:
            os.remove(src_path)
            os.remove(exe_path)
        except:
            pass

    results.sort(key=lambda r: -r[6])

    print(f"\n{'='*60}")
    print(f"Results: {len(results)} passed, {errors} runtime errors, {skipped} compile errors")
    print(f"{'='*60}")

    if results:
        best = results[0]
        ref_gflops = best[6]
        print(f"\n{'BM':>4} {'BN':>4} {'BK':>4} {'TM':>4} {'TN':>4} {'GFLOPS':>9} {'%best':>7}")
        print('-' * 42)
        for r in results[:10]:
            pct = r[6] / best[6] * 100
            print(f"{r[0]:>4} {r[1]:>4} {r[2]:>4} {r[3]:>4} {r[4]:>4} {r[6]:>9.0f} {pct:>6.1f}%")

        print(f"\nBest: BM={best[0]} BN={best[1]} BK={best[2]} TM={best[3]} TN={best[4]} -> {best[6]:.0f} GFLOPS")
    else:
        print("\nNo working configurations found.")
        print("Check CUDA availability and common.cuh dependencies.")


if __name__ == "__main__":
    main()
