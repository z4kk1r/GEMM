#include "../common.cuh"

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void gemm_double_buffered(int M, int N, int K,
                                     const float* A, const float* B, float* C) {
    __shared__ float As[2][BM][BK + 1];
    __shared__ float Bs[2][BK][BN];

    int threadCol = threadIdx.x % (BN / TN);
    int threadRow = threadIdx.x / (BN / TN);

    float threadResults[TM][TN] = {0.0f};

    int numTiles = K / BK;
    int tid = threadIdx.x;
    int numThreads = blockDim.x;

    for (int i = tid; i < BM * BK; i += numThreads) {
        int r = i / BK;
        int c = i % BK;
        As[0][r][c] = A[(blockIdx.y * BM + r) * K + c];
    }
    for (int i = tid; i < BK * BN / 4; i += numThreads) {
        int idx = i * 4;
        int r = idx / BN;
        int c = idx % BN;
        reinterpret_cast<float4*>(&Bs[0][r][c])[0] =
            reinterpret_cast<const float4*>(&B[r * N + blockIdx.x * BN + c])[0];
    }
    __syncthreads();

    int cur = 0;
    for (int t = 0; t < numTiles; t++) {
        int nxt = cur ^ 1;
        int nextTile = t + 1;

        if (nextTile < numTiles) {
            for (int i = tid; i < BM * BK; i += numThreads) {
                int r = i / BK;
                int c = i % BK;
                As[nxt][r][c] = A[(blockIdx.y * BM + r) * K + nextTile * BK + c];
            }
            for (int i = tid; i < BK * BN / 4; i += numThreads) {
                int idx = i * 4;
                int r = idx / BN;
                int c = idx % BN;
                reinterpret_cast<float4*>(&Bs[nxt][r][c])[0] =
                    reinterpret_cast<const float4*>(
                        &B[(nextTile * BK + r) * N + blockIdx.x * BN + c])[0];
            }
        }

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float regA[TM];
            float regB[TN];
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                regA[m] = As[cur][threadRow * TM + m][k];
            }
            #pragma unroll
            for (int n = 0; n < TN; n += 4) {
                reinterpret_cast<float4*>(&regB[n])[0] =
                    reinterpret_cast<const float4*>(&Bs[cur][k][threadCol * TN + n])[0];
            }
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    threadResults[m][n] += regA[m] * regB[n];
                }
            }
        }

        __syncthreads();
        cur = nxt;
    }

    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int row = blockIdx.y * BM + threadRow * TM + m;
            int col = blockIdx.x * BN + threadCol * TN + n;
            if (row < M && col < N) {
                C[row * N + col] = threadResults[m][n];
            }
        }
    }
}

float run_double_buffered(int M, int N, int K,
                          const float* d_A, const float* d_B, float* d_C,
                          int warmup_iters = 5, int bench_iters = 20) {
    int threadsPerBlock = (BM * BN) / (TM * TN);
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    for (int i = 0; i < warmup_iters; i++) {
        gemm_double_buffered<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaTimer timer;
    timer.begin();
    for (int i = 0; i < bench_iters; i++) {
        gemm_double_buffered<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    return timer.end() / bench_iters;
}

#ifndef BENCHMARK_ALL
int main() {
    int M = 4096, N = 4096, K = 4096;

    srand(42);
    printf("=== Phase 7: Double Buffering / Software Pipelining ===\n");
    printf("Config: BM=%d, BN=%d, BK=%d, TM=%d, TN=%d\n", BM, BN, BK, TM, TN);
    print_device_info();

    size_t size_a = (size_t)M * K * sizeof(float);
    size_t size_b = (size_t)K * N * sizeof(float);
    size_t size_c = (size_t)M * N * sizeof(float);

    float *h_A = (float*)malloc(size_a);
    float *h_B = (float*)malloc(size_b);
    float *h_C = (float*)malloc(size_c);
    float *h_C_ref = (float*)malloc(size_c);

    randomize_matrix(h_A, M * K);
    randomize_matrix(h_B, K * N);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_a));
    CHECK_CUDA(cudaMalloc(&d_B, size_b));
    CHECK_CUDA(cudaMalloc(&d_C, size_c));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float ms_ref = cublas_gemm_f32(handle, M, N, K, d_A, d_B, d_C);
    CHECK_CUDA(cudaMemcpy(h_C_ref, d_C, size_c, cudaMemcpyDeviceToHost));
    double gflops_ref = get_gflops(M, N, K, ms_ref);
    printf("cuBLAS: %.3f ms (%.1f GFLOPS)\n\n", ms_ref, gflops_ref);

    float ms = run_double_buffered(M, N, K, d_A, d_B, d_C);
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_c, cudaMemcpyDeviceToHost));
    double gflops = get_gflops(M, N, K, ms);

    printf("Double-buffered kernel: %.3f ms (%.1f GFLOPS)\n", ms, gflops);
    printf("Performance: %.1f%% of cuBLAS\n", 100.0 * gflops / gflops_ref);

    printf("\nVerification against cuBLAS reference:\n");
    bool ok = verify_result(h_C_ref, h_C, (size_t)M * N, "Double-Buffered");
    printf("%s\n\n", ok ? "PASS" : "FAIL");

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    return ok ? 0 : 1;
}
#endif
