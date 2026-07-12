#include "../common.cuh"

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void gemm_vectorized_all(int M, int N, int K,
                                    const float* A, const float* B, float* C) {
    int threadCol = threadIdx.x % (BN / TN);
    int threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float threadResults[TM][TN] = {0.0f};

    int numTiles = K / BK;
    for (int t = 0; t < numTiles; t++) {
        int tid = threadIdx.x;
        int numThreads = blockDim.x;

        for (int i = tid; i < BM * BK / 4; i += numThreads) {
            int idx = i * 4;
            int r = idx / BK;
            int c = idx % BK;
            reinterpret_cast<float4*>(&As[r][c])[0] =
                reinterpret_cast<const float4*>(&A[(blockIdx.y * BM + r) * K + t * BK + c])[0];
        }

        for (int i = tid; i < BK * BN / 4; i += numThreads) {
            int idx = i * 4;
            int r = idx / BN;
            int c = idx % BN;
            reinterpret_cast<float4*>(&Bs[r][c])[0] =
                reinterpret_cast<const float4*>(&B[(t * BK + r) * N + blockIdx.x * BN + c])[0];
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float regA[TM];
            float regB[TN];
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                regA[m] = As[threadRow * TM + m][k];
            }
            #pragma unroll
            for (int n = 0; n < TN; n += 4) {
                reinterpret_cast<float4*>(&regB[n])[0] =
                    reinterpret_cast<const float4*>(&Bs[k][threadCol * TN + n])[0];
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
    }

    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int row = blockIdx.y * BM + threadRow * TM + m;
        #pragma unroll
        for (int n = 0; n < TN; n += 4) {
            int col = blockIdx.x * BN + threadCol * TN + n;
            if (row < M && col + 3 < N) {
                reinterpret_cast<float4*>(&C[row * N + col])[0] =
                    reinterpret_cast<const float4*>(&threadResults[m][n])[0];
            }
        }
    }
}

float run_vectorized_all(int M, int N, int K,
                         const float* d_A, const float* d_B, float* d_C,
                         int warmup_iters = 5, int bench_iters = 20) {
    int threadsPerBlock = (BM * BN) / (TM * TN);
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    for (int i = 0; i < warmup_iters; i++) {
        gemm_vectorized_all<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaTimer timer;
    timer.begin();
    for (int i = 0; i < bench_iters; i++) {
        gemm_vectorized_all<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    return timer.end() / bench_iters;
}

#ifndef BENCHMARK_ALL
int main() {
    int M = 4096, N = 4096, K = 4096;

    srand(42);
    printf("=== Phase 8: Fully Vectorized (float4 A + B + C) ===\n");
    printf("Config: BM=%d, BN=%d, BK=%d, TM=%d, TN=%d (float4 A+B+C)\n", BM, BN, BK, TM, TN);
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

    float ms = run_vectorized_all(M, N, K, d_A, d_B, d_C);
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_c, cudaMemcpyDeviceToHost));
    double gflops = get_gflops(M, N, K, ms);

    printf("Fully vectorized kernel: %.3f ms (%.1f GFLOPS)\n", ms, gflops);
    printf("Performance: %.1f%% of cuBLAS\n", 100.0 * gflops / gflops_ref);

    printf("\nVerification against cuBLAS reference:\n");
    bool ok = verify_result(h_C_ref, h_C, (size_t)M * N, "Fully Vectorized");
    printf("%s\n\n", ok ? "PASS" : "FAIL");

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    return ok ? 0 : 1;
}
#endif
