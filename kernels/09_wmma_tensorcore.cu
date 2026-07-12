#include "../common.cuh"
#include <mma.h>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 64
#define BN 128
#define BK 16

using namespace nvcuda;

__global__ void gemm_wmma(int M, int N, int K,
                          const __half* A, const __half* B, float* C) {
    __shared__ __half As[BM][BK];
    __shared__ __half Bs_raw[BK][BN];
    __shared__ __half Bs_trans[BN][BK];

    int warpId = threadIdx.x / 32;
    int warpRow = warpId / (BN / WMMA_N);
    int warpCol = warpId % (BN / WMMA_N);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> bFrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;

    wmma::fill_fragment(cFrag, 0.0f);

    int numTiles = K / BK;
    int tid = threadIdx.x;
    int numThreads = blockDim.x;
    int bx = blockIdx.x, by = blockIdx.y;

    for (int t = 0; t < numTiles; t++) {
        for (int i = tid; i < BM * BK / 4; i += numThreads) {
            int idx = i * 4;
            int r = idx / BK;
            int c = idx % BK;
            reinterpret_cast<float2*>(&As[r][c])[0] =
                reinterpret_cast<const float2*>(
                    &A[(by * BM + r) * K + t * BK + c])[0];
        }

        for (int i = tid; i < BK * BN; i += numThreads) {
            int k = i / BN;
            int n = i % BN;
            Bs_raw[k][n] = B[(t * BK + k) * N + bx * BN + n];
        }
        __syncthreads();

        for (int i = tid; i < BK * BN; i += numThreads) {
            int k = i / BN;
            int n = i % BN;
            Bs_trans[n][k] = Bs_raw[k][n];
        }
        __syncthreads();

        wmma::load_matrix_sync(aFrag, &As[warpRow * WMMA_M][0], BK);
        wmma::load_matrix_sync(bFrag, &Bs_trans[warpCol * WMMA_N][0], BK);
        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        __syncthreads();
    }

    float* c_warp_start = C +
        (by * BM + warpRow * WMMA_M) * N +
        bx * BN + warpCol * WMMA_N;
    wmma::store_matrix_sync(c_warp_start, cFrag, N, wmma::mem_row_major);
}

float run_wmma(int M, int N, int K,
               const __half* d_A, const __half* d_B, float* d_C,
               int warmup_iters = 5, int bench_iters = 20) {
    int warpsPerBlock = (BM * BN) / (WMMA_M * WMMA_N);
    int threadsPerBlock = warpsPerBlock * 32;
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    for (int i = 0; i < warmup_iters; i++) {
        gemm_wmma<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaTimer timer;
    timer.begin();
    for (int i = 0; i < bench_iters; i++) {
        gemm_wmma<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    return timer.end() / bench_iters;
}

#ifndef BENCHMARK_ALL
int main() {
    int M = 4096, N = 4096, K = 4096;

    srand(42);
    printf("=== Phase 9: Tensor Core GEMM via WMMA ===\n");
    printf("Config: BM=%d, BN=%d, BK=%d, WMMA=%dx%dx%d\n", BM, BN, BK, WMMA_M, WMMA_N, WMMA_K);
    print_device_info();

    size_t size_a = (size_t)M * K * sizeof(__half);
    size_t size_b = (size_t)K * N * sizeof(__half);
    size_t size_c_f32 = (size_t)M * N * sizeof(float);

    __half *h_A = (__half*)malloc(size_a);
    __half *h_B = (__half*)malloc(size_b);
    float *h_C = (float*)malloc(size_c_f32);
    float *h_C_ref = (float*)malloc(size_c_f32);

    randomize_matrix_half(h_A, M * K);
    randomize_matrix_half(h_B, K * N);

    __half *d_A, *d_B;
    float *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_a));
    CHECK_CUDA(cudaMalloc(&d_B, size_b));
    CHECK_CUDA(cudaMalloc(&d_C, size_c_f32));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice));

    float *d_C_ref;
    CHECK_CUDA(cudaMalloc(&d_C_ref, size_c_f32));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float ms_ref = cublas_gemm_f16(handle, M, N, K, d_A, d_B, d_C_ref);
    CHECK_CUDA(cudaMemcpy(h_C_ref, d_C_ref, size_c_f32, cudaMemcpyDeviceToHost));
    double gflops_ref = get_gflops(M, N, K, ms_ref);
    printf("cuBLAS (FP16 tensor core): %.3f ms (%.1f GFLOPS)\n\n", ms_ref, gflops_ref);

    float ms = run_wmma(M, N, K, d_A, d_B, d_C);
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_c_f32, cudaMemcpyDeviceToHost));
    double gflops = get_gflops(M, N, K, ms);

    printf("WMMA tensor core kernel: %.3f ms (%.1f GFLOPS)\n", ms, gflops);
    printf("Performance: %.1f%% of cuBLAS\n", 100.0 * gflops / gflops_ref);

    printf("\nVerification against cuBLAS reference:\n");
    bool ok = verify_result(h_C_ref, h_C, (size_t)M * N, "WMMA Tensor Core", 0.5f);
    printf("%s\n\n", ok ? "PASS" : "FAIL");

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    return ok ? 0 : 1;
}
#endif
