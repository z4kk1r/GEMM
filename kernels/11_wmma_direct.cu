#include "../common.cuh"
#include <mma.h>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 64
#define BN 128
#define BK 16

#define WARP_TILES_M 2
#define WARP_TILES_N 2

using namespace nvcuda;

__global__ void __launch_bounds__(256)
gemm_wmma_direct(int M, int N, int K,
                 const __half* __restrict__ A,
                 const __half* __restrict__ B,
                 float* __restrict__ C) {
    __shared__ __half As[BM][BK];
    __shared__ __half Bs[BN][BK];

    int warpId = threadIdx.x / 32;
    int warpCols = BN / (WMMA_N * WARP_TILES_N);
    int warpRow = warpId / warpCols;
    int warpCol = warpId % warpCols;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major>
        aFrags[WARP_TILES_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major>
        bFrags[WARP_TILES_N];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        cFrags[WARP_TILES_M][WARP_TILES_N];

    #pragma unroll
    for (int m = 0; m < WARP_TILES_M; m++)
        #pragma unroll
        for (int n = 0; n < WARP_TILES_N; n++)
            wmma::fill_fragment(cFrags[m][n], 0.0f);

    int numTiles = K / BK;
    int tid = threadIdx.x;
    int numThreads = blockDim.x;
    int bx = blockIdx.x, by = blockIdx.y;

    for (int t = 0; t < numTiles; t++) {
        for (int i = tid; i < BM * BK / 8; i += numThreads) {
            int idx = i * 8;
            int r = idx / BK;
            int c = idx % BK;
            reinterpret_cast<uint4*>(&As[r][c])[0] =
                reinterpret_cast<const uint4*>(&A[(by * BM + r) * K + t * BK + c])[0];
        }

        for (int i = tid; i < BK * BN / 4; i += numThreads) {
            int idx = i * 4;
            int k = idx / BN;
            int j = idx % BN;
            uint2 data = reinterpret_cast<const uint2*>(&B[(t * BK + k) * N + bx * BN + j])[0];
            Bs[j][k] = ((__half*)&data)[0];
            Bs[j+1][k] = ((__half*)&data)[1];
            Bs[j+2][k] = ((__half*)&data)[2];
            Bs[j+3][k] = ((__half*)&data)[3];
        }
        __syncthreads();

        #pragma unroll
        for (int m = 0; m < WARP_TILES_M; m++)
            wmma::load_matrix_sync(aFrags[m],
                &As[(warpRow * WARP_TILES_M + m) * WMMA_M][0], BK);

        #pragma unroll
        for (int n = 0; n < WARP_TILES_N; n++)
            wmma::load_matrix_sync(bFrags[n],
                &Bs[(warpCol * WARP_TILES_N + n) * WMMA_N][0], BK);

        #pragma unroll
        for (int m = 0; m < WARP_TILES_M; m++)
            #pragma unroll
            for (int n = 0; n < WARP_TILES_N; n++)
                wmma::mma_sync(cFrags[m][n], aFrags[m], bFrags[n], cFrags[m][n]);

        __syncthreads();
    }

    float* c_base = C + by * BM * N + bx * BN;
    #pragma unroll
    for (int m = 0; m < WARP_TILES_M; m++) {
        #pragma unroll
        for (int n = 0; n < WARP_TILES_N; n++) {
            float* c_warp_start = c_base +
                (warpRow * WARP_TILES_M + m) * WMMA_M * N +
                (warpCol * WARP_TILES_N + n) * WMMA_N;
            wmma::store_matrix_sync(c_warp_start, cFrags[m][n], N, wmma::mem_row_major);
        }
    }
}

float run_wmma_direct(int M, int N, int K,
                      const __half* d_A, const __half* d_B, float* d_C,
                      int warmup_iters = 5, int bench_iters = 20) {
    int warpsPerBlock = (BM * BN) / (WMMA_M * WMMA_N * WARP_TILES_M * WARP_TILES_N);
    int threadsPerBlock = warpsPerBlock * 32;
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    for (int i = 0; i < warmup_iters; i++) {
        gemm_wmma_direct<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaTimer timer;
    timer.begin();
    for (int i = 0; i < bench_iters; i++) {
        gemm_wmma_direct<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    return timer.end() / bench_iters;
}

#ifndef BENCHMARK_ALL
int main() {
    int M = 4096, N = 4096, K = 4096;

    srand(42);
    printf("=== Phase 11: Direct-Load WMMA Tensor Core ===\n");
    printf("Config: BM=%d, BN=%d, BK=%d, WMMA=%dx%dx%d, warp tiles=%dx%d, direct B col-major load\n",
           BM, BN, BK, WMMA_M, WMMA_N, WMMA_K, WARP_TILES_M, WARP_TILES_N);
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
    float *d_C, *d_C_ref;
    CHECK_CUDA(cudaMalloc(&d_A, size_a));
    CHECK_CUDA(cudaMalloc(&d_B, size_b));
    CHECK_CUDA(cudaMalloc(&d_C, size_c_f32));
    CHECK_CUDA(cudaMalloc(&d_C_ref, size_c_f32));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float ms_ref = cublas_gemm_f16(handle, M, N, K, d_A, d_B, d_C_ref);
    CHECK_CUDA(cudaMemcpy(h_C_ref, d_C_ref, size_c_f32, cudaMemcpyDeviceToHost));
    double gflops_ref = get_gflops(M, N, K, ms_ref);
    printf("cuBLAS (FP16 tensor core): %.3f ms (%.1f GFLOPS)\n\n", ms_ref, gflops_ref);

    float ms = run_wmma_direct(M, N, K, d_A, d_B, d_C);
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_c_f32, cudaMemcpyDeviceToHost));
    double gflops = get_gflops(M, N, K, ms);

    printf("Direct-load WMMA kernel: %.3f ms (%.1f GFLOPS)\n", ms, gflops);
    printf("Performance: %.1f%% of cuBLAS\n", 100.0 * gflops / gflops_ref);

    printf("\nVerification against cuBLAS reference:\n");
    bool ok = verify_result(h_C_ref, h_C, (size_t)M * N, "Direct-Load WMMA", 0.5f);
    printf("%s\n\n", ok ? "PASS" : "FAIL");

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    return ok ? 0 : 1;
}
#endif
