#include "../common.cuh"
#include <mma.h>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 128
#define BN 128
#define BK 16

#define WARP_TILES_M 2
#define WARP_TILES_N 2

using namespace nvcuda;

__global__ void __launch_bounds__(512)
gemm_wmma_padded(int M, int N, int K,
                 const __half* __restrict__ A,
                 const __half* __restrict__ B,
                 float* __restrict__ C) {
    __shared__ alignas(16) __half As[BM][BK + 8];
    __shared__ alignas(16) __half Bs_raw[BK][BN + 1];
    __shared__ alignas(16) __half Bs_trans[BN][BK + 8];

    int warpId = threadIdx.x / 32;
    int warpRow = warpId / (BN / (WMMA_N * WARP_TILES_N));
    int warpCol = warpId % (BN / (WMMA_N * WARP_TILES_N));

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major>
        aFrag0, aFrag1;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major>
        bFrag0, bFrag1;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        cFrag00, cFrag01, cFrag10, cFrag11;

    wmma::fill_fragment(cFrag00, 0.0f);
    wmma::fill_fragment(cFrag01, 0.0f);
    wmma::fill_fragment(cFrag10, 0.0f);
    wmma::fill_fragment(cFrag11, 0.0f);

    int numTiles = K / BK;
    int tid = threadIdx.x;
    int numThreads = blockDim.x;
    int bx = blockIdx.x, by = blockIdx.y;

    for (int t = 0; t < numTiles; t++) {
        #pragma unroll
        for (int i = tid; i < BM * BK / 8; i += numThreads) {
            int idx = i * 8;
            int r = idx / BK;
            int c = idx % BK;
            uint4 regA = reinterpret_cast<const uint4*>(
                &A[(by * BM + r) * K + t * BK + c])[0];
            __half* h = reinterpret_cast<__half*>(&regA);
            As[r][c + 0] = h[0]; As[r][c + 1] = h[1];
            As[r][c + 2] = h[2]; As[r][c + 3] = h[3];
            As[r][c + 4] = h[4]; As[r][c + 5] = h[5];
            As[r][c + 6] = h[6]; As[r][c + 7] = h[7];
        }

        #pragma unroll
        for (int i = tid; i < BK * BN / 8; i += numThreads) {
            int idx = i * 8;
            int k = idx / BN;
            int n = idx % BN;
            uint4 regB = reinterpret_cast<const uint4*>(
                &B[(t * BK + k) * N + bx * BN + n])[0];
            __half* h = reinterpret_cast<__half*>(&regB);
            Bs_raw[k][n + 0] = h[0]; Bs_raw[k][n + 1] = h[1];
            Bs_raw[k][n + 2] = h[2]; Bs_raw[k][n + 3] = h[3];
            Bs_raw[k][n + 4] = h[4]; Bs_raw[k][n + 5] = h[5];
            Bs_raw[k][n + 6] = h[6]; Bs_raw[k][n + 7] = h[7];
        }
        __syncthreads();

        #pragma unroll
        for (int i = tid; i < BK * BN; i += numThreads) {
            int k = i / BN;
            int n = i % BN;
            Bs_trans[n][k] = Bs_raw[k][n];
        }
        __syncthreads();

        wmma::load_matrix_sync(aFrag0, &As[warpRow * WARP_TILES_M * WMMA_M][0], BK + 8);
        wmma::load_matrix_sync(aFrag1, &As[(warpRow * WARP_TILES_M + 1) * WMMA_M][0], BK + 8);
        wmma::load_matrix_sync(bFrag0, &Bs_trans[warpCol * WARP_TILES_N * WMMA_N][0], BK + 8);
        wmma::load_matrix_sync(bFrag1, &Bs_trans[(warpCol * WARP_TILES_N + 1) * WMMA_N][0], BK + 8);

        wmma::mma_sync(cFrag00, aFrag0, bFrag0, cFrag00);
        wmma::mma_sync(cFrag01, aFrag0, bFrag1, cFrag01);
        wmma::mma_sync(cFrag10, aFrag1, bFrag0, cFrag10);
        wmma::mma_sync(cFrag11, aFrag1, bFrag1, cFrag11);

        __syncthreads();
    }

    float* c_base = C + by * BM * N + bx * BN;
    wmma::store_matrix_sync(
        c_base + (warpRow * WARP_TILES_M * WMMA_M) * N + (warpCol * WARP_TILES_N * WMMA_N),
        cFrag00, N, wmma::mem_row_major);
    wmma::store_matrix_sync(
        c_base + (warpRow * WARP_TILES_M * WMMA_M) * N + ((warpCol * WARP_TILES_N + 1) * WMMA_N),
        cFrag01, N, wmma::mem_row_major);
    wmma::store_matrix_sync(
        c_base + ((warpRow * WARP_TILES_M + 1) * WMMA_M) * N + (warpCol * WARP_TILES_N * WMMA_N),
        cFrag10, N, wmma::mem_row_major);
    wmma::store_matrix_sync(
        c_base + ((warpRow * WARP_TILES_M + 1) * WMMA_M) * N + ((warpCol * WARP_TILES_N + 1) * WMMA_N),
        cFrag11, N, wmma::mem_row_major);
}

float run_wmma_padded(int M, int N, int K,
                      const __half* d_A, const __half* d_B, float* d_C,
                      int warmup_iters = 5, int bench_iters = 20) {
    int warpsPerBlock = (BM * BN) / (WMMA_M * WMMA_N * WARP_TILES_M * WARP_TILES_N);
    int threadsPerBlock = warpsPerBlock * 32;
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    for (int i = 0; i < warmup_iters; i++) {
        gemm_wmma_padded<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaTimer timer;
    timer.begin();
    for (int i = 0; i < bench_iters; i++) {
        gemm_wmma_padded<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    return timer.end() / bench_iters;
}

#ifndef BENCHMARK_ALL
int main() {
    int M = 4096, N = 4096, K = 4096;
    srand(42);
    printf("=== Phase 14: Padded WMMA Tensor Core ===\n");
    printf("Config: BM=%d, BN=%d, BK=%d, WMMA=%dx%dx%d, warp tiles=%dx%d, "
           "BK+8 padding on As/Bs_trans, BN+1 padding on Bs_raw\n",
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

    float ms = run_wmma_padded(M, N, K, d_A, d_B, d_C);
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_c_f32, cudaMemcpyDeviceToHost));
    double gflops = get_gflops(M, N, K, ms);

    printf("Padded WMMA kernel: %.3f ms (%.1f GFLOPS)\n", ms, gflops);
    printf("Performance: %.1f%% of cuBLAS\n", 100.0 * gflops / gflops_ref);

    printf("\nVerification against cuBLAS reference:\n");
    bool ok = verify_result(h_C_ref, h_C, (size_t)M * N, "Padded WMMA", 0.5f);
    printf("%s\n\n", ok ? "PASS" : "FAIL");

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    return ok ? 0 : 1;
}
#endif
