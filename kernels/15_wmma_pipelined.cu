// Phase 15: Double-Buffered (Software-Pipelined) WMMA Tensor Core GEMM
// 2-stage pipeline: prefetch tile (t+1) into regs while computing tile t.
// Completely inlined (no helper functions) to avoid compiler alignment issues.

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
#define NUM_THREADS 512

using namespace nvcuda;

__global__ void __launch_bounds__(NUM_THREADS)
    gemm_wmma_pipelined(int M, int N, int K,
                        const __half *__restrict__ A,
                        const __half *__restrict__ B,
                        float *__restrict__ C)
{
    __shared__ alignas(16) __half As[2][BM][BK + 8];
    __shared__ alignas(16) __half Bs_raw[BK][BN + 1];
    __shared__ alignas(16) __half Bs_trans[2][BN][BK + 8];

    const int tid = threadIdx.x;
    const int by = blockIdx.y, bx = blockIdx.x;
    const int numTiles = K / BK;

    int warpId = tid / 32;
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

    // ---- Prologue: tile 0 -> stage 0 ----
    // Load tile 0 from global into regA, regB
    uint4 regA, regB;
    {
        int idx = tid * 8;
        int r = idx / BK, c = idx % BK;
        if (tid < BM * BK / 8)
        {
            regA = reinterpret_cast<const uint4 *>(
                &A[(by * BM + r) * K + 0 * BK + c])[0];
        }
    }
    {
        int idx = tid * 8;
        int k = idx / BN, n = idx % BN;
        if (tid < BK * BN / 8)
        {
            regB = reinterpret_cast<const uint4 *>(
                &B[(0 * BK + k) * N + bx * BN + n])[0];
        }
    }
    // Store tile 0 from regs to SMEM (scalar halves to avoid alignment issues)
    {
        int idx = tid * 8;
        int r = idx / BK, c = idx % BK;
        if (tid < BM * BK / 8)
        {
            const __half *h = reinterpret_cast<const __half *>(&regA);
            As[0][r][c + 0] = h[0];
            As[0][r][c + 1] = h[1];
            As[0][r][c + 2] = h[2];
            As[0][r][c + 3] = h[3];
            As[0][r][c + 4] = h[4];
            As[0][r][c + 5] = h[5];
            As[0][r][c + 6] = h[6];
            As[0][r][c + 7] = h[7];
        }
    }
    {
        int idx = tid * 8;
        int k = idx / BN, n = idx % BN;
        if (tid < BK * BN / 8)
        {
            const __half *h = reinterpret_cast<const __half *>(&regB);
            Bs_raw[k][n + 0] = h[0];
            Bs_raw[k][n + 1] = h[1];
            Bs_raw[k][n + 2] = h[2];
            Bs_raw[k][n + 3] = h[3];
            Bs_raw[k][n + 4] = h[4];
            Bs_raw[k][n + 5] = h[5];
            Bs_raw[k][n + 6] = h[6];
            Bs_raw[k][n + 7] = h[7];
        }
    }
    __syncthreads();
    // Transpose B tile 0: Bs_raw -> Bs_trans[0]
    for (int i = tid; i < BK * BN; i += NUM_THREADS)
    {
        int k = i / BN, n = i % BN;
        Bs_trans[0][n][k] = Bs_raw[k][n];
    }
    __syncthreads();

    int curStage = 0;
    for (int t = 0; t < numTiles; t++)
    {
        const int nextStage = curStage ^ 1;
        const bool hasNext = (t + 1 < numTiles);

        // Prefetch next tile into regs
        if (hasNext)
        {
            {
                int idx = tid * 8;
                int r = idx / BK, c = idx % BK;
                if (tid < BM * BK / 8)
                {
                    regA = reinterpret_cast<const uint4 *>(
                        &A[(by * BM + r) * K + (t + 1) * BK + c])[0];
                }
            }
            {
                int idx = tid * 8;
                int k = idx / BN, n = idx % BN;
                if (tid < BK * BN / 8)
                {
                    regB = reinterpret_cast<const uint4 *>(
                        &B[((t + 1) * BK + k) * N + bx * BN + n])[0];
                }
            }
        }

        // Compute tile t from curStage
        wmma::load_matrix_sync(aFrag0, &As[curStage][warpRow * WARP_TILES_M * WMMA_M][0], BK + 8);
        wmma::load_matrix_sync(aFrag1, &As[curStage][(warpRow * WARP_TILES_M + 1) * WMMA_M][0], BK + 8);
        wmma::load_matrix_sync(bFrag0, &Bs_trans[curStage][warpCol * WARP_TILES_N * WMMA_N][0], BK + 8);
        wmma::load_matrix_sync(bFrag1, &Bs_trans[curStage][(warpCol * WARP_TILES_N + 1) * WMMA_N][0], BK + 8);

        wmma::mma_sync(cFrag00, aFrag0, bFrag0, cFrag00);
        wmma::mma_sync(cFrag01, aFrag0, bFrag1, cFrag01);
        wmma::mma_sync(cFrag10, aFrag1, bFrag0, cFrag10);
        wmma::mma_sync(cFrag11, aFrag1, bFrag1, cFrag11);

        // Land prefetched tile into nextStage SMEM
        if (hasNext)
        {
            {
                int idx = tid * 8;
                int r = idx / BK, c = idx % BK;
                if (tid < BM * BK / 8)
                {
                    const __half *h = reinterpret_cast<const __half *>(&regA);
                    As[nextStage][r][c + 0] = h[0];
                    As[nextStage][r][c + 1] = h[1];
                    As[nextStage][r][c + 2] = h[2];
                    As[nextStage][r][c + 3] = h[3];
                    As[nextStage][r][c + 4] = h[4];
                    As[nextStage][r][c + 5] = h[5];
                    As[nextStage][r][c + 6] = h[6];
                    As[nextStage][r][c + 7] = h[7];
                }
            }
            {
                int idx = tid * 8;
                int k = idx / BN, n = idx % BN;
                if (tid < BK * BN / 8)
                {
                    const __half *h = reinterpret_cast<const __half *>(&regB);
                    Bs_raw[k][n + 0] = h[0];
                    Bs_raw[k][n + 1] = h[1];
                    Bs_raw[k][n + 2] = h[2];
                    Bs_raw[k][n + 3] = h[3];
                    Bs_raw[k][n + 4] = h[4];
                    Bs_raw[k][n + 5] = h[5];
                    Bs_raw[k][n + 6] = h[6];
                    Bs_raw[k][n + 7] = h[7];
                }
            }
            __syncthreads();
            for (int i = tid; i < BK * BN; i += NUM_THREADS)
            {
                int k = i / BN, n = i % BN;
                Bs_trans[nextStage][n][k] = Bs_raw[k][n];
            }
            __syncthreads();
        }
        curStage = nextStage;
    }

    float *c_base = C + by * BM * N + bx * BN;
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

float run_wmma_pipelined(int M, int N, int K,
                         const __half *d_A, const __half *d_B, float *d_C,
                         int warmup_iters = 5, int bench_iters = 20)
{
    int warpsPerBlock = (BM * BN) / (WMMA_M * WMMA_N * WARP_TILES_M * WARP_TILES_N);
    int threadsPerBlock = warpsPerBlock * 32;
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    for (int i = 0; i < warmup_iters; i++)
    {
        gemm_wmma_pipelined<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaTimer timer;
    timer.begin();
    for (int i = 0; i < bench_iters; i++)
    {
        gemm_wmma_pipelined<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    return timer.end() / bench_iters;
}

#ifndef BENCHMARK_ALL
int main()
{
    int M = 4096, N = 4096, K = 4096;
    srand(42);
    printf("=== Phase 15: Double-Buffered (Pipelined) WMMA Tensor Core ===\n");
    printf("Config: BM=%d, BN=%d, BK=%d, WMMA=%dx%dx%d, warp tiles=%dx%d, "
           "2-stage pipeline, fully inlined\n",
           BM, BN, BK, WMMA_M, WMMA_N, WMMA_K, WARP_TILES_M, WARP_TILES_N);
    print_device_info();

    size_t size_a = (size_t)M * K * sizeof(__half);
    size_t size_b = (size_t)K * N * sizeof(__half);
    size_t size_c_f32 = (size_t)M * N * sizeof(float);

    __half *h_A = (__half *)malloc(size_a);
    __half *h_B = (__half *)malloc(size_b);
    float *h_C = (float *)malloc(size_c_f32);
    float *h_C_ref = (float *)malloc(size_c_f32);

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

    float ms = run_wmma_pipelined(M, N, K, d_A, d_B, d_C);
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_c_f32, cudaMemcpyDeviceToHost));
    double gflops = get_gflops(M, N, K, ms);

    printf("Pipelined WMMA kernel: %.3f ms (%.1f GFLOPS)\n", ms, gflops);
    printf("Performance: %.1f%% of cuBLAS\n", 100.0 * gflops / gflops_ref);

    printf("\nVerification against cuBLAS reference:\n");
    bool ok = verify_result(h_C_ref, h_C, (size_t)M * N, "Pipelined WMMA", 0.5f);
    printf("%s\n\n", ok ? "PASS" : "FAIL");

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_ref);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    return ok ? 0 : 1;
}
#endif
