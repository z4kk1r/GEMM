// Phase 16: Pipelined WMMA, no B transpose
//
// Change from Phase 15: matrix_b fragment is now row_major, loaded directly
// from the same shared buffer B was written into. B is K x N row-major in
// global memory -- that IS row_major layout for a WMMA K x N matrix_b operand,
// so the shared->shared transpose (Bs_raw -> Bs_trans) in Phase 15 was solving
// a problem that didn't exist. Removing it drops one whole shared array, one
// scalar-copy pass over BK*BN elements per tile, and 2 of the 4 __syncthreads()
// per iteration.
//

#include "../common.cuh"
#include <mma.h>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 128
#define BN 128
#define BK 16

#define SKEW_HALF 8

#define WARP_TILES_M 2
#define WARP_TILES_N 2

#define NUM_THREADS 512

using namespace nvcuda;

constexpr int A_TILE_ELEMS = BM * BK;
constexpr int B_TILE_ELEMS = BK * BN;
constexpr int A_VECS = A_TILE_ELEMS / 8;
constexpr int B_VECS = B_TILE_ELEMS / 8;
constexpr int A_ITERS = (A_VECS + NUM_THREADS - 1) / NUM_THREADS;
constexpr int B_ITERS = (B_VECS + NUM_THREADS - 1) / NUM_THREADS;
static_assert(A_ITERS <= 2 && B_ITERS <= 2,
              "Prefetch register arrays are sized for <=2 iters -- bump them if you grow BM/BN/BK.");

__device__ __forceinline__ void load_A_tile_to_regs(
    const __half *__restrict__ A, int K, int by, int t, int tid, uint4 (&regs)[A_ITERS])
{
#pragma unroll
    for (int j = 0; j < A_ITERS; j++)
    {
        int i = tid + j * NUM_THREADS;
        if (i < A_VECS)
        {
            int idx = i * 8;
            int r = idx / BK, c = idx % BK;
            regs[j] = reinterpret_cast<const uint4 *>(
                &A[(by * BM + r) * K + t * BK + c])[0];
        }
    }
}

// B tile loaded in its natural (k, n) row-major order -- no relayout needed.
__device__ __forceinline__ void load_B_tile_to_regs(
    const __half *__restrict__ B, int N, int bx, int t, int tid, uint4 (&regs)[B_ITERS])
{
#pragma unroll
    for (int j = 0; j < B_ITERS; j++)
    {
        int i = tid + j * NUM_THREADS;
        if (i < B_VECS)
        {
            int idx = i * 8;
            int k = idx / BN, n = idx % BN;
            regs[j] = reinterpret_cast<const uint4 *>(
                &B[(t * BK + k) * N + bx * BN + n])[0];
        }
    }
}

__device__ __forceinline__ void store_A_regs_to_smem(
    const uint4 (&regs)[A_ITERS], __half (&As)[BM][BK + SKEW_HALF], int tid)
{
#pragma unroll
    for (int j = 0; j < A_ITERS; j++)
    {
        int i = tid + j * NUM_THREADS;
        if (i < A_VECS)
        {
            int idx = i * 8;
            int r = idx / BK, c = idx % BK;
            reinterpret_cast<uint4 *>(&As[r][c])[0] = regs[j];
        }
    }
}

__device__ __forceinline__ void store_B_regs_to_smem(
    const uint4 (&regs)[B_ITERS], __half (&Bs)[BK][BN + SKEW_HALF], int tid)
{
#pragma unroll
    for (int j = 0; j < B_ITERS; j++)
    {
        int i = tid + j * NUM_THREADS;
        if (i < B_VECS)
        {
            int idx = i * 8;
            int k = idx / BN, n = idx % BN;
            reinterpret_cast<uint4 *>(&Bs[k][n])[0] = regs[j];
        }
    }
}

__global__ void __launch_bounds__(NUM_THREADS)
    gemm_wmma_notranspose(int M, int N, int K,
                          const __half *__restrict__ A,
                          const __half *__restrict__ B,
                          float *__restrict__ C)
{
    __shared__ alignas(16) __half As[2][BM][BK + SKEW_HALF];
    __shared__ alignas(16) __half Bs[2][BK][BN + SKEW_HALF]; // single buffer per stage now -- no Bs_trans

    const int tid = threadIdx.x;
    const int by = blockIdx.y, bx = blockIdx.x;
    const int numTiles = K / BK;

    int warpId = tid / 32;
    int warpRow = warpId / (BN / (WMMA_N * WARP_TILES_N));
    int warpCol = warpId % (BN / (WMMA_N * WARP_TILES_N));

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major>
        aFrag0, aFrag1;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major>
        bFrag0, bFrag1; // <-- row_major now, loaded straight from Bs, no transpose
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        cFrag00, cFrag01, cFrag10, cFrag11;

    wmma::fill_fragment(cFrag00, 0.0f);
    wmma::fill_fragment(cFrag01, 0.0f);
    wmma::fill_fragment(cFrag10, 0.0f);
    wmma::fill_fragment(cFrag11, 0.0f);

    uint4 regA[A_ITERS], regB[B_ITERS];

    // ---- Prologue: single sync now, not two -- nothing to transpose. ----
    load_A_tile_to_regs(A, K, by, 0, tid, regA);
    load_B_tile_to_regs(B, N, bx, 0, tid, regB);
    store_A_regs_to_smem(regA, As[0], tid);
    store_B_regs_to_smem(regB, Bs[0], tid);
    __syncthreads();

    int curStage = 0;
    for (int t = 0; t < numTiles; t++)
    {
        const int nextStage = curStage ^ 1;
        const bool hasNext = (t + 1 < numTiles);

        if (hasNext)
        {
            load_A_tile_to_regs(A, K, by, t + 1, tid, regA);
            load_B_tile_to_regs(B, N, bx, t + 1, tid, regB);
        }

        wmma::load_matrix_sync(aFrag0, &As[curStage][warpRow * WARP_TILES_M * WMMA_M][0], BK + SKEW_HALF);
        wmma::load_matrix_sync(aFrag1, &As[curStage][(warpRow * WARP_TILES_M + 1) * WMMA_M][0], BK + SKEW_HALF);
        // B fragment: K x N chunk starting at row 0 (k=0), column offset for this warp's N-tile.
        // ldm is the stride between k-rows, i.e. BN + SKEW_HALF.
        wmma::load_matrix_sync(bFrag0, &Bs[curStage][0][warpCol * WARP_TILES_N * WMMA_N], BN + SKEW_HALF);
        wmma::load_matrix_sync(bFrag1, &Bs[curStage][0][(warpCol * WARP_TILES_N + 1) * WMMA_N], BN + SKEW_HALF);

        wmma::mma_sync(cFrag00, aFrag0, bFrag0, cFrag00);
        wmma::mma_sync(cFrag01, aFrag0, bFrag1, cFrag01);
        wmma::mma_sync(cFrag10, aFrag1, bFrag0, cFrag10);
        wmma::mma_sync(cFrag11, aFrag1, bFrag1, cFrag11);

        if (hasNext)
        {
            store_A_regs_to_smem(regA, As[nextStage], tid);
            store_B_regs_to_smem(regB, Bs[nextStage], tid);
            __syncthreads(); // one sync, not two -- this is the whole saving vs Phase 15
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

float run_wmma_notranspose(int M, int N, int K,
                           const __half *d_A, const __half *d_B, float *d_C,
                           int warmup_iters = 5, int bench_iters = 20)
{
    int warpsPerBlock = (BM * BN) / (WMMA_M * WMMA_N * WARP_TILES_M * WARP_TILES_N);
    int threadsPerBlock = warpsPerBlock * 32;
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    for (int i = 0; i < warmup_iters; i++)
    {
        gemm_wmma_notranspose<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaTimer timer;
    timer.begin();
    for (int i = 0; i < bench_iters; i++)
    {
        gemm_wmma_notranspose<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    return timer.end() / bench_iters;
}

#ifndef BENCHMARK_ALL
int main()
{
    int M = 4096, N = 4096, K = 4096;
    srand(42);
    printf("=== Phase 16: Pipelined WMMA, No B Transpose ===\n");
    printf("Config: BM=%d, BN=%d, BK=%d, WMMA=%dx%dx%d, warp tiles=%dx%d, SKEW_HALF=%d\n",
           BM, BN, BK, WMMA_M, WMMA_N, WMMA_K, WARP_TILES_M, WARP_TILES_N, SKEW_HALF);
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

    float ms = run_wmma_notranspose(M, N, K, d_A, d_B, d_C);
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_c_f32, cudaMemcpyDeviceToHost));
    double gflops = get_gflops(M, N, K, ms);

    printf("No-transpose pipelined WMMA kernel: %.3f ms (%.1f GFLOPS)\n", ms, gflops);
    printf("Performance: %.1f%% of cuBLAS\n", 100.0 * gflops / gflops_ref);

    printf("\nVerification against cuBLAS reference:\n");
    bool ok = verify_result(h_C_ref, h_C, (size_t)M * N, "No-transpose pipelined WMMA", 0.5f);
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
