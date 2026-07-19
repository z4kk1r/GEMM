// Phase 19: Deep-K WMMA + swizzle + scalar shared-memory stores
//
// Change from Phase 18: replace uint4 vectorized shared-memory stores with
// scalar half-element stores to eliminate ~4x bank conflicts. uint4 stores
// with BK=32 + SKEW_HALF=8 create systematic 4-way bank conflicts because
// the row stride in words is always a multiple of 4, matching the per-thread
// column stride -- only 8 of 32 banks are reachable. Scalar stores spread
// across all banks. GROUP_M=4.
// blocks launched close together in time (and therefore likely co-resident
// or in adjacent waves across the 40 SMs) also share the same B column-tile
// range. This is CUTLASS's GemmIdentityThreadblockSwizzle idea, simplified.
//
// Reverted __launch_bounds__ to the plain single-arg form -- the (NUM_THREADS, 2)
// occupancy hint from Phase 17 showed no clear win (and the benchmark run it
// was tested against was noisy: cuBLAS reference swung ~16% across your last
// three runs, so that result alone isn't strong evidence either way). This
// keeps the swizzle experiment isolated from the occupancy question.
//
// remap_block() below is a group-based rasterization with GROUP_M=4: it's a
// mathematical bijection over the (by, bx) grid *by construction* (worked out
// on paper, not by execution -- see reasoning in the accompanying explanation),
// so it can't create duplicate or missing output tiles. It assumes
// gridDim.y is an exact multiple of GROUP_M, which holds for this problem
// (M=4096, BM=128 -> gridDim.y=32, GROUP_M=8 divides evenly). If you change
// M or BM such that this no longer divides evenly, this mapping needs
// adjusting -- it will not silently corrupt output within this assumption,
// but the assumption must hold.
//
// Still not compiled/executed in my sandbox -- verify against cuBLAS.

#include "../common.cuh"
#include <mma.h>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 128
#define BN 128
#define BK 32

#define SKEW_HALF 8

#define WARP_TILES_M 2
#define WARP_TILES_N 2

#define NUM_THREADS 512

#define GROUP_M 4   // must divide gridDim.y (M / BM) exactly -- see note above

using namespace nvcuda;

constexpr int A_TILE_ELEMS = BM * BK;
constexpr int B_TILE_ELEMS = BK * BN;
constexpr int A_VECS = A_TILE_ELEMS / 8;
constexpr int B_VECS = B_TILE_ELEMS / 8;
constexpr int A_ITERS = (A_VECS + NUM_THREADS - 1) / NUM_THREADS;
constexpr int B_ITERS = (B_VECS + NUM_THREADS - 1) / NUM_THREADS;
static_assert(A_ITERS <= 2 && B_ITERS <= 2,
    "Prefetch register arrays are sized for <=2 iters -- bump them if you grow BM/BN/BK.");

// Remaps CUDA's default raster launch order into groups of GROUP_M=4 rows so
// group_id, as idx_in_group sweeps 0..(GROUP_M*gridDim.x - 1), by takes each
// of GROUP_M values and bx takes each of gridDim.x values exactly once
// (idx_in_group % GROUP_M selects by's offset, idx_in_group / GROUP_M
// selects bx) -- so each group covers its GROUP_M x gridDim.x sub-block of
// the grid exactly once, and groups partition gridDim.y into disjoint
// GROUP_M-row bands, covering the whole grid exactly once overall.
__device__ __forceinline__ void remap_block(int &by, int &bx) {
    int block_id = blockIdx.y * gridDim.x + blockIdx.x;
    int blocks_per_group = GROUP_M * gridDim.x;
    int group_id = block_id / blocks_per_group;
    int idx_in_group = block_id % blocks_per_group;
    by = group_id * GROUP_M + (idx_in_group % GROUP_M);
    bx = idx_in_group / GROUP_M;
}

__device__ __forceinline__ void load_A_tile_to_regs(
    const __half* __restrict__ A, int K, int by, int t, int tid, uint4 (&regs)[A_ITERS]) {
    #pragma unroll
    for (int j = 0; j < A_ITERS; j++) {
        int i = tid + j * NUM_THREADS;
        if (i < A_VECS) {
            int idx = i * 8;
            int r = idx / BK, c = idx % BK;
            regs[j] = reinterpret_cast<const uint4*>(
                &A[(by * BM + r) * K + t * BK + c])[0];
        }
    }
}

__device__ __forceinline__ void load_B_tile_to_regs(
    const __half* __restrict__ B, int N, int bx, int t, int tid, uint4 (&regs)[B_ITERS]) {
    #pragma unroll
    for (int j = 0; j < B_ITERS; j++) {
        int i = tid + j * NUM_THREADS;
        if (i < B_VECS) {
            int idx = i * 8;
            int k = idx / BN, n = idx % BN;
            regs[j] = reinterpret_cast<const uint4*>(
                &B[(t * BK + k) * N + bx * BN + n])[0];
        }
    }
}

// CHANGED: scalar stores instead of a single uint4 store. Profiling showed
// the vectorized store causes a systematic ~4-6x shared-memory bank conflict
// with BK=32 + SKEW_HALF=8 (row stride in words is always a multiple of 4,
// same as the per-thread column stride -- only 8 of 32 banks reachable, for
// any valid SKEW_HALF). The global read stays vectorized (regs is still
// loaded as one uint4) -- only the shared-memory write side is unpacked.
__device__ __forceinline__ void store_A_regs_to_smem(
    const uint4 (&regs)[A_ITERS], __half (&As)[BM][BK + SKEW_HALF], int tid) {
    #pragma unroll
    for (int j = 0; j < A_ITERS; j++) {
        int i = tid + j * NUM_THREADS;
        if (i < A_VECS) {
            int idx = i * 8;
            int r = idx / BK, c = idx % BK;
            const __half* src = reinterpret_cast<const __half*>(&regs[j]);
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                As[r][c + e] = src[e];
            }
        }
    }
}

__device__ __forceinline__ void store_B_regs_to_smem(
    const uint4 (&regs)[B_ITERS], __half (&Bs)[BK][BN + SKEW_HALF], int tid) {
    #pragma unroll
    for (int j = 0; j < B_ITERS; j++) {
        int i = tid + j * NUM_THREADS;
        if (i < B_VECS) {
            int idx = i * 8;
            int k = idx / BN, n = idx % BN;
            const __half* src = reinterpret_cast<const __half*>(&regs[j]);
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                Bs[k][n + e] = src[e];
            }
        }
    }
}

__global__ void __launch_bounds__(NUM_THREADS)
gemm_wmma_scalarstore(int M, int N, int K,
                    const __half* __restrict__ A,
                    const __half* __restrict__ B,
                    float* __restrict__ C) {
    __shared__ alignas(16) __half As[2][BM][BK + SKEW_HALF];
    __shared__ alignas(16) __half Bs[2][BK][BN + SKEW_HALF];

    const int tid = threadIdx.x;
    int by, bx;
    remap_block(by, bx);   // NEW: swizzled tile assignment instead of raw blockIdx
    const int numTiles = K / BK;

    int warpId = tid / 32;
    int warpRow = warpId / (BN / (WMMA_N * WARP_TILES_N));
    int warpCol = warpId % (BN / (WMMA_N * WARP_TILES_N));

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major>
        aFrag0, aFrag1;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major>
        bFrag0, bFrag1;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        cFrag00, cFrag01, cFrag10, cFrag11;

    wmma::fill_fragment(cFrag00, 0.0f);
    wmma::fill_fragment(cFrag01, 0.0f);
    wmma::fill_fragment(cFrag10, 0.0f);
    wmma::fill_fragment(cFrag11, 0.0f);

    uint4 regA[A_ITERS], regB[B_ITERS];

    load_A_tile_to_regs(A, K, by, 0, tid, regA);
    load_B_tile_to_regs(B, N, bx, 0, tid, regB);
    store_A_regs_to_smem(regA, As[0], tid);
    store_B_regs_to_smem(regB, Bs[0], tid);
    __syncthreads();

    int curStage = 0;
    for (int t = 0; t < numTiles; t++) {
        const int nextStage = curStage ^ 1;
        const bool hasNext = (t + 1 < numTiles);

        if (hasNext) {
            load_A_tile_to_regs(A, K, by, t + 1, tid, regA);
            load_B_tile_to_regs(B, N, bx, t + 1, tid, regB);
        }

        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            wmma::load_matrix_sync(aFrag0, &As[curStage][warpRow * WARP_TILES_M * WMMA_M][kk], BK + SKEW_HALF);
            wmma::load_matrix_sync(aFrag1, &As[curStage][(warpRow * WARP_TILES_M + 1) * WMMA_M][kk], BK + SKEW_HALF);
            wmma::load_matrix_sync(bFrag0, &Bs[curStage][kk][warpCol * WARP_TILES_N * WMMA_N], BN + SKEW_HALF);
            wmma::load_matrix_sync(bFrag1, &Bs[curStage][kk][(warpCol * WARP_TILES_N + 1) * WMMA_N], BN + SKEW_HALF);

            wmma::mma_sync(cFrag00, aFrag0, bFrag0, cFrag00);
            wmma::mma_sync(cFrag01, aFrag0, bFrag1, cFrag01);
            wmma::mma_sync(cFrag10, aFrag1, bFrag0, cFrag10);
            wmma::mma_sync(cFrag11, aFrag1, bFrag1, cFrag11);
        }

        if (hasNext) {
            store_A_regs_to_smem(regA, As[nextStage], tid);
            store_B_regs_to_smem(regB, Bs[nextStage], tid);
            __syncthreads();
        }
        curStage = nextStage;
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

float run_wmma_scalarstore(int M, int N, int K,
                         const __half* d_A, const __half* d_B, float* d_C,
                         int warmup_iters = 5, int bench_iters = 20) {
    int warpsPerBlock = (BM * BN) / (WMMA_M * WMMA_N * WARP_TILES_M * WARP_TILES_N);
    int threadsPerBlock = warpsPerBlock * 32;
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);   // unchanged -- same block count, different assignment

    for (int i = 0; i < warmup_iters; i++) {
        gemm_wmma_scalarstore<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CudaTimer timer;
    timer.begin();
    for (int i = 0; i < bench_iters; i++) {
        gemm_wmma_scalarstore<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    return timer.end() / bench_iters;
}

#ifndef BENCHMARK_ALL
int main() {
    int M = 4096, N = 4096, K = 4096;
    srand(42);
    printf("=== Phase 19: Scalar Shared-Mem Stores (bank-conflict fix) (GROUP_M=%d) ===\n", GROUP_M);
    printf("Config: BM=%d, BN=%d, BK=%d, WMMA=%dx%dx%d, warp tiles=%dx%d, SKEW_HALF=%d\n",
           BM, BN, BK, WMMA_M, WMMA_N, WMMA_K, WARP_TILES_M, WARP_TILES_N, SKEW_HALF);
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

    float ms = run_wmma_scalarstore(M, N, K, d_A, d_B, d_C);
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size_c_f32, cudaMemcpyDeviceToHost));
    double gflops = get_gflops(M, N, K, ms);

    printf("Scalar-store WMMA kernel: %.3f ms (%.1f GFLOPS)\n", ms, gflops);
    printf("Performance: %.1f%% of cuBLAS\n", 100.0 * gflops / gflops_ref);

    printf("\nVerification against cuBLAS reference:\n");
    bool ok = verify_result(h_C_ref, h_C, (size_t)M * N, "Scalar-store WMMA", 0.5f);
    printf("%s\n\n", ok ? "PASS" : "FAIL");

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    return ok ? 0 : 1;
}
#endif
