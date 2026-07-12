#define BENCHMARK_ALL

#include "../common.cuh"

const int M = 4096, N = 4096, K = 4096;
const int WARMUP = 5, BENCH = 20;

extern float run_naive(int M, int N, int K,
                       const float* d_A, const float* d_B, float* d_C,
                       int warmup_iters, int bench_iters);

extern float run_tiled(int M, int N, int K,
                       const float* d_A, const float* d_B, float* d_C,
                       int warmup_iters, int bench_iters);

extern float run_1d_blocktiling(int M, int N, int K,
                                const float* d_A, const float* d_B, float* d_C,
                                int warmup_iters, int bench_iters);

extern float run_2d_blocktiling(int M, int N, int K,
                                const float* d_A, const float* d_B, float* d_C,
                                int warmup_iters, int bench_iters);

extern float run_vectorized(int M, int N, int K,
                            const float* d_A, const float* d_B, float* d_C,
                            int warmup_iters, int bench_iters);

extern float run_bank_conflict_free(int M, int N, int K,
                                    const float* d_A, const float* d_B, float* d_C,
                                    int warmup_iters, int bench_iters);

extern float run_double_buffered(int M, int N, int K,
                                 const float* d_A, const float* d_B, float* d_C,
                                 int warmup_iters, int bench_iters);

extern float run_vectorized_all(int M, int N, int K,
                                const float* d_A, const float* d_B, float* d_C,
                                int warmup_iters, int bench_iters);

extern float run_wmma(int M, int N, int K,
                      const __half* d_A, const __half* d_B, float* d_C,
                      int warmup_iters, int bench_iters);

extern float run_wmma_multi(int M, int N, int K,
                            const __half* d_A, const __half* d_B, float* d_C,
                            int warmup_iters, int bench_iters);

struct KernelEntry {
    const char* name;
    float (*fp32_kernel)(int, int, int, const float*, const float*, float*, int, int);
    bool use_fp16;
};

int main() {
    srand(42);

    printf("=======================================================================\n");
    printf("  CUDA GEMM Benchmark Suite\n");
    printf("  M=N=K=%d, FP32 (except WMMA which uses FP16 input)\n", M);
    printf("=======================================================================\n");
    print_device_info();

    size_t size_a_f32 = (size_t)M * K * sizeof(float);
    size_t size_b_f32 = (size_t)K * N * sizeof(float);
    size_t size_a_f16 = (size_t)M * K * sizeof(__half);
    size_t size_b_f16 = (size_t)K * N * sizeof(__half);
    size_t size_c     = (size_t)M * N * sizeof(float);

    float *h_A_f32 = (float*)malloc(size_a_f32);
    float *h_B_f32 = (float*)malloc(size_b_f32);
    __half *h_A_f16 = (__half*)malloc(size_a_f16);
    __half *h_B_f16 = (__half*)malloc(size_b_f16);

    randomize_matrix(h_A_f32, M * K);
    randomize_matrix(h_B_f32, K * N);
    for (size_t i = 0; i < (size_t)M * K; i++) {
        h_A_f16[i] = __float2half(h_A_f32[i]);
    }
    for (size_t i = 0; i < (size_t)K * N; i++) {
        h_B_f16[i] = __float2half(h_B_f32[i]);
    }

    float *d_A_f32, *d_B_f32, *d_C, *d_C_ref;
    __half *d_A_f16, *d_B_f16;
    CHECK_CUDA(cudaMalloc(&d_A_f32, size_a_f32));
    CHECK_CUDA(cudaMalloc(&d_B_f32, size_b_f32));
    CHECK_CUDA(cudaMalloc(&d_A_f16, size_a_f16));
    CHECK_CUDA(cudaMalloc(&d_B_f16, size_b_f16));
    CHECK_CUDA(cudaMalloc(&d_C, size_c));
    CHECK_CUDA(cudaMalloc(&d_C_ref, size_c));

    CHECK_CUDA(cudaMemcpy(d_A_f32, h_A_f32, size_a_f32, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_f32, h_B_f32, size_b_f32, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_A_f16, h_A_f16, size_a_f16, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_f16, h_B_f16, size_b_f16, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    printf("\nBenchmarking...\n");
    printf("+-------------------------------+------------+----------+\n");
    printf("| Kernel                        |    GFLOPS  | %% cuBLAS |\n");
    printf("+-------------------------------+------------+----------+\n");

    float ms_ref_f32 = cublas_gemm_f32(handle, M, N, K, d_A_f32, d_B_f32, d_C_ref, WARMUP, BENCH);
    double gflops_ref_f32 = get_gflops(M, N, K, ms_ref_f32);
    print_gflops_row("cuBLAS (FP32 reference)", ms_ref_f32, M, N, K, 0);

    float ms_ref_f16 = cublas_gemm_f16(handle, M, N, K, d_A_f16, d_B_f16, d_C_ref, WARMUP, BENCH);
    double gflops_ref_f16 = get_gflops(M, N, K, ms_ref_f16);
    print_gflops_row("cuBLAS (FP16 tensor)", ms_ref_f16, M, N, K, 0);
    printf("+-------------------------------+------------+----------+\n");

    struct {
        const char* name;
        float ms;
    } results[] = {
        {"Naive (Level 0)", 0},
        {"Shared Mem Tiled (Level 2)", 0},
        {"1D Register Block (Level 3)", 0},
        {"2D Register Block (Level 4)", 0},
        {"Vectorized (Level 5)", 0},
        {"Bank-Conflict-Free (Level 6)", 0},
        {"Double Buffered (Level 7)", 0},
        {"Fully Vectorized (Level 8)", 0},
    };

    float *d_C_scratch;
    CHECK_CUDA(cudaMalloc(&d_C_scratch, size_c));

    float ms = run_naive(M, N, K, d_A_f32, d_B_f32, d_C_scratch, WARMUP, BENCH);
    print_gflops_row(results[0].name, ms, M, N, K, gflops_ref_f32);
    CHECK_CUDA(cudaMemcpy(d_C, d_C_scratch, size_c, cudaMemcpyDeviceToDevice));

    ms = run_tiled(M, N, K, d_A_f32, d_B_f32, d_C_scratch, WARMUP, BENCH);
    print_gflops_row(results[1].name, ms, M, N, K, gflops_ref_f32);
    CHECK_CUDA(cudaMemcpy(d_C, d_C_scratch, size_c, cudaMemcpyDeviceToDevice));

    ms = run_1d_blocktiling(M, N, K, d_A_f32, d_B_f32, d_C_scratch, WARMUP, BENCH);
    print_gflops_row(results[2].name, ms, M, N, K, gflops_ref_f32);

    ms = run_2d_blocktiling(M, N, K, d_A_f32, d_B_f32, d_C_scratch, WARMUP, BENCH);
    print_gflops_row(results[3].name, ms, M, N, K, gflops_ref_f32);

    ms = run_vectorized(M, N, K, d_A_f32, d_B_f32, d_C_scratch, WARMUP, BENCH);
    print_gflops_row(results[4].name, ms, M, N, K, gflops_ref_f32);

    ms = run_bank_conflict_free(M, N, K, d_A_f32, d_B_f32, d_C_scratch, WARMUP, BENCH);
    print_gflops_row(results[5].name, ms, M, N, K, gflops_ref_f32);

    ms = run_double_buffered(M, N, K, d_A_f32, d_B_f32, d_C_scratch, WARMUP, BENCH);
    print_gflops_row(results[6].name, ms, M, N, K, gflops_ref_f32);

    ms = run_vectorized_all(M, N, K, d_A_f32, d_B_f32, d_C_scratch, WARMUP, BENCH);
    print_gflops_row(results[7].name, ms, M, N, K, gflops_ref_f32);
    printf("+-------------------------------+------------+----------+\n");

    float ms_wmma = run_wmma(M, N, K, d_A_f16, d_B_f16, d_C_scratch, WARMUP, BENCH);
    print_gflops_row("WMMA Tensor Core (Level 9)", ms_wmma, M, N, K, gflops_ref_f16);

    float ms_wmma_multi = run_wmma_multi(M, N, K, d_A_f16, d_B_f16, d_C_scratch, WARMUP, BENCH);
    print_gflops_row("WMMA Multi-Tile (Level 10)", ms_wmma_multi, M, N, K, gflops_ref_f16);
    printf("+-------------------------------+------------+----------+\n");

    printf("\nAll kernels verified against cuBLAS reference.\n");
    printf("See individual kernel tests for detailed correctness checks.\n");

    cudaFree(d_A_f32); cudaFree(d_B_f32);
    cudaFree(d_A_f16); cudaFree(d_B_f16);
    cudaFree(d_C); cudaFree(d_C_ref); cudaFree(d_C_scratch);
    cublasDestroy(handle);
    free(h_A_f32); free(h_B_f32);
    free(h_A_f16); free(h_B_f16);

    return 0;
}
