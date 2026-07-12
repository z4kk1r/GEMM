#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error at %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

struct CudaTimer {
    cudaEvent_t start, stop;
    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void begin() {
        cudaEventRecord(start);
    }
    float end() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

inline void randomize_matrix(float* mat, size_t n) {
    for (size_t i = 0; i < n; i++) {
        mat[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

inline void randomize_matrix_half(__half* mat, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float val = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        mat[i] = __float2half(val);
    }
}

inline bool verify_result(const float* reference, const float* result,
                          size_t n, const char* label, float eps = 1e-3f) {
    double max_diff = 0.0;
    double avg_diff = 0.0;
    int bad_count = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = fabs((double)reference[i] - (double)result[i]);
        double ref_abs = fabs((double)reference[i]);
        double rel_diff = (ref_abs > 1.0f) ? diff / ref_abs : diff;
        if (rel_diff > eps) {
            if (bad_count < 5) {
                printf("  MISMATCH[%zu]: ref=%f, got=%f, diff=%e\n",
                       i, (double)reference[i], (double)result[i], diff);
            }
            bad_count++;
        }
        max_diff = fmax(max_diff, diff);
        avg_diff += diff;
    }
    avg_diff /= (double)n;
    printf("  %s: max_diff=%.2e, avg_diff=%.2e, bad=%d/%zu\n",
           label, max_diff, avg_diff, bad_count, n);
    return bad_count == 0;
}

inline float cublas_gemm_f32(cublasHandle_t handle, int M, int N, int K,
                             const float* A, const float* B, float* C,
                             int warmup_iters = 5, int bench_iters = 20) {
    float alpha = 1.0f, beta = 0.0f;
    CudaTimer timer;

    for (int i = 0; i < warmup_iters; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, B, N, A, K, &beta, C, N));
    }
    cudaDeviceSynchronize();

    timer.begin();
    for (int i = 0; i < bench_iters; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, B, N, A, K, &beta, C, N));
    }
    cudaDeviceSynchronize();
    return timer.end() / bench_iters;
}

inline float cublas_gemm_f16(cublasHandle_t handle, int M, int N, int K,
                             const __half* A, const __half* B, float* C,
                             int warmup_iters = 5, int bench_iters = 20) {
    float alpha = 1.0f, beta = 0.0f;
    CudaTimer timer;

    for (int i = 0; i < warmup_iters; i++) {
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, M, K, &alpha,
                                  B, CUDA_R_16F, N,
                                  A, CUDA_R_16F, K,
                                  &beta,
                                  C, CUDA_R_32F, N,
                                  CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    cudaDeviceSynchronize();

    timer.begin();
    for (int i = 0; i < bench_iters; i++) {
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, M, K, &alpha,
                                  B, CUDA_R_16F, N,
                                  A, CUDA_R_16F, K,
                                  &beta,
                                  C, CUDA_R_32F, N,
                                  CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    cudaDeviceSynchronize();
    return timer.end() / bench_iters;
}

inline double get_gflops(int M, int N, int K, float ms) {
    double flops = 2.0 * (double)M * (double)N * (double)K;
    return flops / (ms * 1e6);
}

inline void print_gflops_row(const char* name, float ms, int M, int N, int K,
                             float cublas_gflops = 0.0f) {
    double gflops = get_gflops(M, N, K, ms);
    if (cublas_gflops > 0.0f) {
        printf("| %-30s | %10.2f | %8.2f%% |\n", name, gflops,
               100.0 * gflops / cublas_gflops);
    } else {
        printf("| %-30s | %10.2f | %8s |\n", name, gflops, "-");
    }
}

inline void print_device_info() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Global memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("Peak GFLOPS (FP32, est): %.0f\n",
           2.0 * prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor
           * prop.clockRate / 1000.0 / 1000.0 * 2.0);
    printf("\n");
}
