cmake_minimum_required(VERSION 3.18)
project(cuda-gemm LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75 -O3")

# Individual kernel targets
set(KERNEL_NAMES 00_naive 02_shared_tiled 03_1d_blocktiling 04_2d_blocktiling
                 05_vectorized 06_bank_conflict_free 07_double_buffered 08_vectorized_all
                 09_wmma_tensorcore 10_wmma_multi 11_wmma_direct 12_wmma_opt 13_wmma_combined)

foreach(KERNEL ${KERNEL_NAMES})
    add_executable(${KERNEL} kernels/${KERNEL}.cu)
    target_link_libraries(${KERNEL} PRIVATE cublas cudart)
endforeach()

# Combined benchmark
add_executable(benchmark_all
    bench/benchmark_all.cu
    kernels/00_naive.cu
    kernels/02_shared_tiled.cu
    kernels/03_1d_blocktiling.cu
    kernels/04_2d_blocktiling.cu
    kernels/05_vectorized.cu
    kernels/06_bank_conflict_free.cu
    kernels/07_double_buffered.cu
    kernels/08_vectorized_all.cu
    kernels/09_wmma_tensorcore.cu
    kernels/10_wmma_multi.cu
    kernels/11_wmma_direct.cu
    kernels/12_wmma_opt.cu
    kernels/13_wmma_combined.cu
)
target_compile_definitions(benchmark_all PRIVATE BENCHMARK_ALL)
target_link_libraries(benchmark_all PRIVATE cublas cudart)
