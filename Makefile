NVCC := nvcc
ARCH := -arch=sm_75
FLAGS := -std=c++17 -O3 -Xcompiler -O3
CUBLAS := -lcublas -lcudart

KERNELS := 00_naive 02_shared_tiled 03_1d_blocktiling 04_2d_blocktiling \
           05_vectorized 06_bank_conflict_free 07_double_buffered 08_vectorized_all \
           09_wmma_tensorcore 10_wmma_multi 11_wmma_direct 12_wmma_opt
KERNEL_SRCS := $(addsuffix .cu, $(addprefix kernels/, $(KERNELS)))
KERNEL_EXES := $(addsuffix .exe, $(addprefix kernels/, $(KERNELS)))

BENCH_SRC := bench/benchmark_all.cu
BENCH_EXE := bench/benchmark_all.exe

.PHONY: all kernels benchmark clean

all: kernels benchmark

kernels: $(KERNEL_EXES)

benchmark: $(BENCH_EXE)

kernels/%.exe: kernels/%.cu common.cuh
	$(NVCC) $(ARCH) $(FLAGS) -o $@ $< $(CUBLAS)

$(BENCH_EXE): $(BENCH_SRC) $(KERNEL_SRCS) common.cuh
	$(NVCC) $(ARCH) $(FLAGS) -DBENCHMARK_ALL -o $@ $(BENCH_SRC) $(KERNEL_SRCS) $(CUBLAS)

run_%: kernels/%.exe
	./$<

run_benchmark: $(BENCH_EXE)
	./$(BENCH_EXE)

clean:
	rm -f $(KERNEL_EXES) $(BENCH_EXE)
