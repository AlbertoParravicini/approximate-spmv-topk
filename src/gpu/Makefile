# This command might be required on Centos to load recent versions of GCC
# scl enable devtoolset-7 bash

CC=/usr/local/cuda-11.1/bin/nvcc
FLAGS = -std=c++14 -O3 -arch=sm_70 -lcusparse
BIN_FOLDER=bin
FILES=host_spmv_csr_gpu.cu host_spmv_coo_gpu.cu host_spmv_topk_csr_gpu.cu host_spmv_topk_coo_gpu.c

.PHONY: clean all csr coo csr_topk coo_topk

all: \
    clean \
    csr \
	coo \
	csr_topk \
	coo_topk

clean:
	rm -rf $(BIN_FOLDER);
	mkdir $(BIN_FOLDER);

csr host_spmv_csr_gpu.cu:
	$(CC) host_spmv_csr_gpu.cu $(FLAGS) -o $(BIN_FOLDER)/approximate-spmv-gpu-csr;

coo host_spmv_coo_gpu.cu:
	$(CC) host_spmv_coo_gpu.cu $(FLAGS) -o $(BIN_FOLDER)/approximate-spmv-gpu-coo;

csr_topk host_spmv_topk_csr_gpu.cu:
	$(CC) host_spmv_topk_csr_gpu.cu $(FLAGS) -o $(BIN_FOLDER)/approximate-spmv-gpu-csr-topk;

coo_topk host_spmv_topk_coo_gpu.cu:
	$(CC) host_spmv_topk_coo_gpu.cu $(FLAGS) -o $(BIN_FOLDER)/approximate-spmv-gpu-coo-topk;