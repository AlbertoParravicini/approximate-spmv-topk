#pragma once

// This code is adapted from LightSpMV, http://lightspmv.sourceforge.net/homepage.htm#latest
// Yongchao Liu and Bertil Schmidt: "LightSpMV: faster CSR-based sparse matrix-vector multiplication on CUDA-enabled GPUs". 26th IEEE International Conference on Application-specific Systems, Architectures and Processors (ASAP 2015), 2015, pp. 82-89

#include <cuda_fp16.h>

//////////////////////////////
//////////////////////////////

#define WARP_SIZE 32
#define THREADS_PER_VECTOR 4
#define MAX_NUM_VECTORS_PER_BLOCK (1024 / THREADS_PER_VECTOR)

//////////////////////////////
//////////////////////////////

template <typename I, typename V>
__global__ void light_spmv(I *cudaRowCounter, I *d_ptr, I *d_cols, V *d_val, V *d_vector, V *d_out, I N) {
    I i;
    V sum;
    I row;
    I rowStart, rowEnd;
    I laneId = threadIdx.x % THREADS_PER_VECTOR;       //lane index in the vector
    I vectorId = threadIdx.x / THREADS_PER_VECTOR;     //vector index in the thread block
    I warpLaneId = threadIdx.x & 31;                   //lane index in the warp
    I warpVectorId = warpLaneId / THREADS_PER_VECTOR;  //vector index in the warp

    __shared__ volatile I space[MAX_NUM_VECTORS_PER_BLOCK][2];

    // Get the row index
    if (warpLaneId == 0) {
        row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
    }
    // Broadcast the value to other threads in the same warp and compute the row index of each vector
    row = __shfl_sync(0xffffffff, row, 0) + warpVectorId;

    while (row < N) {
        // Use two threads to fetch the row offset
        if (laneId < 2) {
            space[vectorId][laneId] = d_ptr[row + laneId];
        }
        rowStart = space[vectorId][0];
        rowEnd = space[vectorId][1];

        sum = 0;
        // Compute dot product
        if (THREADS_PER_VECTOR == 32) {
            // Ensure aligned memory access
            i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

            // Process the unaligned part
            if (i >= rowStart && i < rowEnd) {
                sum += d_val[i] * d_vector[d_cols[i]];
            }

            // Process the aligned part
            for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) {
                sum += d_val[i] * d_vector[d_cols[i]];
            }
        } else {
            for (i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) {
                sum += d_val[i] * d_vector[d_cols[i]];
            }
        }
        // Intra-vector reduction
        for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, i);
        }

        // Save the results
        if (laneId == 0) {
            d_out[row] = sum;
        }

        // Get a new row index
        if (warpLaneId == 0) {
            row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
        }
        // Broadcast the row index to the other threads in the same warp and compute the row index of each vector
        row = __shfl_sync(0xffffffff, row, 0) + warpVectorId;
    }
}

template <typename I, typename V>
__global__ void light_spmv_test(I *cudaRowCounter, I *d_ptr, I *d_cols, V *d_val, V *d_vector, V *d_out, I N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        d_out[i] = d_val[i] * __float2half(2.0f);
    }
}

__global__ void float_to_half(float *in, half *out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = __float2half(in[i]);
    }
}

__global__ void half_to_float(half *in, float *out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        out[i] = __half2float(in[i]);
    }
}