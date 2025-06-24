// Copyright (c) 2020, 2021, NECSTLab, Politecnico di Milano. All rights
// reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NECSTLab nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//  * Neither the name of Politecnico di Milano nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <set>
#include <string>
#include <thread>
#include <cstdio>
#include "../common/utils.h"

//////////////////////////////
//////////////////////////////

#define WARP_SIZE 32
#define THREADS_PER_VECTOR 4
#define MAX_NUM_VECTORS_PER_BLOCK (1024 / THREADS_PER_VECTOR)

/////////////////////////////
/////////////////////////////

// __global__ void spmv(const int *ptr, const int *idx, const int *val, const
// float *vec, float *res, int num_rows, int num_nnz) {
__global__ void spmv(int n, float *res, int *ptr, int *idx, int *val,
                     float *vec, int num_rows, int num_nnz) {  // n=1
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < num_rows;
        n += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int i = ptr[n]; i < ptr[n + 1]; i++) {
            sum += val[i] * vec[idx[i]];
        }
            res[n] = sum;
    }
}

// __global__ void spmv2(const int *ptr, const int *idx, const int *val, const
// float *vec, float *res, int num_rows, int num_nnz) {
__global__ void spmv2(int n, float *res, int *ptr, int *idx, int *val,
                      float *vec, int num_rows, int num_nnz) {  // n=1
    // Thread ID in block
    int t = threadIdx.x;

    // Thread ID in warp
    int lane = t & (WARP_SIZE - 1);

    // Number of warps per block
    int warpsPerBlock = blockDim.x / WARP_SIZE;

    // One row per warp
    int row = (blockIdx.x * warpsPerBlock) + (t / WARP_SIZE);

    extern __shared__ volatile float vals[];

    if (row < num_rows) {
        int rowStart = ptr[row];
        int rowEnd = ptr[row + 1];
        float sum = 0;

        // Use all threads in a warp accumulate multiplied elements
        for (int j = rowStart + lane; j < rowEnd; j += WARP_SIZE) {
            int col = idx[j];
            sum += val[j] * vec[col];
        }
        vals[t] = sum;
        __syncthreads();

        // Reduce partial sums
        if (lane < 16) vals[t] += vals[t + 16];
        if (lane < 8) vals[t] += vals[t + 8];
        if (lane < 4) vals[t] += vals[t + 4];
        if (lane < 2) vals[t] += vals[t + 2];
        if (lane < 1) vals[t] += vals[t + 1];
        __syncthreads();

        // Write result
        if (lane == 0) {
            res[row] = vals[t];
        }
    }
}

// __global__ void spmv3(int *cudaRowCounter, int *d_ptr, int *d_cols, int
// *d_val, float *d_vector, float *d_out, int N) {
__global__ void spmv3(int n, int *cudaRowCounter, float * d_out, int *d_ptr,
                      int *d_cols, int *d_val, float *d_vector, int N) {  // n=2
    int i;
    float sum;
    int row;
    int rowStart, rowEnd;
    int laneId = threadIdx.x % THREADS_PER_VECTOR;  // lane index in the vector
    int vectorId =
        threadIdx.x / THREADS_PER_VECTOR;  // vector index in the thread block
    int warpLaneId = threadIdx.x & 31;     // lane index in the warp
    int warpVectorId =
        warpLaneId / THREADS_PER_VECTOR;  // vector index in the warp

    __shared__ volatile int space[MAX_NUM_VECTORS_PER_BLOCK][2];

    // Get the row index
    if (warpLaneId == 0) {
        row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
    }
    // Broadcast the value to other threads in the same warp and compute the row
    // index of each vector
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
        // Broadcast the row index to the other threads in the same warp and compute
        // the row index of each vector
        row = __shfl_sync(0xffffffff, row, 0) + warpVectorId;
    }
}

__inline__ __device__ float warp_reduce(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// extern "C" __global__ void sum(const float *x, float *z, int N) {
__global__ void sum(int n, float *z, float * x, int N) {  // n=1
    int warp_size = 32;
    float sum = float(0);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
        i += blockDim.x * gridDim.x) {
        sum += x[i];
    }
    sum = warp_reduce(sum);  // Obtain the sum of values in the current warp;
    if ((threadIdx.x & (warp_size - 1)) ==
        0)                // Same as (threadIdx.x % warp_size) == 0 but faster
        atomicAdd(z, sum);  // The first thread in the warp updates the output;
}

// __global__ void divide(const float *x, float *y, float *val, int n) {
__global__ void divide(int output_n, float *y, float * x, float *val,
                       int n) {  // n=1
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
        i += blockDim.x * gridDim.x) {
        y[i] = x[i] / val[0];
    }
}

// __global__ void reset_kernel(float *n1, float *n2, int *r1, int *r2) {
__global__ void reset_kernel(int n, float *n1, float *n2, int *r1,
                             int *r2) {  // n=4
    if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
        *n1 = 0;
        *n2 = 0;
        *r1 = 0;
        *r2 = 0;
    }
}

inline void random_coo(int *x, int *y, int *val, int N, int degree) {
    for (int i = 0; i < N; i++) {
        std::set<int> edges;
        while (edges.size() < degree) {
            edges.insert(rand() % N);
        }
        int j = 0;
        for (auto iter = edges.begin(); iter != edges.end(); iter++, j++) {
            x[i * degree + j] = i;
            y[i * degree + j] = *iter;
            val[i * degree + j] = 1;
        }
    }
}

void host_initialization(int *host_ptr, int *host_ptr2, int *host_idx, int *host_idx2, 
    int *host_val, int *host_val2, int *host_rowCounter1, int *host_rowCounter2, 
    float *host_auth1, float *host_auth2, float *host_hub1, float *host_hub2, 
    float *host_auth_norm, float *host_hub_norm, int N, int degree, int nnz) {
    // #define B7First
//     Regenerate data everytime we change N
#ifdef B7First
    int *ptr_tmp, *idx_tmp, *val_tmp, *ptr2_tmp, *idx2_tmp, *val2_tmp;
    int *x, *y, *v;
    ptr_tmp = (int *)malloc(sizeof(int) * (N + 1));
    ptr2_tmp = (int *)malloc(sizeof(int) * (N + 1));
    idx_tmp = (int *)malloc(sizeof(int) * nnz);
    idx2_tmp = (int *)malloc(sizeof(int) * nnz);
    val_tmp = (int *)malloc(sizeof(int) * nnz);
    val2_tmp = (int *)malloc(sizeof(int) * nnz);
    x = (int *)malloc(nnz * sizeof(int));
    y = (int *)malloc(nnz * sizeof(int));
    v = (int *)malloc(nnz * sizeof(int));

    random_coo(x, y, v, N, degree);
    coo2csr(ptr_tmp, idx_tmp, val_tmp, x, y, v, N, N, nnz);
    coo2csr(ptr2_tmp, idx2_tmp, val2_tmp, y, x, v, N, N, nnz);

    std::cout << "Data creating, writing to file...\n";

    auto tt1 = std::thread(write_to_file<int>, ptr_tmp,(N + 1),"/root/pwx/MultiGPU-Scheduler/benchmarks/multi_stream/b7/data/ptr_tmp_"+std::to_string(N));
    auto tt2 = std::thread(write_to_file<int>, ptr2_tmp,(N + 1),"/root/pwx/MultiGPU-Scheduler/benchmarks/multi_stream/b7/data/ptr2_tmp_"+std::to_string(N));
    auto tt3 = std::thread(write_to_file<int>, idx_tmp,nnz,"/root/pwx/MultiGPU-Scheduler/benchmarks/multi_stream/b7/data/idx_tmp_"+std::to_string(N));
    auto tt4 = std::thread(write_to_file<int>, idx2_tmp,nnz,"/root/pwx/MultiGPU-Scheduler/benchmarks/multi_stream/b7/data/idx2_tmp_"+std::to_string(N));
    auto tt5 = std::thread(write_to_file<int>, val_tmp,nnz,"/root/pwx/MultiGPU-Scheduler/benchmarks/multi_stream/b7/data/val_tmp_"+std::to_string(N));
    auto tt6 = std::thread(write_to_file<int>, val2_tmp,nnz,"/root/pwx/MultiGPU-Scheduler/benchmarks/multi_stream/b7/data/val2_tmp_"+std::to_string(N));
    auto tt7 = std::thread(write_to_file<int>, x,nnz,"/root/pwx/MultiGPU-Scheduler/benchmarks/multi_stream/b7/data/x_tmp_"+std::to_string(N));
    auto tt8 = std::thread(write_to_file<int>, y,nnz,"/root/pwx/MultiGPU-Scheduler/benchmarks/multi_stream/b7/data/y_tmp_"+std::to_string(N));
    tt1.join();
    tt2.join();
    tt3.join();
    tt4.join();
    tt5.join();
    tt6.join();
    tt7.join();
    tt8.join();

    free(ptr_tmp);
    free(ptr2_tmp);
    free(idx_tmp);
    free(idx2_tmp);
    free(val_tmp);
    free(val2_tmp);
    free(x);
    free(y);
    free(v);
    exit(1);
#endif
    auto t1 =std::thread(read_file<int>, host_ptr, (N + 1),  "/root/pwx/MultiGPU-Scheduler/benchmarks/multi_stream/b7/data/ptr_tmp_"+std::to_string(N));
    auto t2 =std::thread(read_file<int>, host_ptr2, (N + 1), "/root/pwx/MultiGPU-Scheduler/benchmarks/multi_stream/b7/data/ptr2_tmp_"+std::to_string(N));
    auto t3 =std::thread(read_file<int>, host_idx, nnz, "/root/pwx/MultiGPU-Scheduler/benchmarks/multi_stream/b7/data/idx_tmp_"+std::to_string(N));
    auto t4 =std::thread(read_file<int>, host_idx2, nnz, "/root/pwx/MultiGPU-Scheduler/benchmarks/multi_stream/b7/data/idx2_tmp_"+std::to_string(N));
    auto t5 =std::thread(read_file<int>, host_val, nnz, "/root/pwx/MultiGPU-Scheduler/benchmarks/multi_stream/b7/data/val_tmp_"+std::to_string(N));
    auto t6 =std::thread(read_file<int>, host_val2, nnz, "/root/pwx/MultiGPU-Scheduler/benchmarks/multi_stream/b7/data/val2_tmp_"+std::to_string(N));
    // auto t7 =std::thread(read_file<int>, x, nnz, "/root/pwx/MultiGPU-Scheduler/benchmarks/multi_stream/b7/data/x_tmp_"+std::to_string(N));
    // auto t8 =std::thread(read_file<int>, y, nnz, "/root/pwx/MultiGPU-Scheduler/benchmarks/multi_stream/b7/data/y_tmp_"+std::to_string(N));
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
    t6.join();
    // t7.join();
    // t8.join();

    for (int i = 0; i < N; i++) {
        host_auth1[i] = 1;
        host_auth2[i] = 1;
        host_hub1[i] = 1;
        host_hub2[i] = 1;
    }
    host_auth_norm[0] = 0;
    host_hub_norm[0] = 0;
    host_rowCounter1[0] = 0;
    host_rowCounter2[0] = 0;
}

int main() {
    // Options
    const size_t N = 1000000;
    const int block_size_1d = 64;
    const int num_blocks = 16;
    const int degree = 100;
    const int nnz = degree * N;
    const int nb = ceil(N / ((float)block_size_1d));
    const int memsize = block_size_1d * sizeof(float);

    // Variable definition
    int *host_ptr, *host_idx, *host_val, *host_ptr2, *host_idx2, *host_val2, 
        *host_rowCounter1, *host_rowCounter2;
    // int *host_x, *host_y, *host_v;
    float *host_auth1, *host_auth2, *host_hub1, *host_hub2, *host_auth_norm, *host_hub_norm;
    int *device_ptr, *device_idx, *device_val, *device_ptr2, *device_idx2, *device_val2, 
        *device_rowCounter1, *device_rowCounter2;
    // int *device_x, *device_y, *device_v;
    float *device_auth1, *device_auth2, *device_hub1, *device_hub2, *device_auth_norm, *device_hub_norm;

    // Allocation
    host_ptr = (int *)malloc(sizeof(int) * (N + 1));
    host_ptr2 = (int *)malloc(sizeof(int) * (N + 1));
    host_idx = (int *)malloc(sizeof(int) * nnz);
    host_idx2 = (int *)malloc(sizeof(int) * nnz);
    host_val = (int *)malloc(sizeof(int) * nnz);
    host_val2 = (int *)malloc(sizeof(int) * nnz);
    host_rowCounter1 = (int *)malloc(sizeof(int));
    host_rowCounter2 = (int *)malloc(sizeof(int));
    host_auth1 = (float *)malloc(sizeof(float) * N);
    host_auth2 = (float *)malloc(sizeof(float) * N);
    host_hub1 = (float *)malloc(sizeof(float) * N);
    host_hub2 = (float *)malloc(sizeof(float) * N);
    host_auth_norm = (float *)malloc(sizeof(float));
    host_hub_norm = (float *)malloc(sizeof(float));

    cudaMalloc(&device_ptr, sizeof(int) * (N + 1));
    cudaMalloc(&device_ptr2, sizeof(int) * (N + 1));
    cudaMalloc(&device_idx, sizeof(int) * nnz);
    cudaMalloc(&device_idx2, sizeof(int) * nnz);
    cudaMalloc(&device_val, sizeof(int) * nnz);
    cudaMalloc(&device_val2, sizeof(int) * nnz);
    cudaMalloc(&device_rowCounter1, sizeof(int));
    cudaMalloc(&device_rowCounter2, sizeof(int));
    cudaMalloc(&device_auth1, sizeof(float) * N);
    cudaMalloc(&device_auth2, sizeof(float) * N);
    cudaMalloc(&device_hub1, sizeof(float) * N);
    cudaMalloc(&device_hub2, sizeof(float) * N);
    cudaMalloc(&device_auth_norm, sizeof(float));
    cudaMalloc(&device_hub_norm, sizeof(float));

    // Host initialization
    host_initialization(host_ptr, host_ptr2, host_idx, host_idx2, host_val, host_val2, 
        host_rowCounter1, host_rowCounter2, host_auth1, host_auth2, host_hub1, host_hub2, 
        host_auth_norm, host_hub_norm, N, degree, nnz);

    // Memory copy from host to device
    cudaMemcpy(device_ptr, host_ptr, sizeof(int) * (N + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(device_ptr2, host_ptr2, sizeof(int) * (N + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(device_idx, host_idx, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(device_idx2, host_idx2, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(device_val, host_val, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(device_val2, host_val2, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(device_auth1, host_auth1, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_auth2, host_auth2, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_hub1, host_hub1, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_hub2, host_hub2, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_auth_norm, host_auth_norm, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_hub_norm, host_hub_norm, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_rowCounter1, host_rowCounter1, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_rowCounter2, host_rowCounter2, sizeof(int), cudaMemcpyHostToDevice);

    // Execution
    spmv3<<<nb, block_size_1d, memsize>>>(2, device_rowCounter1, device_auth2, device_ptr2, device_idx2, device_val2,
                                        device_hub1, N);
    spmv3<<<nb, block_size_1d, memsize>>>(2, device_rowCounter2, device_hub2, device_ptr, device_idx, device_val,
                                        device_auth1, N);
    sum<<<num_blocks, block_size_1d>>>(1, device_auth_norm, device_auth2, N);
    sum<<<num_blocks, block_size_1d>>>(1, device_hub_norm, device_hub2, N);
    divide<<<num_blocks, block_size_1d>>>(1, device_auth1, device_auth2, device_auth_norm, N);
    divide<<<num_blocks, block_size_1d>>>(1, device_hub1, device_hub2, device_hub_norm, N);

    // Memory copy from device to host
    cudaMemcpy(host_auth1, device_auth1, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_auth_norm, device_auth_norm, sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    free(host_ptr);
    free(host_ptr2);
    free(host_idx);
    free(host_idx2);
    free(host_val);
    free(host_val2);
    free(host_rowCounter1);
    free(host_rowCounter2);
    free(host_auth1);
    free(host_auth2);
    free(host_hub1);
    free(host_hub2);
    free(host_auth_norm);
    free(host_hub_norm);
    cudaFree(device_ptr);
    cudaFree(device_ptr2);
    cudaFree(device_idx);
    cudaFree(device_idx2);
    cudaFree(device_val);
    cudaFree(device_val2);
    cudaFree(device_rowCounter1);
    cudaFree(device_rowCounter2);
    cudaFree(device_auth1);
    cudaFree(device_auth2);
    cudaFree(device_hub1);
    cudaFree(device_hub2);
    cudaFree(device_auth_norm);
    cudaFree(device_hub_norm);

    // Print
    printf("b7 finished running\n");

    return 0;
}
