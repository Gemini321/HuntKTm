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

#include <cstdio>
#include <chrono>
#include <thread>
using namespace std;

const int loop_num = 10;

__global__ void square(int output_n, float * y, float * x, int n) {
    for (int l = 0; l < loop_num; l ++) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
            i += blockDim.x * gridDim.x) {
            y[i] = x[i] * x[i];
        }
    }
}

__inline__ __device__ float warp_reduce(float val) {
    int warp_size = 32;
    for (int offset = warp_size / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void reduce(int n, float * z, float * x, float * y, int N) {
    int warp_size = 32;
    for (int l = 0; l < loop_num; l ++) {
        float sum = float(0);
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
            i += blockDim.x * gridDim.x) {
            sum += x[i] - y[i];
        }
        sum = warp_reduce(sum);  // Obtain the sum of values in the current warp;
        if ((threadIdx.x & (warp_size - 1)) == 0)  // Same as (threadIdx.x % warp_size) == 0 but faster
            atomicAdd(z, sum);  // The first thread in the warp updates the output;
    }
}

int main() {
    // Options
    size_t N = 300000000;
    int block_size_1d = 64;
    int num_blocks = 32;

    // Variable definition
    float *host_x, *host_y, *host_res;
    float *device_x, *device_y, *device_x1, *device_y1, *device_res;

    // Host allocation
    host_x = (float *)malloc(sizeof(float) * N);
    host_y = (float *)malloc(sizeof(float) * N);
    host_res = (float *)malloc(sizeof(float));

    // Host initialization
    for (int i = 0; i < N; i++) {
        // float tmp = (float)(rand()) / (float)(RAND_MAX);
        host_x[i] = (float)(rand()) / (float)(RAND_MAX);
        host_y[i] = (float)(rand()) / (float)(RAND_MAX);
    }
    host_res[0] = 0.0;

    auto start = std::chrono::high_resolution_clock::now();
    // Device allocation
    cudaMalloc(&device_x, sizeof(float) * N);
    cudaMalloc(&device_y, sizeof(float) * N);
    cudaMalloc(&device_x1, sizeof(float) * N);
    cudaMalloc(&device_y1, sizeof(float) * N);
    cudaMalloc(&device_res, sizeof(float));

    // Memory copy from host to device
    cudaMemcpy(device_x, host_x, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, host_x, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Execution
    square<<<num_blocks, block_size_1d>>>(1, device_x1, device_x, N);
    square<<<num_blocks, block_size_1d>>>(1, device_y1, device_y, N);
    reduce<<<num_blocks, block_size_1d>>>(1, device_res, device_x1, device_y1, N);

    // Memory copy from device to host
    cudaMemcpy(host_res, device_res, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_x1);
    cudaFree(device_y1);
    cudaFree(device_res);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("b1 device execution time: %ld ms\n", duration.count());

    // Free host memory
    free(host_x);
    free(host_y);
    free(host_res);

    // Print
    printf("b1 finished running\n");

    return 0;
}
