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

const double R = 0.08;
const double V = 0.3;
const double T = 1.0;
const double K = 60.0;
const int loop_num = 10;

__device__ inline double cndGPU(double d) {
  const double A1 = 0.31938153f;
  const double A2 = -0.356563782f;
  const double A3 = 1.781477937f;
  const double A4 = -1.821255978f;
  const double A5 = 1.330274429f;
  const double RSQRT2PI = 0.39894228040143267793994605993438f;

  double K = 1.0 / (1.0 + 0.2316419 * fabs(d));

  double cnd = RSQRT2PI * exp(-0.5f * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

  if (d > 0) cnd = 1.0 - cnd;

  return cnd;
}

__global__ void
// bs(const double *x, double *y, int N, double R, double V, double T, double K) {
bs(int n, double * y, double * x, int N, double R, double V, double T, double K) {
  double sqrtT = 1.0 / rsqrt(T);
  for (int l = 0; l < loop_num; l ++) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
        i += blockDim.x * gridDim.x) {
        double expRT;
        double d1, d2, CNDD1, CNDD2;
        double xi = x[i];
        d1 = (log(xi / K) + (R + 0.5 * V * V) * T) / (V * sqrtT);
        d2 = d1 - V * sqrtT;

        CNDD1 = cndGPU(d1);
        CNDD2 = cndGPU(d2);

        // Calculate Call and Put simultaneously
        expRT = exp(-R * T);
        y[i] = xi * CNDD1 - K * expRT * CNDD2;
    }
  }
}

int main() {
    // Options
    const size_t N = 80000000;
    const int block_size_1d = 64;
    const int num_blocks = 64;
    const int data_size = sizeof(double) * N;

    // Variable definition
    double *host_x_0, *host_x_1, *host_x_2, *host_x_3, *host_x_4;
    double *host_x_5, *host_x_6, *host_x_7, *host_x_8, *host_x_9;
    double *host_y_0, *host_y_1, *host_y_2, *host_y_3, *host_y_4;
    double *host_y_5, *host_y_6, *host_y_7, *host_y_8, *host_y_9;
    double *device_x_0, *device_x_1, *device_x_2, *device_x_3, *device_x_4;
    double *device_x_5, *device_x_6, *device_x_7, *device_x_8, *device_x_9;
    double *device_y_0, *device_y_1, *device_y_2, *device_y_3, *device_y_4;
    double *device_y_5, *device_y_6, *device_y_7, *device_y_8, *device_y_9;

    // Host allocation
    host_x_0 = (double *)malloc(sizeof(double) * N);
    host_x_1 = (double *)malloc(sizeof(double) * N);
    host_x_2 = (double *)malloc(sizeof(double) * N);
    host_x_3 = (double *)malloc(sizeof(double) * N);
    host_x_4 = (double *)malloc(sizeof(double) * N);
    host_x_5 = (double *)malloc(sizeof(double) * N);
    host_x_6 = (double *)malloc(sizeof(double) * N);
    host_x_7 = (double *)malloc(sizeof(double) * N);
    host_x_8 = (double *)malloc(sizeof(double) * N);
    host_x_9 = (double *)malloc(sizeof(double) * N);
    host_y_0 = (double *)malloc(sizeof(double) * N);
    host_y_1 = (double *)malloc(sizeof(double) * N);
    host_y_2 = (double *)malloc(sizeof(double) * N);
    host_y_3 = (double *)malloc(sizeof(double) * N);
    host_y_4 = (double *)malloc(sizeof(double) * N);
    host_y_5 = (double *)malloc(sizeof(double) * N);
    host_y_6 = (double *)malloc(sizeof(double) * N);
    host_y_7 = (double *)malloc(sizeof(double) * N);
    host_y_8 = (double *)malloc(sizeof(double) * N);
    host_y_9 = (double *)malloc(sizeof(double) * N);
    
    // Host initialization
    for (int j = 0; j < N; j++) {
        double tmp = 60 - 0.5 + (double)rand() / RAND_MAX;
        host_x_0[j] = tmp;
        host_x_1[j] = tmp;
        host_x_2[j] = tmp;
        host_x_3[j] = tmp;
        host_x_4[j] = tmp;
        host_x_5[j] = tmp;
        host_x_6[j] = tmp;
        host_x_7[j] = tmp;
        host_x_8[j] = tmp;
        host_x_9[j] = tmp;
    }

    auto start = std::chrono::high_resolution_clock::now();
    // Device Allocation
    cudaMalloc(&device_x_0, sizeof(double) * N);
    cudaMalloc(&device_x_1, sizeof(double) * N);
    cudaMalloc(&device_x_2, sizeof(double) * N);
    cudaMalloc(&device_x_3, sizeof(double) * N);
    cudaMalloc(&device_x_4, sizeof(double) * N);
    cudaMalloc(&device_x_5, sizeof(double) * N);
    cudaMalloc(&device_x_6, sizeof(double) * N);
    cudaMalloc(&device_x_7, sizeof(double) * N);
    cudaMalloc(&device_x_8, sizeof(double) * N);
    cudaMalloc(&device_x_9, sizeof(double) * N);
    cudaMalloc(&device_y_0, sizeof(double) * N);
    cudaMalloc(&device_y_1, sizeof(double) * N);
    cudaMalloc(&device_y_2, sizeof(double) * N);
    cudaMalloc(&device_y_3, sizeof(double) * N);
    cudaMalloc(&device_y_4, sizeof(double) * N);
    cudaMalloc(&device_y_5, sizeof(double) * N);
    cudaMalloc(&device_y_6, sizeof(double) * N);
    cudaMalloc(&device_y_7, sizeof(double) * N);
    cudaMalloc(&device_y_8, sizeof(double) * N);
    cudaMalloc(&device_y_9, sizeof(double) * N);

    // Memory copy from host to device
    cudaMemcpy(device_x_0, host_x_0, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_x_1, host_x_1, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_x_2, host_x_2, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_x_3, host_x_3, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_x_4, host_x_4, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_x_5, host_x_5, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_x_6, host_x_6, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_x_7, host_x_7, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_x_8, host_x_8, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_x_9, host_x_9, data_size, cudaMemcpyHostToDevice);

    // Execution
    bs<<<num_blocks, block_size_1d>>>(1, device_y_0, device_x_0, N, R, V, T, K);
    bs<<<num_blocks, block_size_1d>>>(1, device_y_1, device_x_1, N, R, V, T, K);
    bs<<<num_blocks, block_size_1d>>>(1, device_y_2, device_x_2, N, R, V, T, K);
    bs<<<num_blocks, block_size_1d>>>(1, device_y_3, device_x_3, N, R, V, T, K);
    bs<<<num_blocks, block_size_1d>>>(1, device_y_4, device_x_4, N, R, V, T, K);
    bs<<<num_blocks, block_size_1d>>>(1, device_y_5, device_x_5, N, R, V, T, K);
    bs<<<num_blocks, block_size_1d>>>(1, device_y_6, device_x_6, N, R, V, T, K);
    bs<<<num_blocks, block_size_1d>>>(1, device_y_7, device_x_7, N, R, V, T, K);
    bs<<<num_blocks, block_size_1d>>>(1, device_y_8, device_x_8, N, R, V, T, K);
    bs<<<num_blocks, block_size_1d>>>(1, device_y_9, device_x_9, N, R, V, T, K);

    // Memory copy from device to host
    cudaMemcpy(host_y_0, device_y_0, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_y_1, device_y_1, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_y_2, device_y_2, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_y_3, device_y_3, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_y_4, device_y_4, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_y_5, device_y_5, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_y_6, device_y_6, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_y_7, device_y_7, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_y_8, device_y_8, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_y_9, device_y_9, data_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_x_0);
    cudaFree(device_x_1);
    cudaFree(device_x_2);
    cudaFree(device_x_3);
    cudaFree(device_x_4);
    cudaFree(device_x_5);
    cudaFree(device_x_6);
    cudaFree(device_x_7);
    cudaFree(device_x_8);
    cudaFree(device_x_9);
    cudaFree(device_y_0);
    cudaFree(device_y_1);
    cudaFree(device_y_2);
    cudaFree(device_y_3);
    cudaFree(device_y_4);
    cudaFree(device_y_5);
    cudaFree(device_y_6);
    cudaFree(device_y_7);
    cudaFree(device_y_8);
    cudaFree(device_y_9);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("b5 device execution time: %ld ms\n", duration.count());

    // Free hsot memory
    free(host_x_0);
    free(host_x_1);
    free(host_x_2);
    free(host_x_3);
    free(host_x_4);
    free(host_x_5);
    free(host_x_6);
    free(host_x_7);
    free(host_x_8);
    free(host_x_9);
    free(host_y_0);
    free(host_y_1);
    free(host_y_2);
    free(host_y_3);
    free(host_y_4);
    free(host_y_5);
    free(host_y_6);
    free(host_y_7);
    free(host_y_8);
    free(host_y_9);

    // Print
    printf("b5 finished running\n");

    return 0;
}
