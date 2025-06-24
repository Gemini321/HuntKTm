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

#define NUM_THREADS_PER_BLOCK_2D 8
#define NUM_THREADS_PER_BLOCK 32
#define WARP_SIZE 32
#define NUM_BLOCKS 16

//  __global__ void conv2d(float *out, float *x, float *kernels, int N, int M,
//  int L, int K, int k_out, int stride) {
__global__ void conv2d(int n, float * out, float * x, float * kernels, int N,
                       int M, int L, int K, int k_out, int stride) {  // n=1
  extern __shared__ float kernel_local[];
  int radius = K / 2;

  for (int m = 0; m < k_out; m++) {
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
      for (int j = threadIdx.y; j < K; j += blockDim.y) {
        for (int l = 0; l < L; l++) {
          kernel_local[l + L * (j + K * (i + K * m))] =
              kernels[l + L * (j + K * (i + K * m))];
        }
      }
    }
  }
  __syncthreads();

  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < (int)ceilf((float)N / stride) - radius;
       i += blockDim.x * gridDim.x) {
    int out_index = M * i / stride;
    for (int j = blockIdx.y * blockDim.y + threadIdx.y;
         j < (int)ceilf((float)M / stride) - radius;
         j += blockDim.y * gridDim.y) {
      for (int m = 0; m < k_out; m++) {
        // for (int m = blockIdx.z * blockDim.z + threadIdx.z; m < k_out; m +=
        // blockDim.z * gridDim.z) {
        float res = 0;
        int i_f = i * stride + radius;
        int j_f = j * stride + radius;
        // for (int k_i = -radius; k_i <= radius; k_i++) {
        //   for (int k_j = -radius; k_j <= radius; k_j++) {
        //     int kernel_index = (k_j + radius + K * (k_i + radius + K * m));
        //     for (int l = 0; l < L; l++) {
        //       int ni = i_f + k_i;
        //       int nj = j_f + k_j;
        //       res += kernel_local[l + L * kernel_index] *
        //              x[((ni * M) + nj) * L + l];
        //     }
        //   }
        // }
        for (int k_i = -radius; k_i <= radius; k_i++) {
            int ni = i_f + k_i;
            for (int k_j = -radius; k_j <= radius; k_j++) {
                int nj = j_f + k_j;
                if (ni >= 0 && ni < N && nj >= 0 && nj < M) {
                    int kernel_index = (k_j + radius) + K * ((k_i + radius) + K * m);
                    for (int l = 0; l < L; l++) {
                        res += kernel_local[l + L * kernel_index] * x[((ni * M) + nj) * L + l];
                    }
                }
            }
        }
        // Apply ReLU operator;
        out[m + k_out * (j + out_index)] = max(res, 0.0);
      }
    }
  }
}

//  __global__ void mean_pooling(float *out, float *x, int N, int M, int L, int
//  K, int stride) {
__global__ void mean_pooling(int n, float * out, float * x, int N, int M,
                             int L, int K, int stride) {  // n=1
  int radius = K / 2;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < (int)ceilf((float)N / stride) - radius;
       i += blockDim.x * gridDim.x) {
    int out_index = M * i / stride;
    int i_f = i * stride + radius;
    for (int j = blockIdx.y * blockDim.y + threadIdx.y;
         j < (int)ceilf((float)M / stride) - radius;
         j += blockDim.y * gridDim.y) {
      int j_f = j * stride + radius;
      for (int l = blockIdx.z * blockDim.z + threadIdx.z; l < L;
           l += blockDim.z * gridDim.z) {
        float res = 0;
        for (int k_i = -radius; k_i <= radius; k_i++) {
          int ni = i_f + k_i;
          for (int k_j = -radius; k_j <= radius; k_j++) {
            int nj = j_f + k_j;
            res += x[((ni * M) + nj) * L + l];
          }
        }
        // Apply mean operator;
        out[l + L * (j + out_index)] = res / (K * K);
      }
    }
  }
}

//  __global__ void gap(float *out, float *x, int N, int M, int L) {
__global__ void gap(int n, float * out, float * x, int N, int M, int L) {  // n=1
  extern __shared__ float out_local[];
  for (int i = threadIdx.x; i < L; i += blockDim.x) {
    out_local[i] = 0;
  }
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < M;
         j += blockDim.y * gridDim.y) {
      for (int l = 0; l < L; l++) {
        atomicAdd(out_local + l, x[l + L * (j + M * i)]);
      }
    }
  }
  __syncthreads();
  for (int l = threadIdx.x; l < L; l += blockDim.x) {
    atomicAdd(out + l, out_local[l] / (M * N));
  }
}

__inline__ __device__ float warp_reduce(float val) {
  int warp_size = 32;
  for (int offset = warp_size / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  return val;
}

//  __global__ void dot_product(const float *x, const float *y, float *z, int N)
//  {
__global__ void dot_product(int n, float *z, float *x, float *y,
                            int N) {  // n=1
  int warp_size = 32;
  float sum = float(0);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    sum += x[i] * y[i];
  }
  sum = warp_reduce(sum);  // Obtain the sum of values in the current warp;
  if ((threadIdx.x & (warp_size - 1)) ==
      0)                // Same as (threadIdx.x % warp_size) == 0 but faster
    atomicAdd(z, sum);  // The first thread in the warp updates the output;
}

//  __global__ void concat(float *z, const float *x, const float *y, int n) {
__global__ void concat(int output_n, float * z, float * x, float * y,
                       int n) {  // n=1
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    z[i] = x[i];
    z[i + n] = y[i];
  }
}

int main() {
    // Options
    const size_t N = 14000;
    const int block_size_1d = 16;
    const int block_size_2d = 4;
    const int num_blocks = 16;
    const int num_blocks_div2 = num_blocks / 2;
    const int K = 5;
    const int channels = 1;
    const int stride = 1;
    const int kn1 = 1;
    const int kn2 = 3;
    const int pooling_diameter = 2;
    const int x_len = N * N * channels;
    const int x1_len = (N / stride) * (N / stride) * kn1;
    const int pooled_len = x1_len / (pooling_diameter * pooling_diameter);
    const int x2_len = ((N / stride) / pooling_diameter / stride) * ((N / stride) / pooling_diameter / stride) * kn2;
    const int k1_len = channels * K * K * kn1;
    const int k2_len = kn1 * K * K * kn2;
    const int z_len = 2 * x2_len;
    dim3 grid_size_2_1(num_blocks_div2, num_blocks_div2);
    dim3 block_size_2d_dim_1(block_size_2d, block_size_2d);
    dim3 grid_size_2_2(num_blocks_div2, num_blocks_div2);
    dim3 block_size_2d_dim_2(block_size_2d, block_size_2d);
    dim3 grid_size_3_1(num_blocks_div2, num_blocks_div2, num_blocks_div2);
    dim3 block_size_3d_dim_1(block_size_2d / 4, block_size_2d / 4, block_size_2d / 4);
    dim3 grid_size_3_2(num_blocks_div2, num_blocks_div2, num_blocks_div2);
    dim3 block_size_3d_dim_2(block_size_2d / 4, block_size_2d / 4, block_size_2d / 4);
    dim3 grid_size_2_3(num_blocks_div2, num_blocks_div2);
    dim3 block_size_2d_dim_3(block_size_2d, block_size_2d);
    dim3 grid_size_2_4(num_blocks_div2, num_blocks_div2);
    dim3 block_size_2d_dim_4(block_size_2d, block_size_2d);

    // Variable definition
    float *host_x, *host_y, *host_kernel_1, *host_kernel_2, *host_kernel_3, *host_kernel_4, *host_dense_weights, *host_res;
    float *device_x, *device_x1, *device_x2, *device_y, *device_y1, *device_y2, *device_kernel_1, *device_kernel_2, *device_kernel_3, *device_kernel_4, *device_z, *device_dense_weights, *device_res;
    float *device_x11, *device_y11;

    // Host allocation
    host_x = (float *)malloc(sizeof(float) * x_len);
    host_y = (float *)malloc(sizeof(float) * x_len);
    host_kernel_1 = (float *)malloc(sizeof(float) * k1_len);
    host_kernel_2 = (float *)malloc(sizeof(float) * k2_len);
    host_kernel_3 = (float *)malloc(sizeof(float) * k1_len);
    host_kernel_4 = (float *)malloc(sizeof(float) * k2_len);
    host_dense_weights = (float *)malloc(sizeof(float) * z_len);
    host_res = (float *)malloc(sizeof(float));

    // Host initialization
    float tmp1 = ((float)(rand()) / (float)(RAND_MAX)) * 2 - 1;
    float tmp2 = ((float)(rand()) / (float)(RAND_MAX)) * 2 - 1;
    for (int i = 0; i < x_len; i++) {
        host_x[i] = tmp1;
        host_y[i] = tmp2;
    }
    for (int i = 0; i < k1_len; i++) {
        float tmp = ((float)(rand()) / (float)(RAND_MAX)) * 2 - 1;
        host_kernel_1[i] = tmp;
        host_kernel_3[i] = tmp;
    }
    for (int i = 0; i < k2_len; i++) {
        float tmp = ((float)(rand()) / (float)(RAND_MAX)) * 2 - 1;
        host_kernel_2[i] = tmp;
        host_kernel_4[i] = tmp;
    }
    for (int i = 0; i < z_len; i++) {
        host_dense_weights[i] = (((float)(rand()) / (float)(RAND_MAX)) * 2 - 1) / z_len;
    }

    auto start = std::chrono::high_resolution_clock::now();
    // Device allocation
    cudaMalloc(&device_x, sizeof(float) * x_len);
    cudaMalloc(&device_x1, sizeof(float) * x1_len);
    cudaMalloc(&device_x2, sizeof(float) * x2_len);
    cudaMalloc(&device_y, sizeof(float) * x_len);
    cudaMalloc(&device_y1, sizeof(float) * x1_len);
    cudaMalloc(&device_y2, sizeof(float) * x2_len);
    cudaMalloc(&device_kernel_1, sizeof(float) * k1_len);
    cudaMalloc(&device_kernel_2, sizeof(float) * k2_len);
    cudaMalloc(&device_kernel_3, sizeof(float) * k1_len);
    cudaMalloc(&device_kernel_4, sizeof(float) * k2_len);
    cudaMalloc(&device_z, sizeof(float) * z_len);
    cudaMalloc(&device_dense_weights, sizeof(float) * z_len);
    cudaMalloc(&device_res, sizeof(float));
    cudaMalloc(&device_x11, sizeof(float) * pooled_len);
    cudaMalloc(&device_y11, sizeof(float) * pooled_len);

    // Memory copy from host to device
    cudaMemcpy(device_kernel_1, host_kernel_1, sizeof(float) * k1_len, cudaMemcpyHostToDevice);
    cudaMemcpy(device_kernel_2, host_kernel_2, sizeof(float) * k2_len, cudaMemcpyHostToDevice);
    cudaMemcpy(device_kernel_3, host_kernel_3, sizeof(float) * k1_len, cudaMemcpyHostToDevice);
    cudaMemcpy(device_kernel_4, host_kernel_4, sizeof(float) * k2_len, cudaMemcpyHostToDevice);
    cudaMemcpy(device_dense_weights, host_dense_weights, sizeof(float) * z_len, cudaMemcpyHostToDevice);
    cudaMemcpy(device_x, host_x, sizeof(float) * x_len, cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, host_y, sizeof(float) * x_len, cudaMemcpyHostToDevice);

    // Execution
    conv2d<<<grid_size_2_1, block_size_2d_dim_1, K * K * kn1 * channels * sizeof(float)>>>(
            1, device_x1, device_x, device_kernel_1, N, N, channels, K, kn1, stride);
    conv2d<<<grid_size_2_2, block_size_2d_dim_2, K * K * kn1 * channels * sizeof(float)>>>(
            1, device_y1, device_y, device_kernel_3, N, N, channels, K, kn1, stride);
    mean_pooling<<<grid_size_3_1, block_size_3d_dim_1>>>(
        1, device_x11, device_x1, N / stride, N / stride, kn1, pooling_diameter,
        pooling_diameter);
    mean_pooling<<<grid_size_3_2, block_size_3d_dim_2>>>(
        1, device_y11, device_y1, N / stride, N / stride, kn1, pooling_diameter,
        pooling_diameter);
    conv2d<<<grid_size_2_3, block_size_2d_dim_3, K * K * kn1 * kn2 * sizeof(float)>>>(
        1, device_x2, device_x11, device_kernel_2, N / stride / pooling_diameter,
        N / stride / pooling_diameter, kn1, K, kn2, stride);
    conv2d<<<grid_size_2_4, block_size_2d_dim_4, K * K * kn1 * kn2 * sizeof(float)>>>(
        1, device_y2, device_y11, device_kernel_4, N / stride / pooling_diameter,
        N / stride / pooling_diameter, kn1, K, kn2, stride);
    concat<<<num_blocks, block_size_1d>>>(1, device_z, device_x2, device_y2, x2_len);
    dot_product<<<num_blocks, block_size_1d>>>(1, device_res, device_z, device_dense_weights, x2_len);

    // Memory copy from device to host
    cudaMemcpy(host_res, device_res, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_x);
    cudaFree(device_x1);
    cudaFree(device_x2);
    cudaFree(device_y);
    cudaFree(device_y1);
    cudaFree(device_y2);
    cudaFree(device_kernel_1);
    cudaFree(device_kernel_2);
    cudaFree(device_kernel_3);
    cudaFree(device_kernel_4);
    cudaFree(device_z);
    cudaFree(device_dense_weights);
    cudaFree(device_res);
    cudaFree(device_x11);
    cudaFree(device_y11);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("b10 device execution time: %ld ms\n", duration.count());

    // Free host memory
    free(host_x);
    free(host_y);
    free(host_kernel_1);
    free(host_kernel_2);
    free(host_kernel_3);
    free(host_kernel_4);
    free(host_dense_weights);
    free(host_res);

    // Print
    printf("b10 finished running\n");

    return 0;
}
