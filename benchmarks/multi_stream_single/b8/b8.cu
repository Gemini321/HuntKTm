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

const int test_num = 10;

// __global__ void gaussian_blur(const float *image, float *result, int rows,
// int cols, const float *kernel, int diameter) {
__global__ void gaussian_blur(int n, float *result, float * image, float *kernel, int rows,
                              int cols, int diameter) {  // n=1
//   extern __shared__ float kernel_local[];
//   for (int i = threadIdx.x; i < diameter; i += blockDim.x) {
//     for (int j = threadIdx.y; j < diameter; j += blockDim.y) {
//       kernel_local[i * diameter + j] = kernel[i * diameter + j];
//     }
//   }
//   __syncthreads();

//   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows;
//        i += blockDim.x * gridDim.x) {
//     for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < cols;
//          j += blockDim.y * gridDim.y) {
//       float sum = 0;
//       int radius = diameter / 2;
//       for (int x = -radius; x <= radius; ++x) {
//         for (int y = -radius; y <= radius; ++y) {
//           int nx = x + i;
//           int ny = y + j;
//           if (nx >= 0 && ny >= 0 && nx < rows && ny < cols) {
//             sum += kernel_local[(x + radius) * diameter + (y + radius)] *
//                    image[nx * cols + ny];
//           }
//         }
//       }
//       result[i * cols + j] = sum;
//     }
//   }

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows;
       i += blockDim.x * gridDim.x) {
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < cols;
         j += blockDim.y * gridDim.y) {
      float sum = 0;
      int radius = diameter / 2;
      for (int x = -radius; x <= radius; ++x) {
        for (int y = -radius; y <= radius; ++y) {
          int nx = x + i;
          int ny = y + j;
          if (nx >= 0 && ny >= 0 && nx < rows && ny < cols) {
            sum += kernel[(x + radius) * diameter + (y + radius)] *
                   image[nx * cols + ny];
          }
        }
      }
      result[i * cols + j] = sum;
    }
  }
}

// __global__ void sobel(const float *image, float *result, int rows, int cols)
// {
__global__ void sobel(int n, float * result, float *image, int rows,
                      int cols) {  // n=1
  // int SOBEL_X[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
  // int SOBEL_Y[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  __shared__ int SOBEL_X[9];
  __shared__ int SOBEL_Y[9];
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    SOBEL_X[0] = -1;
    SOBEL_X[1] = -2;
    SOBEL_X[2] = -1;
    SOBEL_X[3] = 0;
    SOBEL_X[4] = 0;
    SOBEL_X[5] = 0;
    SOBEL_X[6] = 1;
    SOBEL_X[7] = 2;
    SOBEL_X[8] = 1;

    SOBEL_Y[0] = -1;
    SOBEL_Y[1] = 0;
    SOBEL_Y[2] = 1;
    SOBEL_Y[3] = -2;
    SOBEL_Y[4] = 0;
    SOBEL_Y[5] = 2;
    SOBEL_Y[6] = -1;
    SOBEL_Y[7] = 0;
    SOBEL_Y[8] = 1;
  }
  __syncthreads();

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows;
       i += blockDim.x * gridDim.x) {
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < cols;
         j += blockDim.y * gridDim.y) {
      float sum_gradient_x = 0.0, sum_gradient_y = 0.0;
      int radius = 1;
      for (int x = -radius; x <= radius; ++x) {
        for (int y = -radius; y <= radius; ++y) {
          int nx = x + i;
          int ny = y + j;
          if (nx >= 0 && ny >= 0 && nx < rows && ny < cols) {
            float neighbour = image[nx * cols + ny];
            int s = (x + radius) * 3 + y + radius;
            sum_gradient_x += SOBEL_X[s] * neighbour;
            sum_gradient_y += SOBEL_Y[s] * neighbour;
          }
        }
      }
      result[i * cols + j] = sqrt(sum_gradient_x * sum_gradient_x +
                                  sum_gradient_y * sum_gradient_y);
    }
  }
}

__device__ float atomicMinf(float *address, float val) {
  int *address_as_int = (int *)address;
  int old = *address_as_int, assumed;
  while (val < __int_as_float(old)) {
    assumed = old;
    old = atomicCAS(address_as_int, assumed, __float_as_int(val));
  }
  return __int_as_float(old);
}

__device__ float atomicMaxf(float *address, float val) {
  int *address_as_int = (int *)address;
  int old = *address_as_int, assumed;
  // If val is smaller than current, don't do anything, else update the current
  // value atomically;
  while (val > __int_as_float(old)) {
    assumed = old;
    old = atomicCAS(address_as_int, assumed, __float_as_int(val));
  }
  return __int_as_float(old);
}

__inline__ __device__ float warp_reduce_max(float val) {
  int warp_size = 32;
  for (int offset = warp_size / 2; offset > 0; offset /= 2)
    val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  return val;
}

__inline__ __device__ float warp_reduce_min(float val) {
  int warp_size = 32;
  for (int offset = warp_size / 2; offset > 0; offset /= 2)
    val = min(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  return val;
}

// __global__ void maximum_kernel(const float *in, float *out, int N) {
__global__ void maximum_kernel(int n, float *out, float * in, int N) {  // n=1
  int warp_size = 32;
  float maximum = -1000;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    maximum = max(maximum, in[i]);
  }
  maximum = warp_reduce_max(
      maximum);  // Obtain the max of values in the current warp;
  if ((threadIdx.x & (warp_size - 1)) ==
      0)  // Same as (threadIdx.x % warp_size) == 0 but faster
    atomicMaxf(out,
               maximum);  // The first thread in the warp updates the output;
}

// __global__ void minimum_kernel(const float *in, float *out, int N) {
__global__ void minimum_kernel(int n, float *out, float * in, int N) {  // n=1
  int warp_size = 32;
  float minimum = 1000;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    minimum = min(minimum, in[i]);
  }
  minimum = warp_reduce_min(
      minimum);  // Obtain the min of values in the current warp;
  if ((threadIdx.x & (warp_size - 1)) ==
      0)  // Same as (threadIdx.x % warp_size) == 0 but faster
    atomicMinf(out,
               minimum);  // The first thread in the warp updates the output;
}

// __global__ void extend(float *x, const float *minimum, const float *maximum,
// int n) {
__global__ void extend(int output_n, float * x, float * input, float *minimum, float *maximum,
                       int n) {  // n=1
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    float res_tmp = 5 * (input[i] - *minimum) / (*maximum - *minimum);
    x[i] = res_tmp > 1 ? 1 : res_tmp;
  }
}

// __global__ void unsharpen(const float *x, const float *y, float *res, float
// amount, int n) {
__global__ void unsharpen(int outout_n, float *res, float * x, float *y,
                          float amount, int n) {  // n=1
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    float res_tmp = x[i] * (1 + amount) - y[i] * amount;
    res_tmp = res_tmp > 1 ? 1 : res_tmp;
    res[i] = res_tmp < 0 ? 0 : res_tmp;
  }
}

// __global__ void combine(const float *x, const float *y, const float *mask,
// float *res, int n) {
__global__ void combine(int output_n, float *res, float *x, float *y,
                        float * mask, int n) {  // n=1
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    res[i] = x[i] * mask[i] + y[i] * (1 - mask[i]);
  }
}

// __global__ void reset_image(float *x, int n) {
__global__ void reset_image(int output_n, float *x, int n) {  // n=1
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    x[i] = 0.0;
  }
}

inline void gaussian_kernel(float *kernel, int diameter, float sigma) {
    int mean = diameter / 2;
    float sum_tmp = 0;
    for (int i = 0; i < diameter; i++) {
        for (int j = 0; j < diameter; j++) {
            kernel[i * diameter + j] = exp(-0.5 * ((i - mean) * (i - mean) + (j - mean) * (j - mean)) / (sigma * sigma));
            sum_tmp += kernel[i * diameter + j];
        }
    }
    for (int i = 0; i < diameter; i++) {
        for (int j = 0; j < diameter; j++) {
            kernel[i * diameter + j] /= sum_tmp;
        }
    }
}

int main() {
    // Options
    size_t N = 17000;
    const int block_size_1d = 64;
    const int block_size_2d = 8;
    const int num_blocks = 32;
    const int kernel_small_diameter = 3;
    const int kernel_large_diameter = 5;
    const int kernel_unsharpen_diameter = 3;
    const int num_blocks_div2 = num_blocks / 2;
    dim3 grid_size_2_1(num_blocks_div2, num_blocks_div2);
    dim3 block_size_2d_dim_1(block_size_2d, block_size_2d);
    dim3 grid_size_2_2(num_blocks_div2, num_blocks_div2);
    dim3 block_size_2d_dim_2(block_size_2d, block_size_2d);
    dim3 grid_size_2_3(num_blocks_div2, num_blocks_div2);
    dim3 block_size_2d_dim_3(block_size_2d, block_size_2d);
    dim3 grid_size_2_4(num_blocks_div2, num_blocks_div2);
    dim3 block_size_2d_dim_4(block_size_2d, block_size_2d);
    dim3 grid_size_2_5(num_blocks_div2, num_blocks_div2);
    dim3 block_size_2d_dim_5(block_size_2d, block_size_2d);

    // Variable definition
    float *host_image, *host_image2, *host_image3, *host_image_unsharpen, *host_mask_small, *host_mask_large, *host_blurred_small, *host_blurred_large, *host_blurred_unsharpen;
    float *host_kernel_small, *host_kernel_large, *host_kernel_unsharpen, *host_maximum, *host_minimum;
    float *device_image, *device_image2, *device_image3, *device_image_unsharpen, *device_mask_small, *device_mask_large, *device_mask_large_extend, *device_blurred_small, *device_blurred_large, *device_blurred_unsharpen;
    float *device_kernel_small, *device_kernel_large, *device_kernel_unsharpen, *device_maximum, *device_minimum;

    // Host allocation
    host_image = (float *)malloc(sizeof(float) * N * N);
    host_image2 = (float *)malloc(sizeof(float) * N * N);
    host_image3 = (float *)malloc(sizeof(float) * N * N);
    host_image_unsharpen = (float *)malloc(sizeof(float) * N * N);
    host_mask_small = (float *)malloc(sizeof(float) * N * N);
    host_mask_large = (float *)malloc(sizeof(float) * N * N);
    // host_mask_unsharpen = (float *)malloc(sizeof(float) * N * N);
    host_blurred_small = (float *)malloc(sizeof(float) * N * N);
    host_blurred_large = (float *)malloc(sizeof(float) * N * N);
    host_blurred_unsharpen = (float *)malloc(sizeof(float) * N * N);
    host_kernel_small = (float *)malloc(sizeof(float) * kernel_small_diameter * kernel_small_diameter);
    host_kernel_large = (float *)malloc(sizeof(float) * kernel_large_diameter * kernel_large_diameter);
    host_kernel_unsharpen = (float *)malloc(sizeof(float) * kernel_unsharpen_diameter * kernel_unsharpen_diameter);
    host_maximum = (float *)malloc(sizeof(float));
    host_minimum = (float *)malloc(sizeof(float));

    // Host initialization
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            host_image[i * N + j] = (float)(rand()) / (float)(RAND_MAX);
        }
    }
    gaussian_kernel(host_kernel_small, kernel_small_diameter, 1);
    gaussian_kernel(host_kernel_large, kernel_large_diameter, 10);
    gaussian_kernel(host_kernel_unsharpen, kernel_unsharpen_diameter, 5);
    *host_maximum = 0;
    *host_minimum = 0;

    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < test_num; i ++) {
        // Device allocation
        cudaMalloc(&device_image, sizeof(float) * N * N);
        cudaMalloc(&device_image2, sizeof(float) * N * N);
        cudaMalloc(&device_image3, sizeof(float) * N * N);
        cudaMalloc(&device_image_unsharpen, sizeof(float) * N * N);
        cudaMalloc(&device_mask_small, sizeof(float) * N * N);
        cudaMalloc(&device_mask_large, sizeof(float) * N * N);
        cudaMalloc(&device_mask_large_extend, sizeof(float) * N * N);
        // cudaMalloc(&device_mask_unsharpen, sizeof(float) * N * N);
        cudaMalloc(&device_blurred_small, sizeof(float) * N * N);
        cudaMalloc(&device_blurred_large, sizeof(float) * N * N);
        cudaMalloc(&device_blurred_unsharpen, sizeof(float) * N * N);

        cudaMalloc(&device_kernel_small, sizeof(float) * kernel_small_diameter *
                                                    kernel_small_diameter);
        cudaMalloc(&device_kernel_large, sizeof(float) * kernel_large_diameter *
                                                    kernel_large_diameter);
        cudaMalloc(
            &device_kernel_unsharpen,
            sizeof(float) * kernel_unsharpen_diameter * kernel_unsharpen_diameter);
        cudaMalloc(&device_maximum, sizeof(float));
        cudaMalloc(&device_minimum, sizeof(float));

        // Memory copy from host to device
        cudaMemcpy(device_blurred_small, host_blurred_small, sizeof(float) * N * N, cudaMemcpyHostToDevice);
        cudaMemcpy(device_image, host_image, sizeof(float) * N * N, cudaMemcpyHostToDevice);
        cudaMemcpy(device_kernel_small, host_kernel_small, sizeof(float) * kernel_small_diameter * kernel_small_diameter, cudaMemcpyHostToDevice);
        cudaMemcpy(device_blurred_large, host_blurred_large, sizeof(float) * N * N, cudaMemcpyHostToDevice);
        cudaMemcpy(device_kernel_large, host_kernel_large, sizeof(float) * kernel_large_diameter * kernel_large_diameter, cudaMemcpyHostToDevice);
        cudaMemcpy(device_blurred_unsharpen, host_blurred_unsharpen, sizeof(float) * N * N, cudaMemcpyHostToDevice);
        cudaMemcpy(device_kernel_unsharpen, host_kernel_unsharpen, sizeof(float) * kernel_unsharpen_diameter * kernel_unsharpen_diameter, cudaMemcpyHostToDevice);
        cudaMemcpy(device_mask_small, host_mask_small, sizeof(float) * N * N, cudaMemcpyHostToDevice);
        cudaMemcpy(device_mask_large, host_mask_large, sizeof(float) * N * N, cudaMemcpyHostToDevice);
        cudaMemcpy(device_image_unsharpen, host_image_unsharpen, sizeof(float) * N * N, cudaMemcpyHostToDevice);
        cudaMemcpy(device_image2, host_image2, sizeof(float) * N * N, cudaMemcpyHostToDevice);
        cudaMemcpy(device_image3, host_image3, sizeof(float) * N * N, cudaMemcpyHostToDevice);
        cudaMemcpy(device_maximum, host_maximum, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_minimum, host_minimum, sizeof(float), cudaMemcpyHostToDevice);

        // Execution
        // gaussian_blur<<<grid_size_2_1, block_size_2d_dim_1,
        //                 kernel_small_diameter * kernel_small_diameter * sizeof(float)>>>(
        //                 1, device_blurred_small, device_image, device_kernel_small, N, N, kernel_small_diameter);
        // gaussian_blur<<<grid_size_2_2, block_size_2d_dim_2, kernel_large_diameter * kernel_large_diameter *
        //                 sizeof(float)>>>(1, device_blurred_large, device_image, device_kernel_large, N, N, kernel_large_diameter);
        // gaussian_blur<<<grid_size_2_3, block_size_2d_dim_3, kernel_unsharpen_diameter * kernel_unsharpen_diameter *
        //                 sizeof(float)>>>(1, device_blurred_unsharpen, device_image, device_kernel_unsharpen, N, N, kernel_unsharpen_diameter);
        gaussian_blur<<<grid_size_2_1, block_size_2d_dim_1>>>(
                    1, device_blurred_small, device_image, device_kernel_small, N, N, kernel_small_diameter);
        gaussian_blur<<<grid_size_2_2, block_size_2d_dim_2>>>(1, device_blurred_large, device_image, device_kernel_large, N, N, kernel_large_diameter);
        gaussian_blur<<<grid_size_2_3, block_size_2d_dim_3>>>(1, device_blurred_unsharpen, device_image, device_kernel_unsharpen, N, N, kernel_unsharpen_diameter);
        sobel<<<grid_size_2_4, block_size_2d_dim_4>>>(1, device_mask_small, device_blurred_small, N, N);
        sobel<<<grid_size_2_5, block_size_2d_dim_5>>>(1, device_mask_large, device_blurred_large, N, N);
        maximum_kernel<<<num_blocks, block_size_1d>>>(1, device_maximum, device_mask_large, N * N);
        minimum_kernel<<<num_blocks, block_size_1d>>>(1, device_minimum, device_mask_large, N * N);
        extend<<<num_blocks, block_size_1d>>>(1, device_mask_large_extend, device_mask_large, device_minimum, device_maximum, N * N);
        unsharpen<<<num_blocks, block_size_1d>>>(1, device_image_unsharpen, device_image, device_blurred_unsharpen, 0.5, N * N);
        combine<<<num_blocks, block_size_1d>>>(1, device_image2, device_image_unsharpen, device_blurred_large, device_mask_large_extend, N * N);
        combine<<<num_blocks, block_size_1d>>>(1, device_image3, device_image2, device_blurred_small, device_mask_small, N * N);

        // Memory copy from device to host
        cudaMemcpy(host_image3, device_image3, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(device_image);
        cudaFree(device_image2);
        cudaFree(device_image3);
        cudaFree(device_image_unsharpen);
        cudaFree(device_mask_small);
        cudaFree(device_mask_large);
        cudaFree(device_mask_large_extend);
        // cudaFree(device_mask_unsharpen);
        cudaFree(device_blurred_small);
        cudaFree(device_blurred_large);
        cudaFree(device_blurred_unsharpen);
        cudaFree(device_kernel_small);
        cudaFree(device_kernel_large);
        cudaFree(device_kernel_unsharpen);
        cudaFree(device_maximum);
        cudaFree(device_minimum);

        cudaDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("b8 device execution time: %ld ms\n", duration.count());

    // Free host memory
    free(host_image);
    free(host_image2);
    free(host_image3);
    free(host_image_unsharpen);
    free(host_mask_small);
    free(host_mask_large);
    // free(host_mask_unsharpen);
    free(host_blurred_small);
    free(host_blurred_large);
    free(host_blurred_unsharpen);
    free(host_kernel_small);
    free(host_kernel_large);
    free(host_kernel_unsharpen);
    free(host_maximum);
    free(host_minimum);

    // Print
    printf("b8 finished running\n");

    return 0;
}
