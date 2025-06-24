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

const int loop_num = 1;

__inline__ __device__ float silu(float x)
{
    return (float)((float)x / (1.0f + __expf((float)-x)));
}

__forceinline__ __device__ float copysignf_pos(float a, float b)
{
    float r;
    r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
    return r;
}

__inline__ __device__ float tanh_opt(float x)
{
#if (__CUDA_ARCH__ >= 750 && CUDART_VERSION >= 11000)
    float r;
    asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
    return r;
#else
    const float exp_val = -1.f * fabs(2 * x);
    return copysignf_pos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
#endif
}

__inline__ __device__ float gelu(float x)
{
    float cdf = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * (x + 0.044715f * x * x * x))));
    return x * cdf;
}

__global__ void add_bias_silu(int output_n, float* out, float* bias, int N)    //output_n = 1
{
    for (int k = 0; k < loop_num; k ++) {
        for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < N; id += blockDim.x * gridDim.x) {
            float val = out[id];
            val = val + __ldg(&bias[id]);
            
            out[id] = silu(val);
        }
    }
}

__global__ void
addBiasGatedRelu(int output_n, float* hidden1, float* hidden2,        //hiden2 is from silu
                float* bias1, float* bias2, int N)                //output_n = 1
{
    for (int k = 0; k < loop_num; k ++) {
        for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < N; id += blockDim.x * gridDim.x) {
            float val1 = hidden1[id];
            float val2 = hidden2[id];
            float reg_bias1 = __ldg(&bias1[id]);
            float reg_bias2 = __ldg(&bias2[id]);
            val1 += reg_bias1;
            val2 += reg_bias2;
            hidden1[id] = val1 > (float)0.0f ? val1 * val2 : (float)0.0f;
            hidden1[id] = gelu(hidden1[id]);
        }
    }
}

__global__ void
addBiasResidual(int output_n,                       //output_n = 1
                float * output,
                float * input1, float * input2, float * input3,
                const int N)
{
    for (int k = 0; k < loop_num; k ++) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
        i += blockDim.x * gridDim.x) {
            output[i] = gelu(input1[i] + input2[i] + input3[i]) / 3;
        }
    }
}

int main() {
    // Options
    const size_t N = 200000000;
    const int block_size_1d = 32;
    const int num_blocks = 16;

    // Variable definition
    float *gpu_zeros;
    float *host_gelu_out_1, *host_gelu_out_2;
    // float *host_relu_out_1, *host_relu_out_2, *host_relu_out_3;
    // float *host_relu_out_4, *host_relu_out_5, *host_relu_out_6;
    float *host_silu_bias_1, *host_silu_bias_2, *host_silu_bias_3, *host_silu_bias_4;
    float *host_silu_bias_5, *host_silu_bias_6;
    float *device_silu_out_1, *device_silu_out_2, *device_silu_out_3, *device_silu_out_4;
    float *device_silu_out_5, *device_silu_out_6;
    float *device_silu_bias_1, *device_silu_bias_2, *device_silu_bias_3, *device_silu_bias_4;
    float *device_silu_bias_5, *device_silu_bias_6;
    float *device_relu_out_1, *device_relu_out_2, *device_relu_out_3;
    float *device_relu_out_4, *device_relu_out_5, *device_relu_out_6;
    float *device_gelu_out_1, *device_gelu_out_2;
    float *device_relu_bias1, *device_relu_bias2;

    // Host allocation
    gpu_zeros = (float *)malloc(sizeof(float) * N);
    host_gelu_out_1 = (float *)malloc(sizeof(float) * N);
    host_gelu_out_2 = (float *)malloc(sizeof(float) * N);
    host_silu_bias_1 = (float *)malloc(sizeof(float) * N);
    host_silu_bias_2 = (float *)malloc(sizeof(float) * N);
    host_silu_bias_3 = (float *)malloc(sizeof(float) * N);
    host_silu_bias_4 = (float *)malloc(sizeof(float) * N);
    host_silu_bias_5 = (float *)malloc(sizeof(float) * N);
    host_silu_bias_6 = (float *)malloc(sizeof(float) * N);
    // host_relu_out_3 = (float *)malloc(sizeof(float) * N);
    // host_relu_out_4 = (float *)malloc(sizeof(float) * N);
    // host_relu_out_5 = (float *)malloc(sizeof(float) * N);
    // host_relu_out_6 = (float *)malloc(sizeof(float) * N);

    // Host initialization
    for (int i = 0; i < N; i++) {
        float tmp_float = (float)(rand()) / (float)(RAND_MAX);
        host_silu_bias_1[i] = tmp_float;
        host_silu_bias_2[i] = tmp_float;
        host_silu_bias_3[i] = tmp_float;
        host_silu_bias_4[i] = tmp_float;
        host_silu_bias_5[i] = tmp_float;
        host_silu_bias_6[i] = tmp_float;
        gpu_zeros[i] = 0;
    }

    auto start = std::chrono::high_resolution_clock::now();
    // Device allocation
    cudaMalloc(&device_silu_out_1, sizeof(float) * N);
    cudaMalloc(&device_silu_out_2, sizeof(float) * N);
    cudaMalloc(&device_silu_out_3, sizeof(float) * N);
    cudaMalloc(&device_silu_out_4, sizeof(float) * N);
    cudaMalloc(&device_silu_out_5, sizeof(float) * N);
    cudaMalloc(&device_silu_out_6, sizeof(float) * N);
    cudaMalloc(&device_silu_bias_1, sizeof(float) * N);
    cudaMalloc(&device_silu_bias_2, sizeof(float) * N);
    cudaMalloc(&device_silu_bias_3, sizeof(float) * N);
    cudaMalloc(&device_silu_bias_4, sizeof(float) * N);
    cudaMalloc(&device_silu_bias_5, sizeof(float) * N);
    cudaMalloc(&device_silu_bias_6, sizeof(float) * N);
    cudaMalloc(&device_relu_out_1, sizeof(float) * N);
    cudaMalloc(&device_relu_out_2, sizeof(float) * N);
    cudaMalloc(&device_relu_out_3, sizeof(float) * N);
    cudaMalloc(&device_relu_out_4, sizeof(float) * N);
    cudaMalloc(&device_relu_out_5, sizeof(float) * N);
    cudaMalloc(&device_relu_out_6, sizeof(float) * N);
    cudaMalloc(&device_relu_bias1, sizeof(float) * N);
    cudaMalloc(&device_relu_bias2, sizeof(float) * N);
    cudaMalloc(&device_gelu_out_1, sizeof(float) * N);
    cudaMalloc(&device_gelu_out_2, sizeof(float) * N);

    // Memory copy from host to device
    // cudaMemcpy(device_silu_out_1, gpu_mod10, sizeof(float) * N, cudaMemcpyHostToDevice);
    // cudaMemcpy(device_silu_out_2, gpu_mod10, sizeof(float) * N, cudaMemcpyHostToDevice);
    // cudaMemcpy(device_silu_out_3, gpu_mod10, sizeof(float) * N, cudaMemcpyHostToDevice);
    // cudaMemcpy(device_silu_out_4, gpu_mod10, sizeof(float) * N, cudaMemcpyHostToDevice);
    // cudaMemcpy(device_silu_out_5, gpu_mod10, sizeof(float) * N, cudaMemcpyHostToDevice);
    // cudaMemcpy(device_silu_out_6, gpu_mod10, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_silu_bias_1, host_silu_bias_1, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_silu_bias_2, host_silu_bias_2, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_silu_bias_3, host_silu_bias_3, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_silu_bias_4, host_silu_bias_4, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_silu_bias_5, host_silu_bias_5, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_silu_bias_6, host_silu_bias_6, sizeof(float) * N, cudaMemcpyHostToDevice);
    // cudaMemcpy(device_relu_out_1, gpu_zeros, sizeof(float) * N, cudaMemcpyHostToDevice);
    // cudaMemcpy(device_relu_out_2, gpu_zeros, sizeof(float) * N, cudaMemcpyHostToDevice);
    // cudaMemcpy(device_relu_out_3, gpu_zeros, sizeof(float) * N, cudaMemcpyHostToDevice);
    // cudaMemcpy(device_relu_out_4, gpu_zeros, sizeof(float) * N, cudaMemcpyHostToDevice);
    // cudaMemcpy(device_relu_out_5, gpu_zeros, sizeof(float) * N, cudaMemcpyHostToDevice);
    // cudaMemcpy(device_relu_out_6, gpu_zeros, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_relu_bias1, gpu_zeros, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_relu_bias2, gpu_zeros, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Execution
    add_bias_silu<<<num_blocks, block_size_1d>>>(1, device_silu_out_1, device_silu_bias_1, N);
    add_bias_silu<<<num_blocks, block_size_1d>>>(1, device_silu_out_2, device_silu_bias_2, N);
    add_bias_silu<<<num_blocks, block_size_1d>>>(1, device_silu_out_3, device_silu_bias_3, N);
    add_bias_silu<<<num_blocks, block_size_1d>>>(1, device_silu_out_4, device_silu_bias_4, N);
    add_bias_silu<<<num_blocks, block_size_1d>>>(1, device_silu_out_5, device_silu_bias_5, N);
    add_bias_silu<<<num_blocks, block_size_1d>>>(1, device_silu_out_6, device_silu_bias_6, N);

    addBiasGatedRelu<<<num_blocks, block_size_1d>>>(1, device_relu_out_1, device_silu_out_1, device_relu_bias1, device_relu_bias2, N);
    addBiasGatedRelu<<<num_blocks, block_size_1d>>>(1, device_relu_out_2, device_silu_out_2, device_relu_bias1, device_relu_bias2, N);
    addBiasGatedRelu<<<num_blocks, block_size_1d>>>(1, device_relu_out_3, device_silu_out_3, device_relu_bias1, device_relu_bias2, N);
    addBiasGatedRelu<<<num_blocks, block_size_1d>>>(1, device_relu_out_4, device_silu_out_4, device_relu_bias1, device_relu_bias2, N);
    addBiasGatedRelu<<<num_blocks, block_size_1d>>>(1, device_relu_out_5, device_silu_out_5, device_relu_bias1, device_relu_bias2, N);
    addBiasGatedRelu<<<num_blocks, block_size_1d>>>(1, device_relu_out_6, device_silu_out_6, device_relu_bias1, device_relu_bias2, N);

    addBiasResidual<<<num_blocks, block_size_1d>>>(1, device_gelu_out_1, device_relu_out_1, device_relu_out_2, device_relu_out_3, N);
    addBiasResidual<<<num_blocks, block_size_1d>>>(1, device_gelu_out_2, device_relu_out_4, device_relu_out_5, device_relu_out_6, N);

    // Memory copy from device to host
    cudaMemcpy(host_gelu_out_1, device_gelu_out_1, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_gelu_out_2, device_gelu_out_2, sizeof(float) * N, cudaMemcpyDeviceToHost);
    // cudaMemcpy(host_relu_out_1, device_relu_out_1, sizeof(float) * N, cudaMemcpyDeviceToHost);
    // cudaMemcpy(host_relu_out_2, device_relu_out_2, sizeof(float) * N, cudaMemcpyDeviceToHost);
    // cudaMemcpy(host_relu_out_3, device_relu_out_3, sizeof(float) * N, cudaMemcpyDeviceToHost);
    // cudaMemcpy(host_relu_out_4, device_relu_out_4, sizeof(float) * N, cudaMemcpyDeviceToHost);
    // cudaMemcpy(host_relu_out_5, device_relu_out_5, sizeof(float) * N, cudaMemcpyDeviceToHost);
    // cudaMemcpy(host_relu_out_6, device_relu_out_6, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_silu_out_1);
    cudaFree(device_silu_out_2);
    cudaFree(device_silu_out_3);
    cudaFree(device_silu_out_4);
    cudaFree(device_silu_out_5);
    cudaFree(device_silu_out_6);
    cudaFree(device_silu_bias_1);
    cudaFree(device_silu_bias_2);
    cudaFree(device_silu_bias_3);
    cudaFree(device_silu_bias_4);
    cudaFree(device_silu_bias_5);
    cudaFree(device_silu_bias_6);
    cudaFree(device_relu_out_1);
    cudaFree(device_relu_out_2);
    cudaFree(device_relu_out_3);
    cudaFree(device_relu_out_4);
    cudaFree(device_relu_out_5);
    cudaFree(device_relu_out_6);
    cudaFree(device_relu_bias1);
    cudaFree(device_relu_bias2);
    cudaFree(device_gelu_out_1);
    cudaFree(device_gelu_out_2);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("b15 device execution time: %ld ms\n", duration.count());

    // Free host memory
    free(gpu_zeros);
    free(host_gelu_out_1);
    free(host_gelu_out_2);
    free(host_silu_bias_1);
    free(host_silu_bias_2);
    free(host_silu_bias_3);
    free(host_silu_bias_4);
    free(host_silu_bias_5);
    free(host_silu_bias_6);

    // Print
    printf("b15 finished running\n");

    return 0;
}
