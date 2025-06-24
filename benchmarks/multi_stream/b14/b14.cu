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

__global__ void addBiasGelu(int output_n, float * out, const float * bias, int N)
{
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < N; id += blockDim.x * gridDim.x) {
        float val = out[id];
        
        float reg_bias = __ldg(&bias[id]);
        val        = val + reg_bias;
        out[id] = (float)(gelu(val));
        // out[id] = val;                                 //This can slow the sync scheme and gain higher speedup in CKE
    }
}

__global__ void
addBiasResidual(int output_n,                       //output_n = 1
                float * output,
                float * input1, float * input2, 
                const float * residual1,
                const float * bias,
                const int N)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
        output[i] =
            (gelu(input1[i] + residual1[i] + bias[i]) + gelu(input2[i] + residual1[i] + bias[i])) / 2;
    }
}

__global__ void
geluReduceAverage(int output_n,                       //output_n = 1
                float * output,
                float * input1, float * input2, 
                float * input3, float * input4,
                const int N)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
        output[i] = gelu(input1[i] + input2[i] + input3[i] + input4[i]) / 4;
    }
}

int main() {
    // Options
    size_t N = 160000000;
    int block_size_1d = 32;
    int num_blocks = 16;

    // Variable definition
    float *zeros_gpu;
    float *host_gelu_input_1_1, *host_gelu_input_1_2, *host_gelu_input_1_3, *host_gelu_input_1_4;
    float *host_gelu_input_1_5, *host_gelu_input_1_6, *host_gelu_input_1_7, *host_gelu_input_1_8;
    float *host_reduce_output_4;
    float *device_gelu_output_1_1, *device_gelu_output_1_2, *device_gelu_output_1_3, *device_gelu_output_1_4;
    float *device_gelu_output_1_5, *device_gelu_output_1_6, *device_gelu_output_1_7, *device_gelu_output_1_8;
    float *device_gelu_output_3_1, *device_gelu_output_3_2, *device_gelu_output_3_3, *device_gelu_output_3_4;
    float *device_reduce_output_4;
    float *device_gelu_input1, *device_gelu_input2, *device_gelu_input3, *device_gelu_input4, *device_gelu_input5, *device_gelu_input6;
    float *device_gelu_input7, *device_gelu_input8;
    float *device_add_output1, *device_add_output2, *device_add_output3, *device_add_output4;
    float *device_residual1, *device_residual2, *device_residual3, *device_residual4;
    float *device_add_bias;

    // Host allocation
    zeros_gpu = (float *)malloc(sizeof(float) * N);
    host_gelu_input_1_1 = (float *)malloc(sizeof(float) * N);
    host_gelu_input_1_2 = (float *)malloc(sizeof(float) * N);
    host_gelu_input_1_3 = (float *)malloc(sizeof(float) * N);
    host_gelu_input_1_4 = (float *)malloc(sizeof(float) * N);
    host_gelu_input_1_5 = (float *)malloc(sizeof(float) * N);
    host_gelu_input_1_6 = (float *)malloc(sizeof(float) * N);
    host_gelu_input_1_7 = (float *)malloc(sizeof(float) * N);
    host_gelu_input_1_8 = (float *)malloc(sizeof(float) * N);
    // host_gelu_output_3_1 = (float *)malloc(sizeof(float) * N);
    // host_gelu_output_3_2 = (float *)malloc(sizeof(float) * N);
    // host_gelu_output_3_3 = (float *)malloc(sizeof(float) * N);
    // host_gelu_output_3_4 = (float *)malloc(sizeof(float) * N);
    host_reduce_output_4 = (float *)malloc(sizeof(float) * N);

    // Host initialization
    for (int i = 0; i < N; i++) {
        float tmp_float = (float)(rand()) / (float)(RAND_MAX);
        host_gelu_input_1_1[i] = tmp_float;
        host_gelu_input_1_2[i] = tmp_float;
        host_gelu_input_1_3[i] = tmp_float;
        host_gelu_input_1_4[i] = tmp_float;
        host_gelu_input_1_5[i] = tmp_float;
        host_gelu_input_1_6[i] = tmp_float;
        host_gelu_input_1_7[i] = tmp_float;
        host_gelu_input_1_8[i] = tmp_float;
        zeros_gpu[i] = 0;
    }

    auto start = std::chrono::high_resolution_clock::now();
    // Device allocation
    cudaMalloc(&device_gelu_input1, sizeof(float) * N);
    cudaMalloc(&device_gelu_input2, sizeof(float) * N);
    cudaMalloc(&device_gelu_input3, sizeof(float) * N);
    cudaMalloc(&device_gelu_input4, sizeof(float) * N);
    cudaMalloc(&device_gelu_input5, sizeof(float) * N);
    cudaMalloc(&device_gelu_input6, sizeof(float) * N);
    cudaMalloc(&device_gelu_input7, sizeof(float) * N);
    cudaMalloc(&device_gelu_input8, sizeof(float) * N);
    cudaMalloc(&device_gelu_output_1_1, sizeof(float) * N);
    cudaMalloc(&device_gelu_output_1_2, sizeof(float) * N);
    cudaMalloc(&device_gelu_output_1_3, sizeof(float) * N);
    cudaMalloc(&device_gelu_output_1_4, sizeof(float) * N);
    cudaMalloc(&device_gelu_output_1_5, sizeof(float) * N);
    cudaMalloc(&device_gelu_output_1_6, sizeof(float) * N);
    cudaMalloc(&device_gelu_output_1_7, sizeof(float) * N);
    cudaMalloc(&device_gelu_output_1_8, sizeof(float) * N);
    cudaMalloc(&device_gelu_output_3_1, sizeof(float) * N);
    cudaMalloc(&device_gelu_output_3_2, sizeof(float) * N);
    cudaMalloc(&device_gelu_output_3_3, sizeof(float) * N);
    cudaMalloc(&device_gelu_output_3_4, sizeof(float) * N);
    cudaMalloc(&device_add_output1, sizeof(float) * N);
    cudaMalloc(&device_add_output2, sizeof(float) * N);
    cudaMalloc(&device_add_output3, sizeof(float) * N);
    cudaMalloc(&device_add_output4, sizeof(float) * N);
    cudaMalloc(&device_residual1, sizeof(float) * N);
    cudaMalloc(&device_residual2, sizeof(float) * N);
    cudaMalloc(&device_residual3, sizeof(float) * N);
    cudaMalloc(&device_residual4, sizeof(float) * N);
    cudaMalloc(&device_add_bias, sizeof(float) * N);
    cudaMalloc(&device_reduce_output_4, sizeof(float) * N);

    // Memory copy from host to device
    cudaMemcpy(device_gelu_input1, host_gelu_input_1_1, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_gelu_input2, host_gelu_input_1_2, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_gelu_input3, host_gelu_input_1_3, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_gelu_input4, host_gelu_input_1_4, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_gelu_input5, host_gelu_input_1_5, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_gelu_input6, host_gelu_input_1_6, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_gelu_input7, host_gelu_input_1_7, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_gelu_input8, host_gelu_input_1_8, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_residual1, zeros_gpu, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_add_bias, zeros_gpu, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Execution
    addBiasGelu<<<num_blocks, block_size_1d>>>(1, device_gelu_output_1_1, device_gelu_input1, N);
    addBiasGelu<<<num_blocks, block_size_1d>>>(1, device_gelu_output_1_2, device_gelu_input2, N);
    addBiasGelu<<<num_blocks, block_size_1d>>>(1, device_gelu_output_1_3, device_gelu_input3, N);
    addBiasGelu<<<num_blocks, block_size_1d>>>(1, device_gelu_output_1_4, device_gelu_input4, N);
    addBiasGelu<<<num_blocks, block_size_1d>>>(1, device_gelu_output_1_5, device_gelu_input5, N);
    addBiasGelu<<<num_blocks, block_size_1d>>>(1, device_gelu_output_1_6, device_gelu_input6, N);
    addBiasGelu<<<num_blocks, block_size_1d>>>(1, device_gelu_output_1_7, device_gelu_input7, N);
    addBiasGelu<<<num_blocks, block_size_1d>>>(1, device_gelu_output_1_8, device_gelu_input8, N);

    addBiasResidual<<<num_blocks, block_size_1d>>>(1, device_add_output1,
        device_gelu_output_1_1, device_gelu_output_1_2, device_residual1, device_add_bias, N);
    addBiasResidual<<<num_blocks, block_size_1d>>>(1, device_add_output2,
        device_gelu_output_1_3, device_gelu_output_1_4, device_residual2, device_add_bias, N);
    addBiasResidual<<<num_blocks, block_size_1d>>>(1, device_add_output3,
        device_gelu_output_1_5, device_gelu_output_1_6, device_residual3, device_add_bias, N);
    addBiasResidual<<<num_blocks, block_size_1d>>>(1, device_add_output4,
        device_gelu_output_1_7, device_gelu_output_1_8, device_residual4, device_add_bias, N);

    addBiasGelu<<<num_blocks, block_size_1d>>>(1, device_gelu_output_3_1, device_add_output1, N);
    addBiasGelu<<<num_blocks, block_size_1d>>>(1, device_gelu_output_3_2, device_add_output2, N);
    addBiasGelu<<<num_blocks, block_size_1d>>>(1, device_gelu_output_3_3, device_add_output3, N);
    addBiasGelu<<<num_blocks, block_size_1d>>>(1, device_gelu_output_3_4, device_add_output4, N);

    geluReduceAverage<<<num_blocks, block_size_1d>>>(1, device_reduce_output_4,
        device_gelu_output_3_1, device_gelu_output_3_2, device_gelu_output_3_3, device_gelu_output_3_4, N);

    // Memory copy from device to host
    // cudaMemcpy(host_gelu_output_3_1, device_gelu_output_3_1, sizeof(float) * N, cudaMemcpyDeviceToHost);
    // cudaMemcpy(host_gelu_output_3_2, device_gelu_output_3_2, sizeof(float) * N, cudaMemcpyDeviceToHost);
    // cudaMemcpy(host_gelu_output_3_3, device_gelu_output_3_3, sizeof(float) * N, cudaMemcpyDeviceToHost);
    // cudaMemcpy(host_gelu_output_3_4, device_gelu_output_3_4, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_reduce_output_4, device_reduce_output_4, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_gelu_output_1_1);
    cudaFree(device_gelu_output_1_2);
    cudaFree(device_gelu_output_1_3);
    cudaFree(device_gelu_output_1_4);
    cudaFree(device_gelu_output_1_5);
    cudaFree(device_gelu_output_1_6);
    cudaFree(device_gelu_output_1_7);
    cudaFree(device_gelu_output_1_8);
    cudaFree(device_gelu_input1);
    cudaFree(device_gelu_input2);
    cudaFree(device_gelu_input3);
    cudaFree(device_gelu_input4);
    cudaFree(device_gelu_input5);
    cudaFree(device_gelu_input6);
    cudaFree(device_gelu_input7);
    cudaFree(device_gelu_input8);
    cudaFree(device_gelu_output_3_1);
    cudaFree(device_gelu_output_3_2);
    cudaFree(device_gelu_output_3_3);
    cudaFree(device_gelu_output_3_4);
    cudaFree(device_add_output1);
    cudaFree(device_add_output2);
    cudaFree(device_add_output3);
    cudaFree(device_add_output4);
    cudaFree(device_residual1);
    cudaFree(device_residual2);
    cudaFree(device_residual3);
    cudaFree(device_residual4);
    cudaFree(device_add_bias);
    cudaFree(device_reduce_output_4);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("b14 device execution time: %ld ms\n", duration.count());

    // Free host memory
    // free(host_gelu_output_3_1);
    // free(host_gelu_output_3_2);
    // free(host_gelu_output_3_3);
    // free(host_gelu_output_3_4);
    free(host_gelu_input_1_1);
    free(host_gelu_input_1_2);
    free(host_gelu_input_1_3);
    free(host_gelu_input_1_4);
    free(host_gelu_input_1_5);
    free(host_gelu_input_1_6);
    free(host_gelu_input_1_7);
    free(host_gelu_input_1_8);
    free(host_reduce_output_4);
    free(zeros_gpu);

    // Print
    printf("b14 finished running\n");

    return 0;
}
