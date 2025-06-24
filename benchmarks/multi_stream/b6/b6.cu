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

__global__ void nb_1(int n, float * z, const int* x, const float* y, int size, int n_feat,
                     int n_classes) {  // n = 1
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
        i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_classes; j++) {
            for (int q = 0; q < n_feat; q++) {
                z[i * n_classes + j] += x[i * n_feat + q] * y[j * n_feat + q];
            }
        }
    }
}

__global__ void nb_2(int n, float* y, const float * x, int n_row_x,
                     int n_col_x) {  // n=1
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x;
        i += blockDim.x * gridDim.x) {
        float curr_max = x[i * n_col_x];
        for (int j = 0; j < n_col_x; j++) {
            curr_max = fmaxf(curr_max, x[i * n_col_x + j]);
        }
        y[i] = curr_max;
    }
}

__global__ void nb_3(int n, float* z, const float * x, const float* y, int n_row_x,
                     int n_col_x) {  // n=1
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x;
        i += blockDim.x * gridDim.x) {
        float sum = 0;
        for (int j = 0; j < n_col_x; j++) {
            sum += expf(x[i * n_col_x + j] - y[i]);
        }
        z[i] = logf(sum) + y[i];
    }
}

__global__ void nb_4(int n, float * x, const float* y, int n_row_x,
                     int n_col_x) {  // n=1
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x;
        i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_col_x; j++) {
            x[i * n_col_x + j] = expf(x[i * n_col_x + j] - y[i]);
        }
    }
}

__global__ void rr_1(int n, float* y, const int* x, int n_row_x,
                     int n_col_x) {  // n=1
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_col_x;
        j += blockDim.x * gridDim.x) {
        float feature_mean = 0;
        float sum_sq = 0;
        // Compute mean and variance;
        for (int i = 0; i < n_row_x; i++) {
            float x_tmp = x[j * n_row_x + i];
            feature_mean += x_tmp;
            sum_sq += x_tmp * x_tmp;
        }
        feature_mean /= n_row_x;
        float std = sqrtf(sum_sq / n_row_x - feature_mean * feature_mean);

        // Update values;
        for (int i = 0; i < n_row_x; i++) {
            y[j * n_row_x + i] = (x[j * n_row_x + i] - feature_mean) / std;
        }
    }
}

__global__ void rr_2(int n, float * z, const float* x, const float* y, int size, int n_feat,
                     int n_classes) {  // n=1
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
        i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_classes; j++) {
            for (int q = 0; q < n_feat; q++) {
                z[i * n_classes + j] += x[i * n_feat + q] * y[j * n_feat + q];
            }
        }
    }
}

__global__ void rr_3(int n, float * x, const float* y, int n_row_x,
                     int n_col_x) {  // n=1
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x;
        i += blockDim.x * gridDim.x) {
        for (int j = 0; j < n_col_x; j++) {
            x[i * n_col_x + j] += y[j];
        }
    }
}

__global__ void softmax(int n, float * x, int n_row_x, int n_col_x) {  // n=1
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x;
        i += blockDim.x * gridDim.x) {
        float row_exp_sum = 0;
        for (int j = 0; j < n_col_x; j++) {
            row_exp_sum += expf(x[i * n_col_x + j]);
        }
        for (int j = 0; j < n_col_x; j++) {
            x[i * n_col_x + j] = expf(x[i * n_col_x + j]) / row_exp_sum;
        }
    }
}

__global__ void argmax(int n, int* z, const float * x, const float * y, int n_row_x,
                       int n_col_x) {  // n=1
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_row_x;
        i += blockDim.x * gridDim.x) {
        int curr_best_index = 0;
        float curr_best = x[i * n_col_x] + y[i * n_col_x];
        for (int j = 0; j < n_col_x; j++) {
            float curr = x[i * n_col_x + j] + y[i * n_col_x + j];
            if (curr > curr_best) {
                curr_best = curr;
                curr_best_index = j;
            }
        }
        z[i] = curr_best_index;
    }
}

void host_initialization(float *host_nb_feat_log_prob, float *host_ridge_coeff, float *host_nb_class_log_prior, 
    float *host_ridge_intercept, int *host_x, float *host_r1, float *host_r2, int N, int num_classes, int num_features) {
    for (int i = 0; i < num_classes; i++) {
        for (int j = 0; j < num_features; j++) {
            host_nb_feat_log_prob[i * num_features + j] = (float)(rand()) / (float)(RAND_MAX);
            host_ridge_coeff[i * num_features + j] = (float)(rand()) / (float)(RAND_MAX);
        }
        host_nb_class_log_prior[i] = (float)(rand()) / (float)(RAND_MAX);
        host_ridge_intercept[i] = (float)(rand()) / (float)(RAND_MAX);
    }
    int max_occurrence_of_ngram = 10;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < num_features; j++) {
            host_x[i * num_features + j] = rand() % max_occurrence_of_ngram;
        }
        for (int j = 0; j < num_classes; j++) {
            host_r1[i * num_classes + j] = host_nb_class_log_prior[j];
            host_r2[i * num_classes + j] = (float)(rand()) / (float)(RAND_MAX);
        }
    }
}

int main() {
    // Options
    size_t N = 3500000;
    int block_size_1d = 64;
    int num_blocks = 64;
    const int num_features = 100;
    const int num_classes = 10;

    // Variable definition
    int *host_x;
    float *host_z;
    float *host_nb_feat_log_prob, *host_nb_class_log_prior, *host_ridge_coeff, *host_ridge_intercept, *host_nb_amax, *host_nb_l, *host_r1, *host_r2;
    int *host_r;
    int *device_x;
    float *device_z;
    float *device_nb_feat_log_prob, *device_ridge_coeff, *device_ridge_intercept, *device_nb_amax, *device_nb_l, *device_r1, *device_r2;
    int *device_r;

    // Host Allocation
    host_x = (int *)malloc(sizeof(int) * N * num_features);
    host_z = (float *)malloc(sizeof(float) * N * num_features);
    host_nb_feat_log_prob = (float *)malloc(sizeof(float) * num_classes * num_features);
    host_nb_class_log_prior = (float *)malloc(sizeof(float) * num_classes);
    host_ridge_coeff = (float *)malloc(sizeof(float) * num_classes * num_features);
    host_ridge_intercept = (float *)malloc(sizeof(float) * num_classes);
    host_nb_amax = (float *)malloc(sizeof(float) * N);
    host_nb_l = (float *)malloc(sizeof(float) * N);
    host_r1 = (float *)malloc(sizeof(float) * N * num_classes);
    host_r2 = (float *)malloc(sizeof(float) * N * num_classes);
    host_r = (int *)malloc(sizeof(int) * N);

    // Host initialization
    host_initialization(host_nb_feat_log_prob, host_ridge_coeff, host_nb_class_log_prior, 
        host_ridge_intercept, host_x, host_r1, host_r2, N, num_classes, num_features);

    auto start = std::chrono::high_resolution_clock::now();
    // Device allocation
    cudaMalloc(&device_x, sizeof(int) * N * num_features);
    cudaMalloc(&device_z, sizeof(float) * N * num_features);
    cudaMalloc(&device_nb_feat_log_prob, sizeof(float) * num_classes * num_features);
    cudaMalloc(&device_ridge_coeff, sizeof(float) * num_classes * num_features);
    cudaMalloc(&device_ridge_intercept, sizeof(float) * num_classes);
    cudaMalloc(&device_nb_amax, sizeof(float) * N);
    cudaMalloc(&device_nb_l, sizeof(float) * N);
    cudaMalloc(&device_r1, sizeof(float) * N * num_classes);
    cudaMalloc(&device_r2, sizeof(float) * N * num_classes);
    cudaMalloc(&device_r, sizeof(int) * N);

    // Memory copy from host to device
    cudaMemcpy(device_x, host_x, sizeof(int) * N * num_features, cudaMemcpyHostToDevice);
    cudaMemcpy(device_z, host_z, sizeof(float) * N * num_features, cudaMemcpyHostToDevice);
    cudaMemcpy(device_r1, host_r1, sizeof(float) * N * num_classes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_r2, host_r2, sizeof(float) * N * num_classes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_nb_feat_log_prob, host_nb_feat_log_prob, sizeof(float) * num_classes * num_features, cudaMemcpyHostToDevice);
    cudaMemcpy(device_ridge_coeff, host_ridge_coeff, sizeof(float) * num_classes * num_features, cudaMemcpyHostToDevice);
    cudaMemcpy(device_nb_amax, host_nb_amax, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_nb_l, host_nb_l, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_ridge_intercept, host_ridge_intercept, sizeof(float) * num_classes, cudaMemcpyHostToDevice);

    // Execution
    rr_1<<<num_blocks, block_size_1d>>>(1, device_z, device_x, N, num_features);
    nb_1<<<num_blocks, block_size_1d>>>(1, device_r1, device_x, device_nb_feat_log_prob, N,
                                        num_features, num_classes);
    rr_2<<<num_blocks, block_size_1d>>>(1, device_r2, device_z, device_ridge_coeff, N, num_features,
                                        num_classes);
    nb_2<<<num_blocks, block_size_1d>>>(1, device_nb_amax, device_r1, N, num_classes);
    nb_3<<<num_blocks, block_size_1d>>>(1, device_nb_l, device_r1, device_nb_amax, N, num_classes);
    rr_3<<<num_blocks, block_size_1d>>>(1, device_r2, device_ridge_intercept, N, num_classes);
    nb_4<<<num_blocks, block_size_1d>>>(1, device_r1, device_nb_l, N, num_classes);
    softmax<<<num_blocks, block_size_1d>>>(1, device_r1, N, num_classes);
    softmax<<<num_blocks, block_size_1d>>>(1, device_r2, N, num_classes);
    argmax<<<num_blocks, block_size_1d>>>(1, device_r, device_r1, device_r2, N, num_classes);
    
    // Memory copy from device to host
    cudaMemcpy(host_r, device_r, sizeof(int) * N, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_x);
    cudaFree(device_z);
    cudaFree(device_nb_feat_log_prob);
    cudaFree(device_ridge_coeff);
    cudaFree(device_ridge_intercept);
    cudaFree(device_nb_amax);
    cudaFree(device_nb_l);
    cudaFree(device_r1);
    cudaFree(device_r2);
    cudaFree(device_r);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("b6 device execution time: %ld ms\n", duration.count());

    // Free host memory
    free(host_x);
    free(host_z);
    free(host_nb_feat_log_prob);
    free(host_nb_class_log_prior);
    free(host_ridge_coeff);
    free(host_ridge_intercept);
    free(host_nb_amax);
    free(host_nb_l);
    free(host_r1);
    free(host_r2);
    free(host_r);

    // Print
    printf("b6 finished running\n");

    return 0;
}
