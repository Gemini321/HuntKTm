#ifndef COMMON_H
#define COMMON_H

#include "runtime/wrapper/wrapper.h"
#include <cassert>
#include <bits/stdint-uintn.h>
#include <sys/mman.h>
#include <pthread.h>
#include <sys/types.h>
#include <semaphore.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <chrono>

#define COMM_FILE "/schedulercomm"
#define MAX_TASK_RUNNING 256
#define MAX_STREAMS_PER_GPU 32
#define MAX_NUM_GPUS 8
#define RESERVED_MEMORY (long)500 * 1024 * 1024 // reserved 500MB memory for MPS and system
#define TOTAL_MEMORY (long)40 * 1024 * 1024 * 1024 // restrict to 20 GB

#define CUDA_SAFE_CALL(x)                                                       \
do {                                                                            \
    cudaError_t err = x;                                                        \
    if (err != cudaSuccess) {                                                   \
        fprintf(stderr, "CUDA error %s: %s", #x, cudaGetErrorString(err));      \
        exit(1);                                                                \
    }                                                                           \
} while (0)

// Task resource requirement
struct TaskReq {
    uint32_t num_blocks;
    uint32_t num_threads_per_block;
    uint32_t num_threads;
    uint32_t num_reg;
    uint32_t num_shmem;
    int64_t memory_alloc;
    int num_stream;

    TaskReq() {}
    TaskReq(uint32_t grids, uint32_t blocks, int64_t memory, int num_stream=1):
        num_blocks(grids), num_threads_per_block(blocks), memory_alloc(memory), num_stream(num_stream) {}
    TaskReq(uint32_t num_threads, uint32_t num_reg, uint32_t num_shmem, int64_t memory, int num_stream=1):
        num_threads(num_threads), num_reg(num_reg), num_shmem(num_shmem), memory_alloc(memory), num_stream(num_stream) {}
};

// Task scheduling result returned from scheduler
struct ScheduleResult {
    int device_id;
    int stream_id;

    ScheduleResult() {}
    ScheduleResult(int device_id, int stream_id):
        device_id(device_id), stream_id(stream_id) {}
};

enum TaskStatus {
    TASK_INIT,
    TASK_READY,
    TASK_RUNNING,
    TASK_UPDATE,
    TASK_FINISHED,
    TASK_SCHEDULED
};

struct Task {
    int task_id;
    int start_time;
    int end_time;
    pthread_mutex_t cond_lock;          // lock for condition variable
    pthread_cond_t cond;                // conditional variable to synchronize task scheduling
    sem_t sem;                          // semaphore to synchronize task execution
    TaskStatus status;
    TaskReq req;
    cudaStream_t original_stream;
    ScheduleResult result;

    Task(TaskStatus status, pthread_mutexattr_t *mutex_attr=nullptr, 
            pthread_condattr_t *cond_attr=nullptr): status(status) {
        pthread_mutex_init(&cond_lock, mutex_attr);
        pthread_cond_init(&cond, cond_attr);
        sem_init(&sem, 1, 0);
    }
    Task(uint32_t num_thread, uint32_t num_reg, uint32_t num_shmem, int64_t memory, TaskStatus status, 
            int num_stream=1, cudaStream_t stream=nullptr, pthread_mutexattr_t *mutex_attr=nullptr, 
            pthread_condattr_t *cond_attr=nullptr): 
        req(num_thread, num_reg, num_shmem, memory, num_stream), status(status), original_stream(stream) {
        pthread_mutex_init(&cond_lock, mutex_attr);
        pthread_cond_init(&cond, cond_attr);
        sem_init(&sem, 1, 0);
    }

    Task(const Task& other) {
        task_id = other.task_id;
        start_time = other.start_time;
        end_time = other.end_time;
        sem_init(&sem, 1, 0);
        pthread_mutex_init(&cond_lock, NULL);
        pthread_cond_init(&cond, NULL);
        status = other.status;
        req = other.req;
        original_stream = other.original_stream;
        result = other.result;
    }

    void set_req(uint32_t grids, uint32_t blocks, int64_t memory) {
        req.num_blocks = grids;
        req.num_threads_per_block = blocks;
        req.memory_alloc = memory;
    }
};

struct MemoryFootprintCompare {
    bool operator()(const Task *lhs, const Task *rhs) const {
        return lhs->req.memory_alloc < rhs->req.memory_alloc;
    }
};

struct comm_t {
    int num_gpus;
    int num_running_task;
    int max_task_id;
    uint32_t tail_p;
    uint32_t head_p;
    pthread_mutex_t lock;
    pthread_cond_t cond;
    pthread_condattr_t cond_attr;
    pthread_mutexattr_t mutex_attr;
};

enum cudaOperandKind {
    cudaMallocOp,
    // cudaMallocHostOp,
    cudaMemcpyOp,
    cudaMemsetOp,
    cudaFreeOp
};

struct cudaOperands {
    cudaOperandKind opKind;

    cudaOperands(cudaOperandKind kind): opKind(kind) {}
    virtual ~cudaOperands() {}
    virtual void update_fake_ptr(void *ptr, void *fake_addr) = 0;
};

struct cudaMallocOperands: public cudaOperands {
    void *dev_ptr;
    size_t size;

    cudaMallocOperands(): cudaOperands(cudaMallocOp) {}
    cudaMallocOperands(void *dev_ptr, size_t size): cudaOperands(cudaMallocOp), dev_ptr(dev_ptr), size(size) {}
    ~cudaMallocOperands() {}
    virtual void update_fake_ptr(void *ptr, void *fake_addr) {
        if (dev_ptr == fake_addr) {
            dev_ptr = ptr;
        }
    }
};

struct cudaMemcpyOperands: public cudaOperands {
    void *dst;
    void *src;
    size_t size;
    enum cudaMemcpyKind kind;

    cudaMemcpyOperands(): cudaOperands(cudaMemcpyOp) {}
    cudaMemcpyOperands(void *dst, void *src, size_t size, enum cudaMemcpyKind kind): 
        cudaOperands(cudaMemcpyOp), dst(dst), src(src), size(size), kind(kind) {}
    ~cudaMemcpyOperands() {}
    virtual void update_fake_ptr(void *ptr, void *fake_addr) {
        if ((kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToDevice) && dst == fake_addr) {
            dst = ptr;
        }
        if ((kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice) && src == fake_addr) {
            src = ptr;
        }
    }
};

struct cudaMemsetOperands: public cudaOperands {
    void *dst;
    int value;
    size_t size;

    cudaMemsetOperands(): cudaOperands(cudaMemsetOp) {}
    cudaMemsetOperands(void *dst, int value, size_t size): 
        cudaOperands(cudaMemsetOp), dst(dst), value(value), size(size) {}
    ~cudaMemsetOperands() {}
    virtual void update_fake_ptr(void *ptr, void *fake_addr) {
        if (dst == fake_addr) {
            dst = ptr;
        }
    }
};

struct cudaFreeOperands: public cudaOperands {
    void *dev_ptr;

    cudaFreeOperands(): cudaOperands(cudaFreeOp) {}
    cudaFreeOperands(void *dev_ptr): cudaOperands(cudaFreeOp), dev_ptr(dev_ptr) {}
    ~cudaFreeOperands() {}
    virtual void update_fake_ptr(void *ptr, void *fake_addr) {
        if (dev_ptr == fake_addr) {
            dev_ptr = ptr;
        }
    }
};

#endif // COMMON_H
