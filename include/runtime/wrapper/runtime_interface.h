#ifndef RUNTIME_INTERFACE_H
#define RUNTIME_INTERFACE_H

#include <bits/stdint-uintn.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <map>
#include <vector>
#include <chrono>
#include "runtime/common/common.h"

#ifdef RUNTIME_DEBUG
#define RUNTIME_LOG(format, ...)                                                              \
    do {                                                                                      \
    fprintf(stderr, "[Runtime] " format "\n", ##__VA_ARGS__);                                 \
    } while (0)
#else
#define RUNTIME_LOG(format, ...) do {} while (0)
#endif

class Runtime {
public:
    Runtime();
    ~Runtime();

    void initialize();
    void finalize();
    void print_comm();
    void set_req(uint32_t num_blocks, uint32_t num_threads_per_block, int64_t memory, int num_stream=1);
    void set_req(uint32_t num_threads, uint32_t num_reg, uint32_t num_shmem, int64_t memory, int num_stream=1);
    void set_lazy(bool is_lazy_runtime);
    void set_use_memory_pool(bool use_memory_pool);
    TaskReq get_req();
    bool is_use_memory_pool();
    bool is_lazy();
    bool was_lazy();
    void *allocate_fake_addr(uint64_t size);
    inline bool is_fake_addr(void *addr) { return ((uint64_t)addr >= 0xffff800000000000); }
    bool is_fake_addr_in_map(void *addr);
    bool is_fake_addr_in_region(void *addr);
    void *get_real_addr(void *addr);
    void register_real_addr(void *real_addr, uint64_t size);
    void register_fake_addr(void *fake_addr, uint64_t size);
    void register_cuda_operation(cudaOperands *op);
    void schedule_task(Task &new_task);
    // cudaStream_t get_stream(int stream_id);
    cudaStream_t get_stream(cudaStream_t original_stream);
    void execute_lazy_operations();
    int64_t cummulate_memory_usage();
    int64_t get_delta_allocated_memory();
    int64_t get_unreleased_memory();
    void update_unreleased_memory(int64_t delta_memory);
    void dump_fake_addr_map();

public:
    comm_t *comm;
    size_t comm_size;
    int opened_file;
    int task_id;
    int device_id;
    bool use_memory_pool;
    bool is_lazy_runtime;
    bool was_lazy_runtime;
    uint64_t cur_fake_addr;
    int64_t last_allocated_memory;
    int64_t delta_allocated_memory;
    int64_t unreleased_memory;
    TaskReq req;
    Task *running_tasks;    // mapped from scheduler
   std::chrono::system_clock::time_point start_time; 

    // cudaStream_t *stream_pool;  // CUDA streams from different devices
    // std::map<cudaStream_t, cudaStream_t> stream_map;

    // Eager runtime
    std::map<void *, uint64_t> addr_size_map;

    // Lazy runtime
    std::map<void *, uint64_t> fake_addr_size_map;
    std::map<void *, std::vector<cudaOperands *>> fake_addr_op_map;
    std::map<void *, void *> fake_addr_real_addr_map;       // fake_addr -> real_addr
    std::map<void *, void *> real_addr_fake_addr_map;       // real_addr -> fake_addr
    std::vector<cudaOperands *> lazy_cuda_operations;
    std::vector<cudaMallocOperands *> lazy_cudaMalloc_operations;
}; // class Runtime

#endif // RUNTIME_INTERFACE_H
