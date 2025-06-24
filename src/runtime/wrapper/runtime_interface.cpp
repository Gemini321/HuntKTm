#include "runtime/wrapper/runtime_interface.h"
#include "runtime/common/common.h"
#include <bits/stdint-uintn.h>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <semaphore.h>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>

Runtime::Runtime() {
    comm = nullptr;
    comm_size = 0;
    opened_file = -1;
    device_id = -1;
    start_time = std::chrono::high_resolution_clock::now();
    initialize();
}

Runtime::~Runtime() {
    RUNTIME_LOG("Begin to finalize runtime");
    finalize();
    RUNTIME_LOG("Finish finalizing runtime");
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    // RUNTIME_LOG("Total execution time: %ld ms", duration.count());
    fprintf(stderr, "Total execution time: %ld ms\n", duration.count());
}

/* 
 * Initialize the shared memory on program runtime
 * memory layout:
 * ----------------------------------------------
 * | comm_t | running_tasks[MAX_TASK_RUNNING] |
 * ----------------------------------------------
 */
void Runtime::initialize() {
    int fd = shm_open(COMM_FILE, O_RDWR, S_IRUSR | S_IWUSR);
    if (fd == -1) {
        RUNTIME_LOG("%s open error: %s\n", COMM_FILE, strerror(errno));
        assert(false && "scheduler_comm file open error");
    }

    // Get the size of the file
    size_t size = sizeof(comm_t) + sizeof(Task) * MAX_TASK_RUNNING
        + sizeof(cudaStream_t) * MAX_NUM_GPUS * MAX_STREAMS_PER_GPU;

    int trunk = ftruncate(fd, size);
    if (trunk == -1) {
        RUNTIME_LOG("ftruncate error: %s\n", strerror(errno));
        assert(false && "ftruncate error");
    }

    // Map the file into memory
    void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        RUNTIME_LOG("mmap error: %s\n", strerror(errno));
        assert(false && "mmap error");
    }

    // Use the mapped memory
    comm = (comm_t *)(addr);
    comm_size = sizeof(comm_t);
    opened_file = fd;
    running_tasks = (Task *)((char *)addr + sizeof(comm_t));
    // stream_pool = (cudaStream_t *)((char *)addr + sizeof(comm_t) + sizeof(Task) * MAX_TASK_RUNNING);

    // Get program ID
    pthread_mutex_lock(&(comm->lock));
    task_id = comm->max_task_id;
    comm->max_task_id += 1;
    pthread_mutex_unlock(&(comm->lock));

    // Initialize lazy runtime
    cur_fake_addr = 0xffff800000000000;
    last_allocated_memory = 0;
    delta_allocated_memory = 0;
    unreleased_memory = 0;

    print_comm();
}

void Runtime::finalize() {
    // Unmap the memory
    if (munmap(static_cast<void*>(comm), comm_size) == -1) {
        RUNTIME_LOG("munmap error: %s\n", strerror(errno));
        assert(false && "munmap error");
    }

    // Close the file
    if (close(opened_file) == -1) {
        RUNTIME_LOG("file close error: %s\n", strerror(errno));
        assert(false && "file close error");
    }

    for (auto it : addr_size_map) {
        RUNTIME_LOG("Unreleased real address %p with size %lu", it.first, it.second);
    }
    addr_size_map.clear();
    fake_addr_size_map.clear();
}

void Runtime::print_comm() {
    fprintf(stderr, "comm_t: num_running_task=%d, max_task_id=%d\n", comm->num_running_task, comm->max_task_id);
    fprintf(stderr, "total tasks: [");
    for (int i = comm->head_p; i != comm->tail_p; i = (i + 1) % MAX_TASK_RUNNING) {
        fprintf(stderr, "%d ", running_tasks[i].task_id);
    }
    fprintf(stderr, "]\n");
}

void Runtime::set_req(uint32_t num_blocks, uint32_t num_threads_per_block, int64_t memory, int num_stream) {
    req.memory_alloc = memory;
    req.num_blocks = num_blocks;
    req.num_threads_per_block = num_threads_per_block;
    req.num_stream = num_stream;
}

void Runtime::set_req(uint32_t num_threads, uint32_t num_reg, uint32_t num_shmem, int64_t memory, int num_stream) {
    req.memory_alloc = memory;
    req.num_threads = num_threads;
    req.num_reg = num_reg;
    req.num_shmem = num_shmem;
    req.num_stream = num_stream;
}

// Set whether use memory pool or not
void Runtime::set_use_memory_pool(bool use_pool) {
    use_memory_pool = use_pool;
}

// Set whether use lazy runtime or not
void Runtime::set_lazy(bool lazy) {
    is_lazy_runtime = lazy;
    if (lazy) {
        was_lazy_runtime = true;
    }
}

// Get task resource requirement
TaskReq Runtime::get_req() {
    return req;
}

// Return whether use memory pool or not
bool Runtime::is_use_memory_pool() {
    return use_memory_pool;
}

// Return whether use lazy runtime or not
bool Runtime::is_lazy() {
    return is_lazy_runtime;
}

// Return whether the runtime was lazy
bool Runtime::was_lazy() {
    return was_lazy_runtime;
}

void *Runtime::allocate_fake_addr(uint64_t size) {
    void *addr = (void *)cur_fake_addr;
    cur_fake_addr += size;
    return addr;
}

bool Runtime::is_fake_addr_in_map(void *addr) {
    return fake_addr_real_addr_map.find(addr) != fake_addr_real_addr_map.end();
}

bool Runtime::is_fake_addr_in_region(void *addr) {
    auto it = fake_addr_real_addr_map.upper_bound(addr);
    if (it == fake_addr_real_addr_map.end()) {
        return false;
    }
    return (it->first <= addr && (uint64_t)addr < (uint64_t)it->first + fake_addr_size_map[it->first]);
}

// @TODO: use `always_lazy` to accelerate address lookup
void *Runtime::get_real_addr(void *addr) {
    if (is_fake_addr(addr)) {
        bool in_map = is_fake_addr_in_map(addr);
        bool in_region = is_fake_addr_in_region(addr);
        // Fake address is not allocated or out of region
        if (!in_map && !in_region) {
            return addr;
        }
        if (in_map) {
            // RUNTIME_LOG("Replace fake addr %p with real addr %p", addr, fake_addr_real_addr_map[addr]);
            return fake_addr_real_addr_map[addr];
        }
        else {
            auto it = fake_addr_size_map.upper_bound(addr);
            void *base = fake_addr_real_addr_map[it->first];
            uint64_t offset = (uint64_t)addr - (uint64_t)it->first;
            // RUNTIME_LOG("Replace fake addr %p with real addr in region %p", addr, (void *)((uint64_t)base + offset));
            return (void *)((uint64_t)base + offset);
        }
    }
    return addr;
}

// Register the real address (allocated memory) to eager runtime
void Runtime::register_real_addr(void *real_addr, uint64_t size) {
    assert(addr_size_map.find(real_addr) == addr_size_map.end());
    addr_size_map[real_addr] = size;
    RUNTIME_LOG("Register real addr %p with size %lu", real_addr, size);
}

// Register the fake address to lazy runtime
void Runtime::register_fake_addr(void *fake_addr, uint64_t size) {
    assert(fake_addr_size_map.find(fake_addr) == fake_addr_size_map.end());
    fake_addr_size_map[fake_addr] = size;
    fake_addr_op_map[fake_addr] = std::vector<cudaOperands *>();
}

// Register the CUDA operation to the runtime
void Runtime::register_cuda_operation(cudaOperands *op) {
    lazy_cuda_operations.push_back(op);
    if (op->opKind == cudaMallocOp) {
        cudaMallocOperands *malloc_op = (cudaMallocOperands *)op;
        assert(is_fake_addr((void *)(malloc_op->dev_ptr)));
        RUNTIME_LOG("Register fake addr %p to cudaMalloc operation", malloc_op->dev_ptr);
        lazy_cudaMalloc_operations.push_back(malloc_op);
        fake_addr_op_map[malloc_op->dev_ptr].push_back(op);
    }
    else {
        // Register the operation to the fake address.
        // Some operands may be updated by `fakeAddrLoopup`.
        switch (op->opKind) {
            case cudaMallocOp: {
                break;
            }
            case cudaMemcpyOp: {
                cudaMemcpyOperands *memcpy_op = (cudaMemcpyOperands *)op;
                if (memcpy_op->kind == cudaMemcpyHostToDevice || memcpy_op->kind == cudaMemcpyDeviceToDevice) {
                    // assert(is_fake_addr(memcpy_op->dst));
                    // fake_addr_op_map[memcpy_op->dst].push_back(op);
                    if (is_fake_addr(memcpy_op->dst)) {
                        fake_addr_op_map[memcpy_op->dst].push_back(op);
                    }
                }
                if (memcpy_op->kind == cudaMemcpyDeviceToHost || memcpy_op->kind == cudaMemcpyDeviceToDevice) {
                    // assert(is_fake_addr(memcpy_op->src));
                    // fake_addr_op_map[memcpy_op->src].push_back(op);
                    if (is_fake_addr(memcpy_op->dst)) {
                        fake_addr_op_map[memcpy_op->dst].push_back(op);
                    }
                }
                break;
            }
            case cudaMemsetOp: {
                cudaMemsetOperands *memset_op = (cudaMemsetOperands *)op;
                // assert(is_fake_addr(memset_op->dst));
                // fake_addr_op_map[memset_op->dst].push_back(op);
                if (is_fake_addr(memset_op->dst)) {
                    fake_addr_op_map[memset_op->dst].push_back(op);
                }
                RUNTIME_LOG("Register fake addr %p to memset operation", memset_op->dst);
                break;
            }
            case cudaFreeOp: {
                cudaFreeOperands *free_op = (cudaFreeOperands *)op;
                // assert(is_fake_addr(free_op->dev_ptr));
                // fake_addr_op_map[free_op->dev_ptr].push_back(op);
                if (is_fake_addr(free_op->dev_ptr)) {
                    fake_addr_op_map[free_op->dev_ptr].push_back(op);
                }
                break;
            }
            default: {
                RUNTIME_LOG("Unknown operation kind: %d\n", op->opKind);
                assert(false && "Unknown operation kind");
            }
        }
    }
}

void Runtime::execute_lazy_operations() {
    // Allocate GPU memory
    for (auto malloc_op: lazy_cudaMalloc_operations) {
        void *fake_addr = malloc_op->dev_ptr;
        void *real_addr = nullptr;
        uint64_t size = malloc_op->size;
        assert(fake_addr_real_addr_map.find(fake_addr) == fake_addr_real_addr_map.end());
        CUDA_SAFE_CALL(cudaMalloc(&real_addr, size));

        fake_addr_real_addr_map[fake_addr] = real_addr;
        real_addr_fake_addr_map[real_addr] = fake_addr;
        RUNTIME_LOG("Allocate fake addr %p to real addr %p", fake_addr, real_addr);
    }

    // Execute CUDA operations
    for (auto op: lazy_cuda_operations) {
        switch (op->opKind) {
            case cudaMallocOp: {
                break;
            }
            case cudaMemcpyOp: {
                cudaMemcpyOperands *memcpy_op = (cudaMemcpyOperands *)op;
                void *dst = get_real_addr(memcpy_op->dst);
                void *src = get_real_addr(memcpy_op->src);
                RUNTIME_LOG("Execute memcpy operation from %p to %p", src, dst);
                CUDA_SAFE_CALL(cudaMemcpy(dst, src, memcpy_op->size, memcpy_op->kind));
                break;
            }
            case cudaMemsetOp: {
                cudaMemsetOperands *memset_op = (cudaMemsetOperands *)op;
                void *dst = get_real_addr(memset_op->dst);
                RUNTIME_LOG("Execute memset operation on real addr %p", dst);
                CUDA_SAFE_CALL(cudaMemset(dst, memset_op->value, memset_op->size));
                break;
            }
            case cudaFreeOp: {
                cudaFreeOperands *free_op = (cudaFreeOperands *)op;
                void *real_addr = free_op->dev_ptr;
                assert(real_addr_fake_addr_map.find(real_addr) != real_addr_fake_addr_map.end() && "real address must be allocated");
                if (real_addr_fake_addr_map.find(real_addr) != real_addr_fake_addr_map.end()) {
                    void *fake_addr = real_addr_fake_addr_map[real_addr];
                    fake_addr_real_addr_map.erase(fake_addr);
                    assert(fake_addr_size_map.find(fake_addr) != fake_addr_size_map.end());
                    fake_addr_size_map.erase(fake_addr);
                    real_addr_fake_addr_map.erase(real_addr);
                }
                RUNTIME_LOG("Execute free operation on real addr %p (fake addr %p)", real_addr, free_op->dev_ptr);
                CUDA_SAFE_CALL(cudaFree(real_addr));
                break;
            }
            default: {
                RUNTIME_LOG("Unknown operation kind: %d\n", op->opKind);
                assert(false && "Unknown operation kind");
            }
        }
    }
    for (cudaOperands *cuda_op : lazy_cuda_operations) {
        delete cuda_op;
    }
    lazy_cuda_operations.clear();
    lazy_cudaMalloc_operations.clear();
    delta_allocated_memory = cummulate_memory_usage() - last_allocated_memory;
    last_allocated_memory = cummulate_memory_usage();
    RUNTIME_LOG("delta_allocated_memory is %ld in execute_lazy_operations", delta_allocated_memory);
    RUNTIME_LOG("Finish executing lazy operations");
}

// Deprecated
// cudaStream_t Runtime::get_stream(int stream_id) {
//     assert(stream_id >= 0 && stream_id < MAX_STREAMS_PER_GPU * MAX_NUM_GPUS);
//     return stream_pool[stream_id];
// }

cudaStream_t Runtime::get_stream(cudaStream_t original_stream) {
    // if (stream_map.find(original_stream) == stream_map.end()) {
    //     cudaStream_t new_stream;
    //     CUDA_SAFE_CALL(cudaStreamCreate(&new_stream));
    //     stream_map[original_stream] = new_stream;
    // }
    // return stream_map[original_stream];
    return original_stream;
}

std::string getCurrentTimeWithMilliseconds() {
    // Get current time with milliseconds
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) % 1000;

    // Format the time
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S")
        << '.' << std::setfill('0') << std::setw(3) << now_ms.count();

    return oss.str();
}

// Schedule a task to a specific device_id and set device to device_id
void Runtime::schedule_task(Task &new_task) {
    new_task.task_id = task_id;

    // add the new task to the running task list
    pthread_mutex_lock(&(comm->lock));

    if (comm->num_running_task == MAX_TASK_RUNNING) {
        RUNTIME_LOG("FATAL: the number of running kernel exceeds %d!!!", MAX_TASK_RUNNING);
        assert(false && "the number of running task exceeds the limit");
    }
    auto begin_time = std::chrono::high_resolution_clock::now();

    running_tasks[comm->tail_p] = new_task;
    Task &k = running_tasks[comm->tail_p];
    comm->num_running_task ++;
    comm->tail_p = (comm->tail_p + 1) % MAX_TASK_RUNNING;
    RUNTIME_LOG("signal task %d with status %d, memory %ld to scheduler (head_p=%d, tail_p = %d)", k.task_id, k.status, k.req.memory_alloc, comm->head_p, comm->tail_p);

    pthread_cond_signal(&(comm->cond));
    pthread_mutex_unlock(&(comm->lock));

    // waiting for schedule result when task is ready to run
    if (k.status == TASK_READY) {
        RUNTIME_LOG("task %d require %ld memory", k.task_id, k.req.memory_alloc);
        RUNTIME_LOG("waiting for cond of task %d", k.task_id);
        sem_wait(&(k.sem));
        device_id = k.result.device_id;
        new_task.result = k.result;

        auto set_device_begin_time = std::chrono::high_resolution_clock::now();
        CUDA_SAFE_CALL(cudaSetDevice(device_id));
        RUNTIME_LOG("scheduled task %d to device %d", k.task_id, device_id);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto end_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto set_device_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - set_device_begin_time);
        auto task_schedule_duration = std::chrono::duration_cast<std::chrono::milliseconds>(set_device_begin_time - begin_time);
        std::string current_time = getCurrentTimeWithMilliseconds();
        fprintf(stderr, "Time when task %d scheduling ends: %ld ms (%s) (task scheduling: %ld ms, set device: %ld ms)\n", 
            k.task_id, end_duration.count(), current_time.c_str(), task_schedule_duration.count(), set_device_duration.count());
    }
    // return directly if task finished
    else if (k.status == TASK_FINISHED) {
        RUNTIME_LOG("signal task %d with status %d, memory %ld to scheduler (head_p=%d, tail_p = %d)", k.task_id, k.status, k.req.memory_alloc, comm->head_p, comm->tail_p);
        RUNTIME_LOG("finished running task %d", k.task_id);
    }
    else {
        RUNTIME_LOG("WARNNING: unexpected status %d from kenrel %d", k.status, k.task_id);
    }
}

// Accumulate the memory usage of task
int64_t Runtime::cummulate_memory_usage() {
    uint64_t total_memory = 0;
    for (auto it : fake_addr_size_map) {
        total_memory += it.second;
    }
    return total_memory;
}

// Get the delta memory usage of task
int64_t Runtime::get_delta_allocated_memory() {
    return delta_allocated_memory;
}

// Get the unreleased memory usage of task
int64_t Runtime::get_unreleased_memory() {
    return unreleased_memory;
}

// Update unreleased memory usage of task
void Runtime::update_unreleased_memory(int64_t delta_memory) {
    unreleased_memory += delta_memory;
    RUNTIME_LOG("update_unreleased_memory: delta_memory=%ld, unreleased_memory=%ld", delta_memory, unreleased_memory);
}

void Runtime::dump_fake_addr_map() {
    RUNTIME_LOG("Unreleased fake address:");
    for (auto it : fake_addr_size_map) {
        RUNTIME_LOG("fake addr %p, size %lu", it.first, it.second);
    }
    for (auto it : fake_addr_real_addr_map) {
        RUNTIME_LOG("fake addr %p -> real addr %p", it.first, it.second);
    }
}
