#include "runtime/wrapper/wrapper.h"
#include "runtime/common/common.h"
#include "runtime/wrapper/runtime_interface.h"
#include <bits/stdint-uintn.h>
#include <chrono>
#include <cstdlib>
#include <string>

Runtime *R = nullptr;

extern "C" {

cudaError_t cudaMallocWrapper(void **dev_ptr, size_t size) {
    if (R->is_lazy()) {
        // @TODO: handle memory excess
        void *fake_addr = R->allocate_fake_addr(size);
        R->register_fake_addr(fake_addr, size);
        *dev_ptr = fake_addr;
        R->register_cuda_operation(new cudaMallocOperands(fake_addr, size));
        WRAPPER_LOG("[Lazy] cudaMallocWrapper: %p, %zu, fake_addr: %p", fake_addr, size, *dev_ptr);
        return cudaSuccess;
    }

    cudaError_t err = cudaMalloc(dev_ptr, size);
    const char *error_string = cudaGetErrorString(err);
    WRAPPER_LOG("cudaMallocWrapper: %p, %zu, %s", *dev_ptr, size, error_string);
    return err;
}

cudaError_t cudaMallocAsyncWrapper(void **dev_ptr, size_t size, cudaStream_t original_stream) {
    if (R->is_lazy()) {
        assert(false && "cudaMallocAsyncWrapper is not implemented for lazy runtime yet.");
    }

    cudaError_t err = cudaMallocAsync(dev_ptr, size, original_stream);
    const char *error_string = cudaGetErrorString(err);
    WRAPPER_LOG("cudaMallocAsyncWrapper: %p, %zu, %p, %s", *dev_ptr, size, original_stream, error_string);
    return err;
}

cudaError_t cudaStreamCreateWrapper(cudaStream_t *stream_ptr) {
    if (R->is_lazy()) {
        assert(false && "cudaStreamCreateWrapper is not implemented for lazy runtime yet.");
    }

    cudaError_t err = cudaStreamCreate(stream_ptr);
    const char *error_string = cudaGetErrorString(err);
    WRAPPER_LOG("cudaStreamCreateWrapper: %p, %p, %s", *stream_ptr, stream_ptr, error_string);
    return err;
}

cudaError_t cudaMemsetWrapper(void *dst, int value, size_t count) {
    if (R->is_lazy() && R->is_fake_addr(dst)) {
        R->register_cuda_operation(new cudaMemsetOperands(dst, value, count));
        WRAPPER_LOG("[Lazy] cudaMemsetWrapper: %p, %d, %zu", dst, value, count);
        return cudaSuccess;
    }

    cudaError_t err = cudaMemset(dst, value, count);
    const char *error_string = cudaGetErrorString(err);
    WRAPPER_LOG("cudaMemsetWrapper: %p, %d, %zu, %s", dst, value, count, error_string);
    return err;
}

cudaError_t cudaMemcpyWrapper(void *dst, void *src, size_t count, enum cudaMemcpyKind kind) {
    if (R->is_lazy()) {
        // if GPU memory has already been allocated, eagerly execute
        // @BUG: if 'dst' is registered to lazily execute before, the execution order can not be guaranteed
        if ((kind == cudaMemcpyHostToDevice && !R->is_fake_addr(dst)) || 
            (kind == cudaMemcpyDeviceToHost && !R->is_fake_addr(src)) ||
            (kind == cudaMemcpyDeviceToDevice && (!R->is_fake_addr(dst) && !R->is_fake_addr(src)))) {
            cudaError_t err = cudaMemcpy(dst, src, count, kind);
            const char *error_string = cudaGetErrorString(err);
            WRAPPER_LOG("cudaMemcpyWrapper: %p, %p, %zu, %d, %s", dst, src, count, kind, error_string);
            return err;
        }
        R->register_cuda_operation(new cudaMemcpyOperands(dst, src, count, kind));
        WRAPPER_LOG("[Lazy] cudaMemcpyWrapper: %p, %p, %zu, %d", dst, src, count, kind);
        return cudaSuccess;
    }

    cudaError_t err = cudaMemcpy(dst, src, count, kind);
    const char *error_string = cudaGetErrorString(err);
    WRAPPER_LOG("cudaMemcpyWrapper: %p, %p, %zu, %d, %s", dst, src, count, kind, error_string);
    return err;
}

cudaError_t cudaMemcpyAsyncWrapper(void *dst, void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t original_stream) {
    if (R->is_lazy()) {
        assert(false && "cudaMemcpyAsyncWrapper is not implemented for lazy runtime yet.");
    }

    cudaError_t err = cudaMemcpyAsync(dst, src, count, kind, original_stream);
    const char *error_string = cudaGetErrorString(err);
    WRAPPER_LOG("cudaMemcpyAsyncWrapper: %p, %p, %zu, %d, %p, %s", dst, src, count, kind, original_stream, error_string);
    return err;
}

cudaError_t cudaFreeWrapper(void *dev_ptr) {
    if (R->is_lazy() || (R->was_lazy() && R->is_fake_addr(dev_ptr))) {
        R->register_cuda_operation(new cudaFreeOperands(dev_ptr));
        WRAPPER_LOG("[Lazy] cudaFreeWrapper: %p", dev_ptr);
        return cudaSuccess;
    }

    cudaError_t err = cudaFree(dev_ptr);
    const char *error_string = cudaGetErrorString(err);
    WRAPPER_LOG("cudaFreeWrapper: %p, %s", dev_ptr, error_string);
    return err;
}

cudaError_t cudaFreeAsyncWrapper(void *dev_ptr, cudaStream_t original_stream) {
    if (R->is_lazy() || (R->was_lazy() && R->is_fake_addr(dev_ptr))) {
        assert(false && "cudaFreeAsyncWrapper is not implemented for lazy runtime yet.");
    }

    cudaError_t err = cudaFreeAsync(dev_ptr, original_stream);
    const char *error_string = cudaGetErrorString(err);
    WRAPPER_LOG("cudaFreeAsyncWrapper: %p, %s", dev_ptr, error_string);
    return err;
}

// Predict the peak memory usage based on the memory graph
int64_t cudaPredictPeakMemory(int64_t **memory_graph, int64_t *num_node_per_stream, int64_t num_stream) {
    int64_t peak_memory = 0;
    for (uint32_t i = 0; i < num_stream; i++) {
        int64_t accumulated_stream_memory = 0;
        int64_t max_stream_memory = 0;
        for (uint32_t j = 0; j < num_node_per_stream[i]; j++) {
            accumulated_stream_memory += memory_graph[i][j];
            max_stream_memory = std::max(max_stream_memory, accumulated_stream_memory);
        }
        peak_memory += max_stream_memory;
    }
    WRAPPER_LOG("Predicted peak memory usage: %lu bytes", peak_memory);
    return peak_memory;
}

// Schedule the whole task to a specific device based on number of threads and memory usage
cudaError_t cudaTaskSchedule(uint32_t num_threads, uint32_t num_reg, uint32_t num_shmem, uint64_t total_mem, uint32_t num_stream, bool use_memory_pool) {
    // ignore grid_dim and block_dim currently
    if (num_stream <= 0) {
        num_stream = 1;
    }
    auto begin_time = std::chrono::high_resolution_clock::now();
    auto begin_during = std::chrono::duration_cast<std::chrono::milliseconds>(begin_time - R->start_time);
    fprintf(stderr, "Time when task scheduling starts: %ld ms\n", begin_during.count());
    WRAPPER_LOG("cudaTaskSchedule: (%u, %u, %u, %zu, %u)", num_threads, num_reg, num_shmem, total_mem, num_stream);
    Task new_task(num_threads, num_reg, num_shmem, total_mem, TASK_READY, num_stream);
    WRAPPER_LOG("begin task scheduling");
    R->schedule_task(new_task);
    R->set_req(num_threads, num_reg, num_shmem, total_mem, num_stream);
    R->update_unreleased_memory(total_mem);
    WRAPPER_LOG("Task %d is scheduled to device %d", new_task.task_id, R->device_id);

    // Allocate memory pool
    R->set_use_memory_pool(use_memory_pool);
    if (use_memory_pool) {
        cudaMemPool_t memPool;
        cudaMemPoolProps poolProps = {};
        cuuint64_t poolSize = total_mem;
        poolProps.allocType = cudaMemAllocationTypePinned;
        poolProps.location.id = R->device_id;
        poolProps.location.type = cudaMemLocationTypeDevice;

        // CUDA_SAFE_CALL(cudaMemPoolCreate(&memPool, &poolProps));
        CUDA_SAFE_CALL(cudaDeviceGetDefaultMemPool(&memPool, R->device_id));
        CUDA_SAFE_CALL(cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &poolSize));

        // Warmup memory pool
        // cudaStream_t stream;
        // cudaStreamCreate(&stream);
        // void *ptr;
        // cudaMallocAsync(&ptr, poolSize, stream);
        // cudaFreeAsync(ptr, stream);
        // cudaStreamSynchronize(stream);

        WRAPPER_LOG("Allocated memory pool with size %lu", poolSize);
    }
    else {
        WRAPPER_LOG("Memory pool is not used");
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto end_during = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time);
    fprintf(stderr, "Time when task scheduling ends: %ld ms\n", end_during.count());

    return cudaSuccess;
}

// Schedule the whole task to a specific device based on number of threads and memory usage
cudaError_t cudaTaskScheduleLazy() {
    uint32_t num_threads = 0, num_reg = 0, num_shmem = 0;
    int64_t delta_mem = R->get_delta_allocated_memory();
    WRAPPER_LOG("delta_mem is %ld in cudaTaskScheduleLazy", delta_mem);
    // ignore grid_dim and block_dim currently
    WRAPPER_LOG("cudaTaskScheduleLazy running");
    Task new_task(num_threads, num_reg, num_shmem, delta_mem, TASK_READY);
    R->schedule_task(new_task);
    R->set_req(num_threads, num_reg, num_shmem, delta_mem);
    R->update_unreleased_memory(delta_mem);
    WRAPPER_LOG("Task %d is scheduled to device %d", new_task.task_id, new_task.result.device_id);

    WRAPPER_LOG("[Lazy] cudaTaskScheduleLazy: lazy mode");
    R->execute_lazy_operations();
    R->set_lazy(false);
    return cudaSuccess;
}

// Map original_stream to a new stream
cudaError_t cudaLaunchKernelWrapper(void *func, dim3 grid_dim, dim3 block_dim, void **args, size_t shared_mem=0, cudaStream_t original_stream=0) {
    // cudaStream_t stream = R->get_stream(original_stream);
    cudaStream_t stream = original_stream;
    WRAPPER_LOG("ready to launch kernel");
    cudaError_t err = cudaLaunchKernel(func, grid_dim, block_dim, args, shared_mem, stream);
    const char *error_string = cudaGetErrorString(err);
    WRAPPER_LOG("cudaLaunchKernelWrapper: %p, (%d, %d, %d), (%d, %d, %d), %p, %zu, %p, %s", \
        func, grid_dim.x, grid_dim.y, grid_dim.z, \
        block_dim.x, block_dim.y, block_dim.z, args, shared_mem, stream, error_string);
    return err;
}

// Ensure that the task is scheduled before launching a kernel
void cudaLaunchKernelPrepare(uint64_t grid_dim_xy, int grid_dim_z, uint64_t block_dim_xy, int block_dim_z) {
    int gx = (grid_dim_xy & 0x00000000FFFFFFFF);
    int gy = (grid_dim_xy & 0xFFFFFFFF00000000) >> 32, gz = grid_dim_z;
    int bx = (block_dim_xy & 0x00000000FFFFFFFF);
    int by = (block_dim_xy & 0xFFFFFFFF00000000) >> 32, bz = block_dim_z;
    WRAPPER_LOG("cudaLaunchKernelPrepare: %s", R->is_lazy() ? "lazy" : "eager");
    WRAPPER_LOG("grid_dim: (%d, %d, %d), block_dim: (%d, %d, %d)", gx, gy, gz, bx, by, bz);
    
    // If cudaTaskScheduleLazy is not instrumented statically, call it before each kernel launching
    if (R->is_lazy()) {
        cudaTaskScheduleLazy();
        WRAPPER_LOG("[Lazy] lazily scheduling task in cudaLaunchKernelWrapper");
        R->set_lazy(true);
    }
}

cudaError_t cudaEventCreateWrapper(cudaEvent_t *event_ptr) {
    if (R->is_lazy()) {
        assert(false && "cudaStreamCreateWrapper is not implemented for lazy runtime yet.");
    }

    cudaError_t err = cudaEventCreate(event_ptr);
    const char *error_string = cudaGetErrorString(err);
    WRAPPER_LOG("cudaEventCreateWrapper: %p, %p, %s", *event_ptr, event_ptr, error_string);
    return err;
}

cudaError_t cudaEventRecordWrapper(cudaEvent_t event, cudaStream_t stream) {
    if (R->is_lazy()) {
        assert(false && "cudaStreamCreateWrapper is not implemented for lazy runtime yet.");
    }

    cudaError_t err = cudaEventRecord(event, stream);
    const char *error_string = cudaGetErrorString(err);
    WRAPPER_LOG("cudaEventRecordWrapper: %p, %p, %s", event, stream, error_string);
    return err;
}

cudaError_t cudaStreamWaitEventWrapper(cudaStream_t stream, cudaEvent_t event, unsigned int flags = 0) {
    if (R->is_lazy()) {
        assert(false && "cudaStreamCreateWrapper is not implemented for lazy runtime yet.");
    }

    cudaError_t err = cudaStreamWaitEvent(stream, event, flags);
    const char *error_string = cudaGetErrorString(err);
    WRAPPER_LOG("cudaStreamWaitEventWrapper: %p, %p, %d, %s", stream, event, flags, error_string);
    return err;
}

// cudaError_t cudaAllocMemPoolWrapper(cudaStream_t stream, size_t size) {
//     if (R->is_lazy()) {
//         assert(false && "cudaStreamCreateWrapper is not implemented for lazy runtime yet.");
//     }

//     cudaMemPool_t memPool;
//     cudaMemPoolProps poolProps = {};
//     poolProps.allocType = cudaMemAllocationTypePinned;
//     poolProps.location.id = R->device_id;
//     poolProps.location.type = cudaMemLocationTypeDevice;

//     cudaMemPoolCreate(&memPool, &poolProps);
//     cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &size);
    
//     return cudaSuccess;
// }

// Lookup the real address of a fake address
void *fakeAddrLookup(void *addr) {
    void *real_addr = R->get_real_addr(addr);
    // WRAPPER_LOG("fakeAddrLookup: %p -> %p", addr, real_addr);
    return real_addr;
}

// Initialize runtime status
void cudaInitialize(bool is_lazy_runtime) {
    R = new Runtime();
    R->set_lazy(is_lazy_runtime);
    assert(is_lazy_runtime == false && "Lazy runtime is not supported yet.");
    WRAPPER_LOG("cudaInitialize: using %s runtime", is_lazy_runtime ? "lazy" : "eager");
}

// Release task resource
void cudaFinalize() {
    WRAPPER_LOG("cudaFinalize begin");
    if (R->is_lazy()) {
        R->execute_lazy_operations();
        if (R->cummulate_memory_usage() != 0) {
            R->dump_fake_addr_map();
        }
        assert(R->cummulate_memory_usage() == 0 && "Some memory has not been released when finalizing.");
    }
    cudaError_t error_string = cudaGetLastError();
    WRAPPER_LOG("cudaFinalize last error: %s", cudaGetErrorString(error_string));
    Task finish_task(TASK_FINISHED);
    finish_task.req = R->get_req();
    finish_task.req.memory_alloc = R->get_unreleased_memory();
    R->schedule_task(finish_task);
    delete R;
    WRAPPER_LOG("cudaFinalize end");
}

} // extern "C"
