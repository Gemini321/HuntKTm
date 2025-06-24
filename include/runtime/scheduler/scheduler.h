#ifndef SCHEDULER_H
#define SCHEDULER_H

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <unordered_map>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "runtime/common/common.h"

#ifdef RUNTIME_SHCEDULER_DEBUG
static char timeBuf[80];
#define SCHEDULER_LOG(format, ...)                                              \
    do {                                                                        \
    time_t now = time(0);                                                       \
    strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%d %H:%M:%S", localtime(&now));   \
    fprintf(stderr, "[Scheduler] " format "\n", ##__VA_ARGS__);                 \
    } while (0)
#define ALIVE_MSG()                                                             \
    do {                                                                        \
    fprintf(stderr, "[Scheduler] Alive message\n");                             \
    } while (0)
#else
#define SCHEDULER_LOG(format, ...) do {} while (0)
#endif

#define MUTEX_WAITTIME 100

struct DeviceStatus {
    int num_free_streams;
    int num_running_tasks;
    int num_pending_tasks;
    int num_finished_tasks;
    int warp_size;
    std::vector<int> running_tasks;

    // resources
    uint32_t num_SM;
    int64_t total_memory;
    int64_t free_memory;
    uint32_t total_reg_per_SM;
    uint32_t used_reg;
    uint32_t total_shmem_per_SM;
    uint32_t used_shmem;
    uint32_t total_thread_per_SM;
    uint32_t used_thread;
};

class Scheduler {
public:
    Scheduler();
    ~Scheduler();

    void initialize();
    void finalize();
    void print_comm();
    void print_status();
    bool has_task_availale();
    void schedule_task();

private:
    comm_t *comm;
    size_t comm_size;
    int opened_file;

    Task *running_tasks;
    std::vector<DeviceStatus> device_status;

    cudaStream_t *stream_pool;  // CUDA streams from different devices (deprecated)
    // map original stream from one device to stream id
    std::unordered_map<cudaStream_t, int> stream_map[MAX_NUM_GPUS];
    std::unordered_map<int, int> device_map;            // map task_id to device_id
    int num_gpus;
    int cur_device_id;

    void init_shm();
    void init_scheduler_status();
    void init_device_status();
    int schedule_device(const TaskReq &req, int task_id);
    // int schedule_stream(cudaStream_t stream, int device_id);
}; // class Scheduler

#endif // SCHEDULER_H
