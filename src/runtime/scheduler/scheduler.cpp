#include "runtime/scheduler/scheduler.h"
#include "runtime/common/common.h"
#include <pthread.h>
#include <semaphore.h>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <iostream>
#include "libstatus/profiler.h"

Scheduler S;

char LIBSTATUS_FILENAME_BASE[] = "sched";
Profiler p(LIBSTATUS_FILENAME_BASE);

void dump_stats(void) {
#define STATS_LOG(str)    \
  do {                    \
    stats_file << str;    \
    stats_file.flush();   \
    SCHEDULER_LOG(str);   \
  } while (0)
  std::ofstream stats_file;
  stats_file.open("sched-stats.out");
  stats_file.close();
}

void sigint_handler(int unused) {
    SCHEDULER_LOG("Caught interrupt. Exiting.\n");
    p.stop_sampling();
    dump_stats();
    SCHEDULER_LOG("Finished dumpping states.\n");
    _exit(0);
}

void set_timeout(struct timespec *ts, int timeout_ms) {
    clock_gettime(CLOCK_REALTIME, ts);
    ts->tv_sec += timeout_ms / 1000;
    ts->tv_nsec += (timeout_ms % 1000) * 1000000;
    if (ts->tv_nsec >= 1000000000) {
        ts->tv_sec += 1;
        ts->tv_nsec -= 1000000000;
    }
}

std::string getCurrentTimeWithMilliseconds() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);

    // Get millisecond
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) % 1000;

    // Format the time
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S")
        << '.' << std::setfill('0') << std::setw(3) << now_ms.count();

    return oss.str();
}

Scheduler::Scheduler() {
    initialize();
}

Scheduler::~Scheduler() {
    finalize();
}

/* 
 * Initialize the shared memory on scheduler
 * memory layout:
 * ------------------------------------------------
 * | comm_t | running_tasks[MAX_TASK_RUNNING] |
 * ------------------------------------------------
 */
void Scheduler::init_shm()  {
    int fd = shm_open(COMM_FILE, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    if (fd == -1) {
        SCHEDULER_LOG("%s open error: %s\n", COMM_FILE, strerror(errno));
        assert(false && "scheduler_comm open error");
    }

    // Get the size of the file
    size_t size = sizeof(comm_t) 
        + sizeof(Task) * MAX_TASK_RUNNING
        + sizeof(cudaStream_t) * MAX_NUM_GPUS * MAX_STREAMS_PER_GPU;

    int trunk = ftruncate(fd, size);
    if (trunk == -1) {
        SCHEDULER_LOG("ftruncate error: %s\n", strerror(errno));
        assert(false && "ftruncate error");
    }

    // Map the file into memory
    void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        SCHEDULER_LOG("mmap error: %s\n", strerror(errno));
        assert(false && "mmap error");
    }
    memset(addr, 0, size);

    // Use the mapped memory
    comm = (comm_t *)(addr);
    comm_size = sizeof(comm_t);
    opened_file = fd;
    running_tasks = (Task *)((char *)addr + sizeof(comm_t));
    stream_pool = nullptr;
    // stream_pool = (cudaStream_t *)((char *)addr + sizeof(comm_t) + sizeof(Task) * MAX_TASK_RUNNING);
    comm->head_p = comm->tail_p = 0;
    comm->num_running_task = 0;
    comm->max_task_id = 0;

    // Initialize lock
    pthread_mutexattr_init(&comm->mutex_attr);
    pthread_condattr_init(&comm->cond_attr);
    pthread_mutexattr_setpshared(&comm->mutex_attr, PTHREAD_PROCESS_SHARED);
    pthread_condattr_setpshared(&comm->cond_attr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&comm->lock, &comm->mutex_attr);
    pthread_cond_init(&comm->cond, &comm->cond_attr);

    SCHEDULER_LOG("Initialized shared memory");
}

void Scheduler::init_scheduler_status() {
    for (int i = 0; i < MAX_TASK_RUNNING; i ++) {
        running_tasks[i] = Task(TASK_INIT, &comm->mutex_attr, &comm->cond_attr);
    }

    SCHEDULER_LOG("Initialized scheduler status");

    CUDA_SAFE_CALL(cudaGetDeviceCount(&num_gpus));
    if (num_gpus == 0) {
        SCHEDULER_LOG("No GPUs found\n");
        assert(false && "No GPUs found");
    }
    SCHEDULER_LOG("Found %d GPUs", num_gpus);

    comm->num_gpus = num_gpus;
    cur_device_id = 0;
    device_status.resize(num_gpus);

    SCHEDULER_LOG("Allocated device status");

    // stream_pool = (cudaStream_t *)malloc(sizeof(cudaStream_t) * num_gpus * MAX_STREAMS_PER_GPU);
    // for (int i = 0; i < num_gpus * MAX_STREAMS_PER_GPU; i ++) {
    //     CUDA_SAFE_CALL(cudaSetDevice(i / MAX_STREAMS_PER_GPU));
    //     // CUDA_SAFE_CALL(cudaStreamCreate(&stream_pool[i]));
    // }
}

void Scheduler::init_device_status() {
    for (int i = 0; i < num_gpus; i ++) {
        cudaDeviceProp prop;
        CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, i));
        device_status[i].num_free_streams = MAX_STREAMS_PER_GPU;
        device_status[i].num_running_tasks = 0;
        device_status[i].num_pending_tasks = 0;
        device_status[i].num_finished_tasks = 0;
        // device_status[i].total_memory = prop.totalGlobalMem - RESERVED_MEMORY;
        // device_status[i].free_memory = prop.totalGlobalMem - RESERVED_MEMORY;
        // Control the maximum memory capacity
        device_status[i].total_memory = TOTAL_MEMORY - RESERVED_MEMORY;
        device_status[i].free_memory = TOTAL_MEMORY - RESERVED_MEMORY;
        device_status[i].warp_size = prop.warpSize;
        device_status[i].total_reg_per_SM = prop.regsPerBlock;
        device_status[i].used_reg = 0;
        device_status[i].total_shmem_per_SM = prop.sharedMemPerBlock;
        device_status[i].used_shmem = 0;
        device_status[i].total_thread_per_SM = prop.maxThreadsPerMultiProcessor;
        device_status[i].used_thread = 0;
        device_status[i].num_SM = prop.multiProcessorCount;
        device_status[i].running_tasks.clear();
    }

    SCHEDULER_LOG("Initialized device status");
}

void Scheduler::initialize() {
    init_shm();
    init_scheduler_status();
    init_device_status();

    print_status();
}

void Scheduler::finalize() {
    // Unmap the memory
    if (munmap(static_cast<void*>(comm), comm_size) == -1) {
        SCHEDULER_LOG("munmap error: %s\n", strerror(errno));
        assert(false && "No GPUs found");
    }

    // Close the file
    if (shm_unlink(COMM_FILE) == -1) {
        SCHEDULER_LOG("shm_unlink error: %s\n", strerror(errno));
        assert(false && "shm_unlink error");
    }

    // Destroy lock
    pthread_mutex_destroy(&comm->lock);

    if (running_tasks) {
        free(running_tasks);
    }
}

void Scheduler::print_comm() {
    fprintf(stderr, "comm_t: num_running_task=%d, max_task_id=%d, head_p=%d, tail_p=%d\n", 
        comm->num_running_task, comm->max_task_id, comm->head_p, comm->tail_p);
    fprintf(stderr, "running tasks: [");
    for (int i = comm->head_p; i != comm->tail_p; i = (i + 1) % MAX_TASK_RUNNING) {
        const char *status;
        if (running_tasks[i].status == TASK_READY) {
            status = "READY";
        }
        else if (running_tasks[i].status == TASK_FINISHED) {
            status = "FINISHED";
        }
        else {
            status = "INVALID";
        }
        fprintf(stderr, "%d(%s) ", running_tasks[i].task_id, status);
    }
    fprintf(stderr, "]\n");
}

void Scheduler::print_status() {
    fprintf(stderr, "Scheduler status:\n");
    fprintf(stderr, "num_gpus: %d\n", num_gpus);

    for (int i = 0; i < num_gpus; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        fprintf(stderr, "GPU %d: %s\n", i, prop.name);
        fprintf(stderr, "  - Compute Capability: %d.%d\n", prop.major, prop.minor);
        fprintf(stderr, "  - Total Global Memory: %lu GBs\n", prop.totalGlobalMem / 1024 / 1024 / 1024);
        fprintf(stderr, "  - Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        fprintf(stderr, "  - Max Block Size: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        fprintf(stderr, "  - Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        fprintf(stderr, "  - Number of SMs: %d\n", prop.multiProcessorCount);
        fprintf(stderr, "  - Warp Size: %d\n", prop.warpSize);
        fprintf(stderr, "  - Registers per Block: %d\n", prop.regsPerBlock);
        fprintf(stderr, "  - Shared Memory per Block: %lu Bytes\n", prop.sharedMemPerBlock);
        fprintf(stderr, "\n");
    }
}

bool Scheduler::has_task_availale() {
    if (comm->num_running_task > 0) {
        SCHEDULER_LOG("has task available: %d", comm->num_running_task);
    }
    return comm->num_running_task;
}

// Schedule a task to a specific device
int Scheduler::schedule_device(const TaskReq &req, int task_id) {
    int device_id;
    // All tasks in a task should be scheduled to the same device
    if (device_map.count(task_id) > 0) {
        device_id = device_map[task_id];
        SCHEDULER_LOG("Schedule task %d to device %d again", task_id, device_id);
        assert(false && "Schedule task to the same device again is not allowed in eager mode");

        if ((int64_t)device_status[device_id].free_memory < req.memory_alloc) {
            SCHEDULER_LOG("%zu memory required exceeds %zu free memory on device %d", 
                req.memory_alloc, device_status[device_id].free_memory, device_id);
            return -1;
        }
    }
    else {
        float max_avail_SM = 0;
        device_id = -1;
        for (int i = 0; i < num_gpus; i ++) {
            SCHEDULER_LOG("Device %d free memory: %ld, required memory: %ld", i, device_status[i].free_memory, req.memory_alloc);
            if (device_status[i].free_memory > req.memory_alloc && device_status[i].num_free_streams > req.num_stream) {
                float avail_SM_in_reg = device_status[i].num_SM - 
                    (float)(device_status[i].used_reg + req.num_reg) / device_status[i].total_reg_per_SM;
                float avail_SM_in_shmem = device_status[i].num_SM - 
                    (float)(device_status[i].used_shmem + req.num_shmem) / device_status[i].total_shmem_per_SM;
                float avail_SM_in_thread = device_status[i].num_SM - 
                    (float)(device_status[i].used_thread + req.num_threads) / device_status[i].total_thread_per_SM;
                float avail_SM = std::min(std::min(avail_SM_in_reg, avail_SM_in_shmem), avail_SM_in_thread);
                SCHEDULER_LOG("device %d number of SMs: %d, used reg: %d, used shmem: %d, used thread: %d, total reg: %d, total shmem: %d", 
                    i, device_status[i].num_SM, device_status[i].used_reg, device_status[i].used_shmem, device_status[i].used_thread,
                    device_status[i].total_reg_per_SM, device_status[i].total_shmem_per_SM);
                SCHEDULER_LOG("request reg: %d, request shmem: %d, request thread: %d", req.num_reg, req.num_shmem, req.num_threads);
                SCHEDULER_LOG("available SM: %f, max available SM: %f", avail_SM, max_avail_SM);
                // Select the device that can provide maximal available SM resources for new task
                if (avail_SM >= max_avail_SM && req.num_stream <= device_status[i].num_free_streams) {
                    max_avail_SM = avail_SM;
                    device_id = i;
                    device_map[task_id] = device_id;
                }
            }
        }
    }
    SCHEDULER_LOG("Finish device selection for task %d, device %d", task_id, device_id);

    return device_id;
}

// Schedule a kernel to a specific stream
// int Scheduler::schedule_stream(cudaStream_t stream, int device_id) {
//     if (stream_map[device_id].count(stream) > 0) {
//         return stream_map[device_id][stream];
//     }
//     else {
//         int stream_id = stream_map[device_id].size();
//         assert(stream_id < MAX_STREAMS_PER_GPU && "number of streams in GPU exceeds");
//         stream_map[device_id][stream] = stream_id + device_id * MAX_STREAMS_PER_GPU;
//         return stream_map[device_id][stream];
//     }
// }

// Schedule a task from the head of task queue
// make sure at least one task in queue
void Scheduler::schedule_task() {
    timespec ts;
    std::vector<Task *> pending_tasks;

    while (true) {
        set_timeout(&ts, MUTEX_WAITTIME);
        pthread_mutex_lock(&comm->lock);
        int ret = pthread_cond_timedwait(&comm->cond, &comm->lock, &ts);
        if (ret != 0 && ret != ETIMEDOUT) {
            SCHEDULER_LOG("Error while locking: %s\n", strerror(ret));
            assert(false);
        }
        pthread_mutex_unlock(&comm->lock);
        // ALIVE_MSG();
        while (comm->head_p != comm->tail_p) {
            print_comm();

            uint32_t cur_p = comm->head_p;
            const uint32_t tail_p = comm->tail_p;
            // Attempt to schedule tasks in the circular queue
            while (cur_p != tail_p) {
                Task *k = &running_tasks[cur_p];
                SCHEDULER_LOG("cur_p: %d, tail_p: %d, task status: %d, memory alloc: %ld", cur_p, tail_p, k->status, k->req.memory_alloc);
                if (k->status == TASK_READY) {
                    auto begin_time = std::chrono::high_resolution_clock::now();
                    // Schedule a task as soon as it is ready
                    int device_id = schedule_device(k->req, k->task_id);
                    if (device_id == -1) {
                        pending_tasks.push_back(k);
                    }
                    else {
                        k->result.device_id = device_id;
                        k->status = TASK_SCHEDULED;
                        device_status[device_id].free_memory -= k->req.memory_alloc;
                        device_status[device_id].num_free_streams -= k->req.num_stream;
                        device_status[device_id].used_reg += k->req.num_reg;
                        device_status[device_id].used_shmem += k->req.num_shmem;
                        device_status[device_id].used_thread += k->req.num_threads;
                        device_status[device_id].num_running_tasks += 1;
                        device_status[device_id].running_tasks.push_back(k->task_id);
                        SCHEDULER_LOG("finish scheduling task %d to device %d", k->task_id, device_id);

                        SCHEDULER_LOG("scheduled task %d to device %d, allocated memory %ld, free device memory %zu", 
                            k->task_id, device_id, k->req.memory_alloc, device_status[device_id].free_memory);
                        SCHEDULER_LOG("current running tasks on device %d:", device_id);
                        for (auto task_id : device_status[device_id].running_tasks) {
                            SCHEDULER_LOG("%d, ", task_id);
                        }
                        SCHEDULER_LOG("\n");

                        sem_post(&(k->sem));
                        SCHEDULER_LOG("signal task %d with status %d, memory %ld to scheduler (head_p=%d, tail_p = %d)", 
                            k->task_id, k->status, k->req.memory_alloc, comm->head_p, comm->tail_p);
                        auto end_time = std::chrono::high_resolution_clock::now();
                        auto during = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time);
                        std::string current_time = getCurrentTimeWithMilliseconds();
                        SCHEDULER_LOG("Time when task scheduling ends: %ld ms (%s)", during.count(), current_time.c_str());
                    }
                }
                else if (k->status == TASK_FINISHED) {
                    // Release used resources
                    int device_id = device_map[k->task_id];
                    device_status[device_id].free_memory += k->req.memory_alloc;
                    device_status[device_id].num_free_streams += k->req.num_stream;
                    device_status[device_id].used_reg -= k->req.num_reg;
                    device_status[device_id].used_shmem -= k->req.num_shmem;
                    device_status[device_id].used_thread -= k->req.num_threads;
                    device_status[device_id].num_running_tasks -= 1;
                    device_status[device_id].running_tasks.erase(std::remove(device_status[device_id].running_tasks.begin(), 
                        device_status[device_id].running_tasks.end(), k->task_id), device_status[device_id].running_tasks.end());
                    SCHEDULER_LOG("finished running task %d, freed memory %ld, device %d free memory: %ld", 
                        k->task_id, k->req.memory_alloc, device_id, device_status[device_id].free_memory);
                    device_map.erase(k->task_id);
                }
                else {
                    SCHEDULER_LOG("invalid task status\n");
                    assert(false);
                }
                cur_p = (cur_p + 1) % MAX_TASK_RUNNING;
            }
            std::sort(pending_tasks.begin(), pending_tasks.end(), MemoryFootprintCompare());

            // Schedule pending tasks
            std::vector<Task *> unscheduled_tasks;
            for (auto k : pending_tasks) {
                int device_id = schedule_device(k->req, k->task_id);
                if (device_id == -1) {
                    unscheduled_tasks.push_back(k);
                    SCHEDULER_LOG("pending task %d cannot be scheduled due to memory restriction", k->task_id);
                    continue;
                }
                k->result.device_id = device_id;
                k->status = TASK_SCHEDULED;
                device_status[device_id].free_memory -= k->req.memory_alloc;
                device_status[device_id].num_free_streams -= k->req.num_stream;
                device_status[device_id].used_reg += k->req.num_reg;
                device_status[device_id].used_shmem += k->req.num_shmem;
                device_status[device_id].used_thread += k->req.num_threads;
                device_status[device_id].num_running_tasks += 1;
                device_status[device_id].running_tasks.push_back(k->task_id);

                SCHEDULER_LOG("scheduled pending task %d to device %d, allocated memory %ld, free device memory %zu", 
                    k->task_id, device_id, k->req.memory_alloc, device_status[device_id].free_memory);
                SCHEDULER_LOG("current running tasks on device %d: [", device_id);
                for (auto task_id : device_status[device_id].running_tasks) {
                    SCHEDULER_LOG("%d, ", task_id);
                }
                SCHEDULER_LOG("]\n");

                sem_post(&(k->sem));
                SCHEDULER_LOG("signal task %d with status %d, memory %ld to scheduler (head_p=%d, tail_p = %d)", 
                    k->task_id, k->status, k->req.memory_alloc, comm->head_p, comm->tail_p);
            }
            pending_tasks = std::move(unscheduled_tasks);

            // @TODO: tasks in 'pending_tasks' may be overlapped if queue head moves
            pthread_mutex_lock(&comm->lock);
            comm->head_p = tail_p;
            comm->num_running_task = pending_tasks.size();
            pthread_mutex_unlock(&comm->lock);
        }
    }
}

int main() {
    signal(SIGINT, sigint_handler);
    p.start_sampling();
    S.schedule_task();

    return 0;
}
