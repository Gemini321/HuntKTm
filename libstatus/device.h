#ifndef _DEVICE_H_
#define _DEVICE_H_

#include <chrono>
#include <ctime>
#include <iostream>
#include <thread>
#include <vector>
#include <nvml.h>
#include <dcgm_agent.h>

#include "common.h"

int constexpr device_name_length = 64;
class GPUDevice {
private:
    nvmlDevice_t device; // device hanlder
    char name[device_name_length]; // device name
    std::vector<nvmlUtilization_t> utilizations; // contains samplings of utilizations (Cores)
    std::vector<nvmlMemory_v2_st> memory_utilizations; // contains samplings of utilizations (Memory)
    std::vector<double> sm_occupancies; // contains samplings of utilizations (Memory)
public:
    GPUDevice(int device_id) {
        NVML_RT_CALL(nvmlDeviceGetHandleByIndex_v2(device_id, &device));
        NVML_RT_CALL(nvmlDeviceGetName(device, name, device_name_length));

        // NVML_RT_CALL(nvmlGpmSampleAlloc(&gpmSample1));
        // NVML_RT_CALL(nvmlGpmSampleAlloc(&gpmSample2));
        // NVML_RT_CALL(nvmlGpmSampleGet(device, gpmSample1));
        // nvmlGpmSupport_t gpmSupport;
        // gpmSupport.version = NVML_GPM_SUPPORT_VERSION;
        // NVML_RT_CALL(nvmlGpmQueryDeviceSupport(device, &gpmSupport));
        // std::cout << "GPM Support: " << gpmSupport.isSupportedDevice << "\n";
        // metricsGet.version = NVML_GPM_METRICS_GET_VERSION;
        // metricsGet.numMetrics = 1;
        std::cout << "Device " << device_id << "(" << device << ") : " << name << "\n";
    }
    ~GPUDevice() {
        // NVML_RT_CALL(nvmlGpmSampleFree(gpmSample1));
        // NVML_RT_CALL(nvmlGpmSampleFree(gpmSample2));
    }

    void query();
    nvmlUtilization_t get_utilization(int i); // get both gpu and memory utilization
    nvmlMemory_v2_t get_memory_utilization(int i); // get both gpu and memory utilization
    nvmlUtilization_t get_utilization(); // get both gpu and memory utilization
    nvmlMemory_v2_t get_memory_utilization(); // get both gpu and memory utilization
    double get_sm_occupancy(); // get sm occupancy

    nvmlGpmMetricsGet_t metricsGet;
    nvmlGpmSample_t gpmSample1;
    nvmlGpmSample_t gpmSample2;
};

#endif // _DEVICE_H_
