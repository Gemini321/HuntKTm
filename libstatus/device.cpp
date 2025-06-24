#include "device.h"

void GPUDevice::query() {
    nvmlUtilization_t util;
    nvmlMemory_v2_t mem;
    nvmlDeviceGetUtilizationRates(device, &util);
    nvmlDeviceGetMemoryInfo_v2(device, &mem);
    utilizations.push_back(util);
    memory_utilizations.push_back(mem);

    // NVML_RT_CALL(nvmlGpmSampleGet(device, gpmSample1));
    // NVML_RT_CALL(nvmlGpmSampleGet(device, gpmSample2));
    // metricsGet.sample1 = gpmSample1;
    // metricsGet.sample2 = gpmSample2;
    // NVML_RT_CALL(nvmlGpmMetricsGet(&metricsGet));
    // sm_occupancies.push_back(metricsGet.metrics[NVML_GPM_METRIC_GRAPHICS_UTIL].value);
    // gpmSample1 = gpmSample2;
}

double GPUDevice::get_sm_occupancy() {
    if (sm_occupancies.size() == 0) {
        return 0;
    } else { 
        return sm_occupancies.back();
    }
}

nvmlUtilization_t GPUDevice::get_utilization(int i) {
    nvmlUtilization_t util;
    if (i > utilizations.size()) {
        util.gpu = 0;
        util.memory = 0;
    } else { 
        util = utilizations[i];
    }
    return util;
}

nvmlMemory_v2_t GPUDevice::get_memory_utilization(int i) {
    nvmlMemory_v2_t util;
    if (i > memory_utilizations.size()) {
        util.used = 0;
    } else { 
        util = memory_utilizations[i];
    }
    return util;
}

nvmlUtilization_t GPUDevice::get_utilization() {
    nvmlUtilization_t util;
    if (0 == utilizations.size()) {
        util.gpu = 0;
        util.memory = 0;
    } else { 
        util = utilizations.back();
    }
    return util;
}

nvmlMemory_v2_t GPUDevice::get_memory_utilization() {
    nvmlMemory_v2_t util;
    if (0 == memory_utilizations.size()) {
        util.used = 0;
    } else { 
        util = memory_utilizations.back();
    }
    return util;
}
