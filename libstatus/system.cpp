#include <fstream>
#include <dcgm_agent.h>
#include "common.h"
#include "system.h"

GPUSystem::GPUSystem(std::string filename): fname(filename), loop(false) {
    NVML_RT_CALL(nvmlInit());
    DCGM_RT_CALL(dcgmInit());

    // Query the num of devices 
    NVML_RT_CALL(nvmlDeviceGetCount(&num_devices));

    // Start embedded DCGM and load corresponding modules
    DCGM_RT_CALL(dcgmStartEmbedded(DCGM_OPERATION_MODE_AUTO, &dcgmHandle));

    // initialize DCGM for all devices
    unsigned int DCGMGpuIds[DCGM_MAX_NUM_DEVICES];
    int gpuCount = 0;
    DCGM_RT_CALL(dcgmGetAllDevices(dcgmHandle, DCGMGpuIds, &gpuCount));
    for (int i = 0; i < gpuCount; i ++) {
        printf("DCGM GPU %d: %d\n", i, DCGMGpuIds[i]);
    }
    DCGM_RT_CALL(dcgmGroupCreate(dcgmHandle, DCGM_GROUP_DEFAULT, "my_group", &groupId));
    for (int i = 0; i < num_devices; i ++) {
        dcgmGroupAddDevice(dcgmHandle, groupId, i);
        fp32_utilizations.push_back(std::vector<double>());
        mem_utilizations.push_back(std::vector<double>());
        sm_utilizations.push_back(std::vector<double>());
    }

    // Create a field group for GPU utilization and copy engine utilization
    num_fields = 3;
    fieldIds = new unsigned short[num_fields];
    fieldIds[0] = DCGM_FI_PROF_PIPE_FP32_ACTIVE;
    fieldIds[1] = DCGM_FI_DEV_MEM_COPY_UTIL;
    fieldIds[2] = DCGM_FI_PROF_SM_OCCUPANCY;
    DCGM_RT_CALL(dcgmFieldGroupCreate(dcgmHandle, num_fields, fieldIds, "util_fg", &fieldGroupId));

    // Watch the fields (frequency of 50ms, duration of 300s, no limitation for sample number)
    DCGM_RT_CALL(dcgmWatchFields(dcgmHandle, groupId, fieldGroupId, 200000, 300, 0));

    // for each device create and GPUDevice object
    for (int i = 0; i < num_devices; i++) {
        devices.push_back(new GPUDevice(i));
    }
}

GPUSystem::~GPUSystem() {
    for (int i = 0; i < num_devices; i++) {
        delete devices[i];
    } 
    NVML_RT_CALL(nvmlShutdown());

    // stop DCGM
    NVML_RT_CALL(dcgmFieldGroupDestroy(dcgmHandle, fieldGroupId));
    NVML_RT_CALL(dcgmGroupDestroy(dcgmHandle, groupId));
    NVML_RT_CALL(dcgmStopEmbedded(dcgmHandle));
    NVML_RT_CALL(dcgmShutdown());
}

void GPUSystem::start() {
    loop = true;
    while(loop) {
        std::time_t stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        timestamps.push_back(stamp);
        for (int i = 0; i < num_devices; i++) {
            // DCGM query
            dcgmFieldValue_v1 fieldValues[3];
            DCGM_RT_CALL(dcgmGetLatestValuesForFields(dcgmHandle, i, fieldIds, num_fields, fieldValues));
            DCGM_RT_CALL(fieldValues[0].status);
            DCGM_RT_CALL(fieldValues[1].status);
            DCGM_RT_CALL(fieldValues[2].status);
            fp32_utilizations[i].push_back(fieldValues[0].value.dbl);
            mem_utilizations[i].push_back(fieldValues[1].value.i64);
            sm_utilizations[i].push_back(fieldValues[2].value.dbl);

            devices[i]->query();
            // print for debug
            // nvmlUtilization_t u = devices[i]->get_utilization();
            // nvmlMemory_v2_t m = devices[i]->get_memory_utilization();
            // double sm_occupancy = devices[i]->get_sm_occupancy();
            // std::cout << "Device " << i 
            //           << ": GPU util: " << u.gpu 
            //           << ", MEM Util: " << int(m.used / m.total * 100) 
            //           << ", SM Occupancy: " << sm_occupancy << "\n";
            //           << ", FP32 util: " << fieldValues[0].value.dbl << ", (int) " << fieldValues[0].value.i64
            //           << ", copy engine util:" << fieldValues[1].value.dbl << ", (int) " << fieldValues[1].value.i64
            //           << ", SM util: " << fieldValues[2].value.dbl << ", (int) " << fieldValues[2].value.i64 << "\n";
        }
        std::this_thread::sleep_for( std::chrono::milliseconds( 50 ) );
    }
}

void GPUSystem::stop() {
    std::this_thread::sleep_for( std::chrono::seconds( 2 ) ); // Retrive a few empty samples
    loop = false;
}

void GPUSystem::print_header(std::ofstream &ofs) {
    ofs << "timestamp";
    for (int i = 0; i < num_devices; i++) ofs << ",device_" << i;
    ofs << "\n";
}

void GPUSystem::dump() {
    std::ofstream core_util_file(fname+"_gpu.csv", std::ios::out);
    std::ofstream mem_util_file(fname+"_mem.csv", std::ios::out);
    std::ofstream fp32_util_file(fname+"_fp32_util.csv", std::ios::out);
    std::ofstream mem_engine_file(fname+"_mem_copy.csv", std::ios::out);
    std::ofstream sm_occupancy_file(fname+"_sm_occupancy.csv", std::ios::out);

    print_header(core_util_file);
    print_header(mem_util_file);
    print_header(fp32_util_file);
    print_header(mem_engine_file);
    print_header(sm_occupancy_file);

    for(int i = 0; i < timestamps.size(); i++) {
        core_util_file << timestamps[i];
        mem_util_file << timestamps[i];
        fp32_util_file << timestamps[i];
        mem_engine_file << timestamps[i];
        sm_occupancy_file << timestamps[i];
        for (int j = 0; j < num_devices; j++) {
            nvmlUtilization_t u = devices[j]->get_utilization(i);
            nvmlMemory_v2_t m = devices[j]->get_memory_utilization(i);
            core_util_file << "," << u.gpu;
            mem_util_file << "," << (int)((double)m.used / m.total * 100);

            fp32_util_file << "," << fp32_utilizations[j][i];
            mem_engine_file << "," << mem_utilizations[j][i];
            sm_occupancy_file << "," << sm_utilizations[j][i];
        }
        core_util_file << "\n";
        mem_util_file << "\n";
        fp32_util_file << "\n";
        mem_engine_file << "\n";
        sm_occupancy_file << "\n";
    }
    core_util_file.close();
    mem_util_file.close();
    fp32_util_file.close();
    mem_engine_file.close();
    sm_occupancy_file.close();
}