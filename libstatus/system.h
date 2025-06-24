#ifndef _SYSTEM_H_
#define _SYSTEM_H_

#include <iostream>
#include <vector>
#include <nvml.h>

#include "common.h"
#include "device.h"

class GPUSystem {
private:
    unsigned int num_devices;
    std::string fname; // file name for logging data
    std::vector<GPUDevice *> devices;
    std::vector<std::time_t> timestamps;
    bool loop;

    // DCGM variables
    dcgmHandle_t dcgmHandle;
    dcgmGpuGrp_t groupId;
    dcgmFieldGrp_t fieldGroupId;
    int num_fields;
    unsigned short *fieldIds;

    std::vector<std::vector<double> > sm_utilizations;
    std::vector<std::vector<double> > fp32_utilizations;
    std::vector<std::vector<double> > mem_utilizations;

    void print_header(std::ofstream &ofs);

public:
    GPUSystem(std::string filename);
    void start();
    void stop();
    void dump();
    ~GPUSystem();
};


#endif // _SYSTEM_H_