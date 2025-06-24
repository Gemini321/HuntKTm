#ifndef _COMMON_H_
#define _COMMON_H_

#include <ctime>
#include <nvml.h>

#ifndef NVML_RT_CALL
#define NVML_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<nvmlReturn_t>( call );                                                               \
        if ( status != NVML_SUCCESS )                                                                                  \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA NVML call \"%s\" in line %d of file %s failed "                                      \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     nvmlErrorString( status ),                                                                        \
                     status );                                                                                         \
    }
#endif  // NVML_RT_CALL

#ifndef DCGM_RT_CALL
#define DCGM_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<dcgmReturn_t>( call );                                                               \
        if ( status != DCGM_ST_OK )                                                                                    \
        {                                                                                                              \
            fprintf( stderr,                                                                                           \
                     "ERROR: DCGM call \"%s\" in line %d of file %s failed "                                           \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     errorString( status ),                                                                           \
                     status );                                                                                         \
        }                                                                                                              \
    }
#endif  // DCGM_RT_CALL

typedef struct _stats {
    std::time_t        timestamp;
    // uint               temperature;
    // uint               powerUsage;
    // uint               powerLimit;
    nvmlUtilization_t  utilization;
    nvmlMemory_t       memory;
    // unsigned long long throttleReasons;
    // uint               clockSM;
    // uint               clockGraphics;
    // uint               clockMemory;
    // uint               clockMemoryMax;
    nvmlPstates_t      performanceState;
} stats_t;

#endif // _COMMON_H_