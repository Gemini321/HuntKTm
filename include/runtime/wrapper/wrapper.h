#ifndef WRAPPER_H
#define WRAPPER_H

#include <bits/stdint-uintn.h>
#include <cstdio>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef WRAPPER_DEBUG
#define WRAPPER_LOG(format, ...)                                                \
    do {                                                                        \
    fprintf(stderr, "[Wrapper] " format "\n", ##__VA_ARGS__);                   \
    fflush(stderr);                                                             \
    } while (0)
#else
#define WRAPPER_LOG(format, ...) do {} while (0)
#endif

#endif // WRAPPER_H
