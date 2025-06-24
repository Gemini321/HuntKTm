add_executable(RuntimeScheduler
    scheduler.cpp
)
target_compile_features(RuntimeScheduler PRIVATE cxx_range_for cxx_auto_type)

include_directories(${CMAKE_SOURCE_DIR})

# link C++ runtime and CUDA runtime
message("libstatus directory: ${CMAKE_SOURCE_DIR}/libstatus")
message("CUDA include directory: ${CUDA_INCLUDE_DIRS}")
find_library(CUDART_LIBRARY cudart HINTS ${CUDA_INCLUDE_DIRS}/../lib64)
find_library(LIBSTATUS status HINTS ${CMAKE_SOURCE_DIR}/libstatus)

target_link_directories(RuntimeScheduler PRIVATE ${CMAKE_SOURCE_DIR}/libstatus)
target_link_libraries(RuntimeScheduler PRIVATE rt ${CUDART_LIBRARY} pthread ${LIBSTATUS})
set(CMAKE_CXX_FLAGS "-g -O3")

# if(RUNTIME_SHCEDULER_DEBUG)
#     add_definitions(-DRUNTIME_SHCEDULER_DEBUG)
# endif()
