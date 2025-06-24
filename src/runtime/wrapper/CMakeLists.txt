add_library(Wrapper SHARED
    wrapper.cpp
    runtime_interface.cpp
)
target_compile_features(Wrapper PRIVATE cxx_range_for cxx_auto_type)
set(CMAKE_CXX_FLAGS "-g -O3")

# if(WRAPPER_DEBUG)
#     add_definitions(-DWRAPPER_DEBUG -DRUNTIME_DEBUG)
# endif()
