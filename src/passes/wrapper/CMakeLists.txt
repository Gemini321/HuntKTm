add_library(WrapperPass MODULE
    wrapper_pass.cpp
    ../analyzer/analyzer.cpp
    ../analyzer/util.cpp
 )
target_compile_features(WrapperPass PRIVATE cxx_range_for cxx_auto_type)

# LLVM is not compiled with RTTI(Run-Time Type Information) by default
set_target_properties(WrapperPass PROPERTIES COMPILE_FLAGS "-fno-rtti")

add_library(WrapperMemoryPass MODULE
    wrapper_pass.cpp
    ../analyzer/analyzer.cpp
    ../analyzer/util.cpp
 )
target_compile_features(WrapperMemoryPass PRIVATE cxx_range_for cxx_auto_type)

# LLVM is not compiled with RTTI(Run-Time Type Information) by default
set_target_properties(WrapperMemoryPass PROPERTIES COMPILE_FLAGS "-fno-rtti")

target_compile_definitions(WrapperMemoryPass PRIVATE USE_MEMORY_SCHEDULE)

add_library(WrapperNowrapPass MODULE
    wrapper_pass.cpp
    ../analyzer/analyzer.cpp
    ../analyzer/util.cpp
 )
target_compile_features(WrapperNowrapPass PRIVATE cxx_range_for cxx_auto_type)

# LLVM is not compiled with RTTI(Run-Time Type Information) by default
set_target_properties(WrapperNowrapPass PROPERTIES COMPILE_FLAGS "-fno-rtti")

target_compile_definitions(WrapperNowrapPass PRIVATE NO_WRAPPER)

add_library(WrapperMemoryNowrapPass MODULE
    wrapper_pass.cpp
    ../analyzer/analyzer.cpp
    ../analyzer/util.cpp
 )
target_compile_features(WrapperMemoryNowrapPass PRIVATE cxx_range_for cxx_auto_type)

# LLVM is not compiled with RTTI(Run-Time Type Information) by default
set_target_properties(WrapperMemoryNowrapPass PROPERTIES COMPILE_FLAGS "-fno-rtti")

target_compile_definitions(WrapperMemoryNowrapPass PRIVATE USE_MEMORY_SCHEDULE NO_WRAPPER)
