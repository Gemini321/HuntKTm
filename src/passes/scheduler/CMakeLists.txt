# SchedulerPass use 8 streams by default
add_library(SchedulerPass MODULE
    stream_scheduler.cpp
    graph.cpp
 )
target_compile_features(SchedulerPass PRIVATE cxx_range_for cxx_auto_type)

# SchedulerPass requires the support of RTTI
set_target_properties(SchedulerPass PROPERTIES COMPILE_FLAGS "-frtti")
# target_compile_definitions(SchedulerPass PRIVATE MAX_STREAM_NUM=8)

# # MemSchedulerPass use 4 streams by default
# add_library(MemSchedulerPass MODULE
#     stream_scheduler.cpp
#     graph.cpp
#  )
# target_compile_features(MemSchedulerPass PRIVATE cxx_range_for cxx_auto_type)

# # SchedulerPass requires the support of RTTI
# set_target_properties(MemSchedulerPass PROPERTIES COMPILE_FLAGS "-frtti")
# target_compile_definitions(MemSchedulerPass PRIVATE MAX_STREAM_NUM=4)
