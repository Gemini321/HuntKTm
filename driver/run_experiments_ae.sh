#!/bin/bash

# set -x

#
# Structure of this bash script:
#
#   for each scheduling algorithm
#     for each workload
#       Start the scheduler
#       Start the workload driver
#       (Workload driver completes)
#       Stop scheduler
#       Parse workload driver output
#       Move scheduler results to results folder
#

source ae_aux.sh

BASE_PATH=/root/HuntKTm
BEMPS_SCHED_PATH=${BASE_PATH}/obj
WORKLOADER_PATH=${BASE_PATH}/driver
WORKLOADS_PATH=${BASE_PATH}/driver/workloads/taco24
RESULTS_PATH=results


function usage_and_exit() {
    echo
    echo "  Usage:"
    echo "    $0 <figure>"
    echo
    echo "    Figure must be one of:"
    echo "      {figure-1, figure-4, figure-5, figure-7}"
    echo
    exit 1
}


if [ "$1" == "motivation-1-1" ]; then
    echo "Running experiments for Movitation-1 results"
    WORKLOADS=(
        motivation-1-ms.wl
    )
    MultiGPU_ARGS_ARR=(
        1
    )
    declare -A SCHED_ALG_TO_ARGS_ARR=(
        [MultiGPU]="MultiGPU_ARGS_ARR"
    )
elif [ "$1" == "motivation-1-2" ]; then
    echo "Running experiments for Movitation-1 results"
    WORKLOADS=(
        motivation-1.wl
        motivation-1-ms.wl
    )
    MultiGPU_ARGS_ARR=(
        2
    )
    declare -A SCHED_ALG_TO_ARGS_ARR=(
        [MultiGPU]="MultiGPU_ARGS_ARR"
    )
elif [ "$1" == "motivation-2-1" ]; then
    echo "Running experiments for Movitation-2 results"
    WORKLOADS=(
        motivation-2-ms.wl
    )
    MultiGPU_ARGS_ARR=(
        2
    )
    declare -A SCHED_ALG_TO_ARGS_ARR=(
        [MultiGPU]="MultiGPU_ARGS_ARR"
    )
elif [ "$1" == "motivation-2-2" ]; then
    echo "Running experiments for Movitation-2 results"
    WORKLOADS=(
        motivation-2-ms-mem.wl
    )
    MultiGPU_ARGS_ARR=(
        3
    )
    declare -A SCHED_ALG_TO_ARGS_ARR=(
        [MultiGPU]="MultiGPU_ARGS_ARR"
    )
elif [ "$1" == "figure-1" ]; then
    echo "Running experiments for original & multi-stream results"
    WORKLOADS=(
        A100_16_16jobs_ms.wl
        A100_16_32jobs_ms.wl
        A100_25_16jobs_ms.wl
        A100_25_32jobs_ms.wl
        A100_33_16jobs_ms.wl
        A100_33_32jobs_ms.wl
        A100_50_16jobs_ms.wl
        A100_50_32jobs_ms.wl
    )
    # 1 workers for 1 GPU MS only
    # MultiGPU_ARGS_ARR=(
    #     1
    # )
    # 2 workers for 2 GPU MS only
    # MultiGPU_ARGS_ARR=(
    #     2
    # )
    # 3 workers for 3 GPU MS only
    # MultiGPU_ARGS_ARR=(
    #     3
    # )
    # 4 workers for MS only
    MultiGPU_ARGS_ARR=(
        4
    )
    declare -A SCHED_ALG_TO_ARGS_ARR=(
        [MultiGPU]="MultiGPU_ARGS_ARR"
    )
elif [ "$1" == "figure-2" ]; then
    echo "Running experiments for memory management results"
    WORKLOADS=(
        A100_16_16jobs_ms.wl
        A100_16_32jobs_ms.wl
        A100_25_16jobs_ms.wl
        A100_25_32jobs_ms.wl
        A100_33_16jobs_ms.wl
        A100_33_32jobs_ms.wl
        A100_50_16jobs_ms.wl
        A100_50_32jobs_ms.wl
    )
    # 3 workers for 1 GPU
    # MultiGPU_ARGS_ARR=(
    #     3
    # )
    # 6 workers for 2 GPU and 20GB memory
    # MultiGPU_ARGS_ARR=(
    #     6
    # )
    # 9 workers for 3 GPU and 30GB memory
    # MultiGPU_ARGS_ARR=(
    #     9
    # )
    # 12 workers for 40GB memory
    MultiGPU_ARGS_ARR=(
        12
    )
    declare -A SCHED_ALG_TO_ARGS_ARR=(
        [MultiGPU]="MultiGPU_ARGS_ARR"
    )
elif [ "$1" == "figure-3" ]; then
    echo "Running experiments for memory management results"
    WORKLOADS=(
        A100_16_16jobs_ms_mem.wl
        A100_16_32jobs_ms_mem.wl
        A100_25_16jobs_ms_mem.wl
        A100_25_32jobs_ms_mem.wl
        A100_33_16jobs_ms_mem.wl
        A100_33_32jobs_ms_mem.wl
        A100_50_16jobs_ms_mem.wl
        A100_50_32jobs_ms_mem.wl
    )
    # # 5 workers for 1 GPU
    # MultiGPU_ARGS_ARR=(
    #     5
    # )
    # 10 workers for 2 GPU and 20GB memory
    # MultiGPU_ARGS_ARR=(
    #     10
    # )
    # # 15 workers for 3 GPU and 30GB memory
    # MultiGPU_ARGS_ARR=(
    #     15
    # )
    # 20 workers for 40GB memory
    MultiGPU_ARGS_ARR=(
        20
    )
    declare -A SCHED_ALG_TO_ARGS_ARR=(
        [MultiGPU]="MultiGPU_ARGS_ARR"
    )
else
    usage_and_exit
fi

ae_run

echo "Experiments complete"
echo "Exiting normally"
