#!/bin/bash
ROOT_DIR=/root/HuntKTm
MULTISTREAM_BASE_PATH=${ROOT_DIR}/benchmarks/multi_stream_single
TASKFLOW_BASE_PATH=/root/taskflow/benchmarks/multi_stream_single/obj

MULTISTREAM_BMARK_PATHS=(
    ${MULTISTREAM_BASE_PATH}/b1
    ${MULTISTREAM_BASE_PATH}/b5
    ${MULTISTREAM_BASE_PATH}/b6
    ${MULTISTREAM_BASE_PATH}/b8
    ${MULTISTREAM_BASE_PATH}/b10
    ${MULTISTREAM_BASE_PATH}/b14
    ${MULTISTREAM_BASE_PATH}/b15
)

./build.sh

rm ${ROOT_DIR}/time.csv
touch ${ROOT_DIR}/time.csv

# Build multi_stream benchmarks
for BMARK_PATH in ${MULTISTREAM_BMARK_PATHS[@]}; do
    echo -e "\e[32mBuilding benchmark\e[0m: "$BMARK_PATH
    cd $BMARK_PATH
    make clean
    make
done

# Run multi_stream benchmarks
echo "Benchmark,Original Time (ms),Taskflow Time(ms),MS Time (ms),MS Mem Time (ms)" > ${ROOT_DIR}/time.csv
# Temporary files to store the output
original_time_file=$(mktemp)
ms_time_file=$(mktemp)
ms_mem_time_file=$(mktemp)
taskflow_time_file=$(mktemp)
for BMARK_PATH in ${MULTISTREAM_BMARK_PATHS[@]}; do
    echo -e "\e[32mRunning benchmark\e[0m: "$BMARK_PATH
    cd $BMARK_PATH
    BMARK_NAME=$(basename $BMARK_PATH)
    ./${BMARK_NAME}_nowrap | grep "device execution time:" | awk '{print $5}' > $original_time_file
    ./${BMARK_NAME}_ms_nowrap | grep "device execution time:" | awk '{print $5}' > $ms_time_file
    ./${BMARK_NAME}_ms_mem_nowrap | grep "device execution time:" | awk '{print $5}' > $ms_mem_time_file
    
    # cd $TASKFLOW_BASE_PATH
    # ./${BMARK_NAME} | grep "device execution time:" | awk '{print $5}' > $taskflow_time_file

    # Read the output from the temporary files
    original_time=$(cat $original_time_file)
    ms_time=$(cat $ms_time_file)
    ms_mem_time=$(cat $ms_mem_time_file)
    # taskflow_time=$(cat $taskflow_time_file)
    taskflow_time=0

    echo "$BMARK_NAME,$original_time,$taskflow_time,$ms_time,$ms_mem_time" >> ${ROOT_DIR}/time.csv
    echo "$BMARK_NAME,$original_time,$taskflow_time,$ms_time,$ms_mem_time"
done

# Print the contents of the CSV files
echo -e "\e[32mTime CSV:\e[0m"
cat ${ROOT_DIR}/time.csv

# Clean up temporary files
rm $original_time_file $ms_time_file $ms_mem_time_file $taskflow_time_file
