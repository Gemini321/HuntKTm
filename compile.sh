#!/bin/bash
ROOT_DIR=/root/HuntKTm
MULTISTREAM_BASE_PATH=${ROOT_DIR}/benchmarks/multi_stream

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

# Build multi_stream benchmarks
for BMARK_PATH in ${MULTISTREAM_BMARK_PATHS[@]}; do
    echo -e "\e[32mBuilding benchmark\e[0m: "$BMARK_PATH
    cd $BMARK_PATH
    make clean
    make
done
