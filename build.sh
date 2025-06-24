#!/bin/sh
BUILD_DIR="./build"
LIB_DIR="./lib"
SCHEDULERPASS_DIR="./passes/scheduler"
WRAPPERPASS_DIR="./passes/wrapper"
WRAPPER_DIR="./runtime/wrapper"
SCHEDULER_DIR="./runtime/scheduler"
SCHEDULERPASS_NAME="SchedulerPass"
WRAPPERPASS_NAME="WrapperPass"
WRAPPERMEMORYPASS_NAME="WrapperPass"
WRAPPER_NAME="Wrapper"
SCHEDULER_NAME="RuntimeScheduler"
LIB_SCHEDULER="lib${SCHEDULERPASS_NAME}.so"
LIB_WRAPPERPASS="lib${WRAPPERPASS_NAME}.so"
LIB_WRAPPERMEMORYPASS="lib${WRAPPERMEMORYPASS_NAME}.so"
LIB_WRAPPER="lib${WRAPPER_NAME}.so"
LD_DIR="/root/.local/lib"
NVCC_DIR=$(dirname `which nvcc`)
CUDA_DIR=$(realpath ${NVCC_DIR}/..)
CUDA_LIB_DIR="${CUDA_DIR}/lib64"
CLANG_DIR=$(dirname `which clang`)
LLVM_DIR=$(realpath ${CLANG_DIR}/..)
GCC_BIN_DIR=$(dirname `which gcc`)
GCC_DIR=$(realpath ${GCC_BIN_DIR}/..)
CLANG_VERSION=$("${CLANG_DIR}/clang" --version | grep -oE '[0-9\.]+' | head -1)
GCC_VERSION=$("${GCC_BIN_DIR}/gcc" --version | grep -oE '[0-9\.]+' | head -1)
INPUT_NAME="test"
INPUT_FILE="test.cu"
IR_FILE="test.ll"
OPT_FILE="test_opt.ll"
OBJ_FILE="test.o"
BIN_FILE="test"
INPUT_PATH="."
OUTPUT_PATH="./obj"

if [ "${GCC_VERSION}" = "10.2.1" ]; then
    GCC_VERSION="10"
fi

echo "llvm path: ${LLVM_DIR}"
echo "clang version: ${CLANG_VERSION}"
echo "gcc path: ${GCC_DIR}"
echo "gcc version: ${GCC_VERSION}"
echo "cuda dir: ${CUDA_DIR}"

rm -rf ${OUTPUT_PATH}
mkdir ${OUTPUT_PATH}

if [ ! -d "${LIB_DIR}" ]; then
    mkdir ${LIB_DIR}
fi

# rm -rf ${BUILD_DIR}

# build project
CC="clang" CXX="clang++" cmake -B ${BUILD_DIR} \
    -DCUDA_INCLUDE_DIRS=${CUDA_DIR}/include \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1       \
    -DWRAPPER_DEBUG=1                       \
    -DRUNTIME_SHCEDULER_DEBUG=1
cmake --build ${BUILD_DIR} -j 8

# check build status and copy .so
check_build_status_and_copy() {
    OBJ_DIR=$1
    OBJ_LIB=$2
    if [ -s "${BUILD_DIR}/src/${OBJ_DIR}/${OBJ_LIB}" ]; then
        echo "${OBJ_LIB} is built successfully"
        cp ${BUILD_DIR}/src/${OBJ_DIR}/${OBJ_LIB} ${LIB_DIR}
    else
        echo "${OBJ_LIB} is not built"
        exit 1
    fi
}

check_build_status_and_copy "${SCHEDULERPASS_DIR}" "${LIB_SCHEDULER}"
check_build_status_and_copy "${WRAPPERPASS_DIR}" "${LIB_WRAPPERPASS}"
check_build_status_and_copy "${WRAPPERPASS_DIR}" "${LIB_WRAPPERMEMORYPASS}"
check_build_status_and_copy "${WRAPPER_DIR}" "${LIB_WRAPPER}"
cp ${LIB_DIR}/${LIB_WRAPPER} ${LD_DIR}
cp ${BUILD_DIR}/src/${SCHEDULER_DIR}/${SCHEDULER_NAME} ${OUTPUT_PATH}

cd ./libstatus
make
