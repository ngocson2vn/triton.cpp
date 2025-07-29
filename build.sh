#!/bin/bash

set -e

ROOT_DIR=$(pwd)
echo "ROOT_DIR=${ROOT_DIR}"
mkdir -p ${ROOT_DIR}/build

# git submodule update --init --recursive


pre_hash=""
if [ -f ${ROOT_DIR}/.cmake.sha256 ]; then
  pre_hash=$(cat ${ROOT_DIR}/.cmake.sha256)
fi
now_hash=$(sha256sum ${ROOT_DIR}/CMakeLists.txt | awk '{print $1}')

if [ "${now_hash}" != "${pre_hash}" ]; then
  echo "${now_hash} != ${pre_hash}"
  echo
  echo "==================================================="
  echo "Generate ninja build file"
  echo "==================================================="
  cd ${ROOT_DIR}/build
  cmake -G Ninja -DTRITON_CODEGEN_BACKENDS=nvidia .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_PROJECTS="mlir;compiler-rt" \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="Native;X86;NVPTX;AMDGPU" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_LLD=ON \
    -DLLVM_CCACHE_BUILD=ON \
    -DCOMPILER_RT_BUILD_GWP_ASAN=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DCOMPILER_RT_BUILD_SANITIZERS=ON
fi

echo
echo "==================================================="
echo "Run ninja build"
echo "==================================================="
cd ${ROOT_DIR}/build
cmake --build .

yes | echo ${now_hash} > ${ROOT_DIR}/.cmake.sha256
echo "DONE"