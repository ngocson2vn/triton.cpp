#!/bin/bash

set -e

ROOT_DIR=$(pwd)
echo "ROOT_DIR=${ROOT_DIR}"

# git submodule update --init --recursive

CUDA_VERSION=${CUDA_VERSION:-13.1}
if [ -z ${CUDA_VERSION} ]; then
  echo "CUDA_VERSION is empty! Please export it."
  exit 1
fi
export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
echo "CUDA_HOME=${CUDA_HOME}"

echo
echo "==================================================="
echo "1. Build llvm-project"
echo "==================================================="
if [ ! -f ./.build_llvm.done ]; then
  mkdir -p llvm-project/build
  cmake -G Ninja -S llvm-project/llvm -B llvm-project/build/ \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-D__STDC_FORMAT_MACROS -Wno-c23-extensions -Wno-c2y-extensions -Wno-deprecated-declarations -Wno-unused-command-line-argument" \
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
  
  cmake --build llvm-project/build/
  
  touch ./.build_llvm.done
fi
echo "DONE"

echo
echo "==================================================="
echo "2. Build triton.cpp"
echo "==================================================="
mkdir -p build/
cmake -G Ninja -S . -B build/ \
  -DTRITON_CODEGEN_BACKENDS=nvidia \
  -DMLIR_DIR=${ROOT_DIR}/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=${ROOT_DIR}/llvm-project/build/lib/cmake/llvm

cmake --build build/

echo "DONE"