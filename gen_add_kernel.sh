#!/bin/bash

TRITON_PATH=$(python3 -c "import triton; print(triton.__path__[0])")

if [ ! -d ${TRITON_PATH} ]; then
  echo "Triton path ${TRITON_PATH} doesn't exist!!!"
fi

echo "Triton path: ${TRITON_PATH}"

# Compile a sample kernel
python3 ${TRITON_PATH}/tools/compile.py \
  --kernel-name add_kernel \
  --signature "*fp32,*fp32,*fp32,i32,64" \
  --grid=1024,1,1 \
  ./vector_add.py