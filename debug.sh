#!/bin/bash

set -e

export SONY_LOG_LEVEL=0

./build/bin/triton_debug ./async_tma_copy_global_to_local.ttngir
if [ -f ./debug_lowering.mlir ]; then
  echo DONE
  code ./debug_lowering.mlir
else
  echo "Something goes wrong!"
fi
