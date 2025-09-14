#!/bin/bash

set -e

./build/bin/triton_compiler ./add_kernel.ttir > compile.log 2>&1
if [ -f ./lowering.mlir ]; then
  echo DONE
  code ./lowering.mlir
fi
