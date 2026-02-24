#!/bin/bash

set -e

./build/bin/triton_compiler ./triton_dot.ttir
if [ -f ./lowering.mlir ]; then
  echo DONE
  code ./lowering.mlir
else
  echo "Something goes wrong!"
fi
