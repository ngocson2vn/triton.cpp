#!/bin/bash

# build/bin/run_add_kernel cubinFile kernelName numElements blockSize
# Note: blockSize MUST be equal to the value that was set at compile time
build/bin/run_add_kernel `pwd`/output.cubin add_kernel 500 64
