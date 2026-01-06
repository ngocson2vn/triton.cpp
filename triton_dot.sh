#!/bin/bash

# Cache dir
export TRITON_CACHE_DIR=./tmp
rm -rfv ${TRITON_CACHE_DIR}/*

# Autotune
export TRITON_PRINT_AUTOTUNING="1"


python3.11 triton_dot.py

find ${TRITON_CACHE_DIR}