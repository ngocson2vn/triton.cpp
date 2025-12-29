#!/bin/bash

set -e
set +o noclobber


CURRENT_DIR=$(pwd)
echo "CURRENT_DIR = ${CURRENT_DIR}/"
TRITON_DIR=~/workspace/triton_dev/openai/triton

#======================================================================
# Prerequisites
#======================================================================
# TRITON_DIR=~/git/triton
# TRITON_HASH=984b694dc2916ee4f8cd18d3a28d1d8da14e076d
# mkdir -p ${TRITON_DIR}
# cd ${TRITON_DIR}
# if ! git remote -v 2>&1>/dev/null; then
#   git init
#   git remote add origin https://github.com/triton-lang/triton.git
#   git fetch origin --depth 1 ${TRITON_HASH}
#   git checkout FETCH_HEAD
# fi
#======================================================================

rsync -rvP ${TRITON_DIR}/lib                         ${CURRENT_DIR}/
rsync -rvP ${TRITON_DIR}/include                     ${CURRENT_DIR}/
rsync -rvP ${TRITON_DIR}/third_party/nvidia          ${CURRENT_DIR}/third_party/
rsync -rvP ${TRITON_DIR}/third_party/f2reduce        ${CURRENT_DIR}/third_party/
rsync -rvP ${TRITON_DIR}/third_party/proton          ${CURRENT_DIR}/third_party/
# rsync -rvP ${TRITON_DIR}/third_party/amd             ${CURRENT_DIR}/third_party/
rsync -rvP ${TRITON_DIR}/cmake                       ${CURRENT_DIR}/
rsync -rvP ${TRITON_DIR}/bin/*.h                     ${CURRENT_DIR}/bin/
rsync -rvP ${TRITON_DIR}/bin/*.txt                   ${CURRENT_DIR}/bin/
rsync -rvP ${TRITON_DIR}/python/src                  ${CURRENT_DIR}/python/

git checkout CMakeLists.txt
git checkout bin/CMakeLists.txt

echo
echo "DONE"
