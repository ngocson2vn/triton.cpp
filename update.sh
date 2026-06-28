#!/bin/bash

set -e

git add bin/triton_test.cpp
git add bin/test_*
git add refs/
git commit -m "Update refs"
git push
