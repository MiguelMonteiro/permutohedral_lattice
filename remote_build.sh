#!/usr/bin/env bash

CUDA_COMPILER=/usr/local/cuda/bin/nvcc
CXX_COMPILER=/usr/bin/g++-4.8

cmake -DCMAKE_BUILD_TYPE=Debug -D CMAKE_CUDA_COMPILER=${CUDA_COMPILER} -D CMAKE_CXX_COMPILER=${CXX_COMPILER} -D CMAKE_CUDA_HOST_COMPILER=${CXX_COMPILER} -G "CodeBlocks - Unix Makefiles" ~/permutohedral_lattice/