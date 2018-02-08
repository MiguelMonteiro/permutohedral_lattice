#!/usr/bin/env bash

cmake -DCMAKE_BUILD_TYPE=Debug -D CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -G "CodeBlocks - Unix Makefiles" ~/permutohedral_lattice/



cmake -DCMAKE_BUILD_TYPE=Debug -D CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -D CMAKE_CXX_COMPILER=/usr/bin/g++-4.8 -D CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-4.8 -G "CodeBlocks - Unix Makefiles" ~/permutohedral_lattice/