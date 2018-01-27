#!/usr/bin/env bash

cmake -DCMAKE_BUILD_TYPE=Debug -D CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -G "CodeBlocks - Unix Makefiles" ~/permutohedral_lattice/
