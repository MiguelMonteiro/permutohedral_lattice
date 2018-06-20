#!/usr/bin/env bash

rm lattice_filter.so
mkdir build_dir
cd build_dir
rm -r *
CUDA_COMPILER=/usr/local/cuda/bin/nvcc
CXX_COMPILER=/usr/bin/g++-4.8

SPATIAL_DIMS=2
INPUT_CHANNELS=3
REFERENCE_CHANNELS=3
MAKE_TESTS=False

cmake -DCMAKE_BUILD_TYPE=Debug -D CMAKE_CUDA_COMPILER=${CUDA_COMPILER} \
                               -D CMAKE_CXX_COMPILER=${CXX_COMPILER} \
                               -D CMAKE_CUDA_HOST_COMPILER=${CXX_COMPILER} \
                               -D SPATIAL_DIMS=${SPATIAL_DIMS} \
                               -D INPUT_CHANNELS=${INPUT_CHANNELS} \
                               -D REFERENCE_CHANNELS=${REFERENCE_CHANNELS} \
                               -D MAKE_TESTS=${MAKE_TESTS} \
                               -G "CodeBlocks - Unix Makefiles" ../


make

cp lattice_filter.so ../

