/*Copyright (c) 2018 Miguel Monteiro, Andrew Adams, Jongmin Baek, Abe Davis

Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

#ifndef PERMUTOHEDRAL_LATTICE_BILATERALKERNEL_H
#define PERMUTOHEDRAL_LATTICE_BILATERALKERNEL_H
#include "tensorflow/core/framework/op_kernel.h"
#include "DeviceMemoryAllocator.h"

using namespace tensorflow;

template<typename Device, typename T, int pd, int vd>
struct LatticeFilter {

    LatticeFilter(OpKernelContext *context,
                  int num_hypervoxels,
                  int num_input_channels,
                  int num_spatial_dims,
                  int num_reference_channels,
                  int *host_spatial_dims,
                  const T *spatial_std,
                  const T *color_std,
                  bool reverse);

    void operator()(T* output_image, const T *input_image, const T *reference_image);

private:
    OpKernelContext *context;
    int num_hypervoxels;
    int num_input_channels;
    int num_spatial_dims;
    int num_reference_channels;
    int *spatial_dims;
    const T* spatial_std;
    const T* color_std;
    bool reverse;
    DeviceMemoryAllocator allocator;
    T * position_vectors;
};


template<typename Device, typename T, int pd, int vd>
struct LatticeFilterDiff {

    LatticeFilterDiff(OpKernelContext *context,
                  int num_hypervoxels,
                  int num_input_channels,
                  int num_spatial_dims,
                  int num_reference_channels,
                  int *host_spatial_dims,
                  const T *spatial_std,
                  const T *color_std,
                  bool reverse);

    void operator()(T* output_image, const T *input_image, const T *reference_image);

private:
    OpKernelContext *context;
    int num_hypervoxels;
    int num_input_channels;
    int num_spatial_dims;
    int num_reference_channels;
    int *spatial_dims;
    const T* spatial_std;
    const T* color_std;
    bool reverse;
    DeviceMemoryAllocator allocator;
    T * position_vectors;
};

#endif //PERMUTOHEDRAL_LATTICE_BILATERALKERNEL_H
