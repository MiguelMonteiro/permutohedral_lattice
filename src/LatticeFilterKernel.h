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

template<typename Device, typename T>
struct LatticeFilter {
    void operator()(const Device &d,
                    OpKernelContext* context,
                    T *output,
                    const T *input,
                    const T * positions,
                    int num_super_pixels,
                    int pd,
                    int vd,
                    bool reverse);
};


template<typename Device, typename T>
struct ComputeKernel {
    void operator()(const Device &d,
                    OpKernelContext* context,
                    const T *reference_image,
                    T * positions,
                    int num_super_pixels,
                    int n_spatial_dims,
                    int *spatial_dims,
                    int n_reference_channels,
                    T spatial_std,
                    T features_std);
};



#if GOOGLE_CUDA

// Partially specialize functor for GpuDevice.
template<typename T>
struct LatticeFilter<Eigen::GpuDevice, T> {
    void operator()(const Eigen::GpuDevice &d,
                    OpKernelContext* context,
                    T *output,
                    const T *input,
                    const T * positions,
                    int num_super_pixels,
                    int pd,
                    int vd,
                    bool reverse);
};

template<typename  T>
struct ComputeKernel<Eigen::GpuDevice, T> {
    void operator()(const Eigen::GpuDevice &d,
                    OpKernelContext* context,
                    const T * reference_image,
                    T * positions,
                    int num_super_pixels,
                    int n_spatial_dims,
                    int * spatial_dims,
                    int n_reference_channels,
                    T spatial_std,
                    T features_std);
};



#endif


#endif //PERMUTOHEDRAL_LATTICE_BILATERALKERNEL_H
