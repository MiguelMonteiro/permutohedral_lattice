//
// Created by Miguel Monteiro on 29/01/2018.
//

#ifndef PERMUTOHEDRAL_LATTICE_BILATERALKERNEL_H
#define PERMUTOHEDRAL_LATTICE_BILATERALKERNEL_H

#include "tensorflow/core/framework/op_kernel.h"

template<typename Device, typename T>
struct LatticeFilter {
    void operator()(const Device &d,
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
