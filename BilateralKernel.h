//
// Created by Miguel Monteiro on 29/01/2018.
//

#ifndef PERMUTOHEDRAL_LATTICE_BILATERALKERNEL_H
#define PERMUTOHEDRAL_LATTICE_BILATERALKERNEL_H

#include "tensorflow/core/framework/op_kernel.h"

template<typename Device, typename T>
struct ExampleFunctor {
    void operator()(const Device &d,
                    T * output,
                    const T *input,
                    const T *reference_image,
                    int num_super_pixels,
                    int n_spatial_dims,
                    int *spatial_dims,
                    int n_input_channels,
                    int n_reference_channels,
                    float theta_alpha,
                    float theta_beta);
};


#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct ExampleFunctor<Eigen::GpuDevice, T> {
      void operator()(const Eigen::GpuDevice& d,
                    T * output,
                    const T *input,
                    const T *reference_image,
                    int num_super_pixels,
                    int n_spatial_dims,
                    int *spatial_dims,
                    int n_input_channels,
                    int n_reference_channels,
                    float theta_alpha,
                    float theta_beta);
};
#endif


#endif //PERMUTOHEDRAL_LATTICE_BILATERALKERNEL_H
