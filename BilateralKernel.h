//
// Created by Miguel Monteiro on 29/01/2018.
//

#ifndef PERMUTOHEDRAL_LATTICE_BILATERALKERNEL_H
#define PERMUTOHEDRAL_LATTICE_BILATERALKERNEL_H


template <typename Device, typename T>
struct ExampleFunctor {
    void operator()(const Device& d, int ref_channels, int input_channels, int num_super_pixels, const T* input, T* image);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename Eigen::GpuDevice, typename T>
struct ExampleFunctor {
  void operator()(const Eigen::GpuDevice& d, int size, const T* in, T* out);
};
#endif


#endif //PERMUTOHEDRAL_LATTICE_BILATERALKERNEL_H
