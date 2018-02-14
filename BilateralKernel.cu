

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "BilateralKernel.h"

#include "tensorflow/core/framework/op_kernel.h"
//#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "PermutohedralLatticeGPU.cuh"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

class OpNotImplemented: public std::exception
{
    virtual const char* what() const throw()
    {
        return "The op was not compiled for the number of input and reference channels used, "
                "recompile op with correct values of pd and vd";
    }
} OpNotImplemented;

template<typename T>
void ComputeKernel<GPUDevice, T>::operator()(const GPUDevice& d,
                                             const T *reference_image,
                                             T * positions,
                                             int num_super_pixels,
                                             int n_spatial_dims,
                                             int *spatial_dims,
                                             int n_reference_channels,
                                             T spatial_std,
                                             T features_std){

    dim3 blocks((num_super_pixels - 1) / BLOCK_SIZE + 1, 1, 1);
    dim3 blockSize(BLOCK_SIZE, 1, 1);

    compute_kernel<T><<<blocks, blockSize, 0, d.stream()>>>(reference_image, positions,
            num_super_pixels, n_reference_channels, n_spatial_dims, spatial_dims, spatial_std, features_std);


};

//declaration of what lattices (pd and vd) can be used


// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void LatticeFilter<GPUDevice, T>::operator()(const GPUDevice& d,
                                             T* output,
                                             const T *input,
                                             const T *positions,
                                             int num_super_pixels,
                                             int pd,
                                             int vd,
                                             bool reverse) {


    if(pd == 5 && vd == 4){
        auto lattice = PermutohedralLatticeGPU<T, 5, 4>(num_super_pixels, d.stream());
        lattice.filter(output, input, positions, reverse);
        return;
    }
    if(pd == 2 && vd == 4){
        auto lattice = PermutohedralLatticeGPU<T, 2, 4>(num_super_pixels, d.stream());
        lattice.filter(output, input, positions, reverse);
        return;
    }
    else{
        throw OpNotImplemented;
    }
}


// Explicitly instantiate functors for the types of OpKernels registered.
template struct ComputeKernel<GPUDevice, float>;
template struct LatticeFilter<GPUDevice, float>;
template struct ComputeKernel<GPUDevice, double>;
template struct LatticeFilter<GPUDevice, double>;
//template struct LatticeFilter<GPUDevice, int32>;

#endif  // GOOGLE_CUDA