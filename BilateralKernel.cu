// kernel_example.cu.cc
//#define GOOGLE_CUDA 1

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "BilateralKernel.h"

#include "tensorflow/core/framework/op_kernel.h"
//#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "PermutohedralLatticeGPU.cuh"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;


template<typename T>
void ComputeKernel<GPUDevice, T>::operator()(const GPUDevice& d,
                                                      const T *reference_image,
                                                      T * positions,
                                                      int num_super_pixels,
                                                      int n_spatial_dims,
                                                      int *spatial_dims,
                                                      int n_reference_channels,
                                                      float theta_alpha,
                                                      float theta_beta,
                                                      float theta_gamma,
                                                      bool bilateral){

    dim3 blocks((num_super_pixels - 1) / BLOCK_SIZE + 1, 1, 1);
    dim3 blockSize(BLOCK_SIZE, 1, 1);

    if(bilateral){
        compute_kernel<<<blocks, blockSize, 0, d.stream()>>>(reference_image, positions,
                num_super_pixels, n_reference_channels, n_spatial_dims, spatial_dims, theta_alpha, theta_beta);

    } else {
        compute_kernel<<<blocks, blockSize, 0, d.stream()>>>(nullptr,
                positions, num_super_pixels, 0, n_spatial_dims, spatial_dims, theta_gamma, 0);
    }


};



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
        auto lattice = PermutohedralLatticeGPU<5, 4>(num_super_pixels, d.stream());
        lattice.filter(output, input, positions, reverse);
    }
    else
        return;

}


// Explicitly instantiate functors for the types of OpKernels registered.
template struct ComputeKernel<GPUDevice, float>;
template struct LatticeFilter<GPUDevice, float>;
//template struct LatticeFilter<GPUDevice, int32>;

#endif  // GOOGLE_CUDA