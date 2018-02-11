// kernel_example.cu.cc
//#define GOOGLE_CUDA 1

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "BilateralKernel.h"

#include "tensorflow/core/framework/op_kernel.h"
//#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "PermutohedralLatticeGPU.cu"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;


// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void ExampleFunctor<GPUDevice, T>::operator()(const GPUDevice& d,
                                              T* output,
                                              const T *input,
                                              const T *reference_image,
                                              int num_super_pixels,
                                              int n_spatial_dims,
                                              int *spatial_dims,
                                              int n_input_channels,
                                              int n_reference_channels,
                                              float theta_alpha,
                                              float theta_beta,
                                              bool reverse) {

    int pd = n_reference_channels + n_spatial_dims;
    int vd = n_input_channels + 1;
    int n = num_super_pixels;
    //
    int* spatial_dims_gpu;
    cudaMalloc((void**)&(spatial_dims_gpu), n_spatial_dims*sizeof(int));
    cudaMemcpy(spatial_dims_gpu, spatial_dims, n_spatial_dims*sizeof(int), cudaMemcpyHostToDevice);


    T* positions;
    cudaMalloc((void**)&(positions), n*pd*sizeof(T));

    printf("%d %d %d %f %f\n", n_reference_channels, num_super_pixels, n_spatial_dims, theta_alpha, theta_beta);
    for(int i=0; i < n_spatial_dims; i++)
        printf("%d", spatial_dims[i]);

    compute_bilateral_kernel_gpu(reference_image,
                                 positions,
                                 num_super_pixels,
                                 n_reference_channels,
                                 n_spatial_dims,
                                 spatial_dims_gpu,
                                 theta_alpha,
                                 theta_beta);


    lattice_filter_gpu(output, input, positions, pd, vd, n, reverse);
    cudaFree(positions);
    cudaFree(spatial_dims_gpu);
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct ExampleFunctor<GPUDevice, float>;
//template struct ExampleFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA