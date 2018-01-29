// kernel_example.cu.cc
#define GOOGLE_CUDA
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "BilateralKernel.h"

#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "permutohedral.cu"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;



// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void ExampleFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, int ref_channels, int input_channels, int num_super_pixels, const T* input, T* image) {

    filter(input, reference, n);
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct ExampleFunctor<GPUDevice, float>;
template struct ExampleFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA