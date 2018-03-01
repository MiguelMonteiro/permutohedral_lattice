#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "LatticeFilterKernel.h"

#include "tensorflow/core/framework/op_kernel.h"
//#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "PermutohedralLatticeGPU.cuh"
#include "DeviceMemoryAllocator.h"

#ifndef SPATIAL_DIMS
#define SPATIAL_DIMS 2
#endif
#ifndef INPUT_CHANNELS
#define INPUT_CHANNELS 3
#endif
#ifndef REFERENCE_CHANNELS
#define REFERENCE_CHANNELS 3
#endif


using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;


template<typename T>
void ComputeKernel<GPUDevice, T>::operator()(const GPUDevice& d,
                                             OpKernelContext* context,
                                             const T *reference_image,
                                             T * positions,
                                             int num_super_pixels,
                                             int n_spatial_dims,
                                             int *spatial_dims,
                                             int n_reference_channels,
                                             T spatial_std,
                                             T features_std){

    auto allocator = DeviceMemoryAllocator(context);

    int* spatial_dims_gpu;
    allocator.allocate_device_memory<int>((void**)&spatial_dims_gpu, n_spatial_dims);
    cudaMemcpy(spatial_dims_gpu, spatial_dims, n_spatial_dims*sizeof(int), cudaMemcpyHostToDevice);

    dim3 blocks((num_super_pixels - 1) / BLOCK_SIZE + 1, 1, 1);
    dim3 blockSize(BLOCK_SIZE, 1, 1);

    compute_kernel<T><<<blocks, blockSize, 0, d.stream()>>>(reference_image, positions,
            num_super_pixels, n_reference_channels, n_spatial_dims, spatial_dims_gpu, spatial_std, features_std);

};

//declaration of what lattices (pd and vd) can be used


// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void LatticeFilter<GPUDevice, T>::operator()(const GPUDevice& d,
                                             OpKernelContext* context,
                                             T* output,
                                             const T *input,
                                             const T *positions,
                                             int num_super_pixels,
                                             int pd,
                                             int vd,
                                             bool reverse) {

    auto allocator = DeviceMemoryAllocator(context);
    //bilateral
    if(pd == SPATIAL_DIMS + REFERENCE_CHANNELS && vd == INPUT_CHANNELS + 1){
        auto lattice = PermutohedralLatticeGPU<T, 5, 4>(num_super_pixels, &allocator, d.stream());
        lattice.filter(output, input, positions, reverse);
        return;
    }
    //spatial only
    if(pd == SPATIAL_DIMS && vd == INPUT_CHANNELS + 1){
        auto lattice = PermutohedralLatticeGPU<T, 2, 4>(num_super_pixels, &allocator, d.stream());
        lattice.filter(output, input, positions, reverse);
        return;
    }
    else{
        LOG(FATAL) << "GPU filter not compiled for these spatial dimensions, input and/or reference channels";
    }
}


// Explicitly instantiate functors for the types of OpKernels registered.
template struct ComputeKernel<GPUDevice, float>;
template struct LatticeFilter<GPUDevice, float>;
template struct ComputeKernel<GPUDevice, double>;
template struct LatticeFilter<GPUDevice, double>;

//template struct LatticeFilter<GPUDevice, int32>;

#endif  // GOOGLE_CUDA