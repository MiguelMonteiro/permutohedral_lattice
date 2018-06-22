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

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "LatticeFilterKernel.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "PermutohedralLatticeGPU.cuh"
#include "DeviceMemoryAllocator.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

void check_cuda_error(){
    auto code = cudaGetLastError();
    if(cudaSuccess != cudaGetLastError())
        LOG(FATAL) << "GPU Error: %s\n", cudaGetErrorString(code);
}

template<typename T, int pd, int vd>
struct LatticeFilter<GPUDevice , T, pd, vd>{

    LatticeFilter(OpKernelContext *context,
                  int num_hypervoxels,
                  int num_input_channels,
                  int num_spatial_dims,
                  int num_reference_channels,
                  int *host_spatial_dims,
                  const T *spatial_std,
                  const T *color_std,
                  bool reverse) : context(context),
                                  num_hypervoxels(num_hypervoxels),
                                  num_input_channels(num_input_channels),
                                  num_spatial_dims(num_spatial_dims),
                                  num_reference_channels(num_reference_channels),
                                  spatial_std(spatial_std),
                                  color_std(color_std),
                                  reverse(reverse),
                                  allocator(DeviceMemoryAllocator(context)) {
        if((pd != num_spatial_dims + num_reference_channels || pd != num_spatial_dims) && vd != num_input_channels + 1)
            LOG(FATAL) << "GPU filter not compiled for these spatial dimensions, input_image and/or reference channels";
        allocator.allocate_device_memory<int>((void**)&spatial_dims, num_spatial_dims);
        allocator.memcpy<int>(spatial_dims, host_spatial_dims, num_spatial_dims);
        allocator.allocate_device_memory<T>((void**)&position_vectors, num_hypervoxels * pd);
    }

    void compute_position_vectors(const T *reference_image){
        dim3 blocks((num_hypervoxels - 1) / BLOCK_SIZE + 1, 1, 1);
        dim3 blockSize(BLOCK_SIZE, 1, 1);
        auto stream = context->eigen_device<GPUDevice>().stream();
        compute_position_vectors<T><<<blocks, blockSize, 0, stream>>>(reference_image,
                                                                      position_vectors,
                                                                      num_hypervoxels,
                                                                      num_reference_channels,
                                                                      num_spatial_dims,
                                                                      spatial_dims,
                                                                      spatial_std,
                                                                      color_std);
        check_cuda_error();
    }
    void filter(T* output_image, const T *input_image){
        auto stream = context->eigen_device<GPUDevice>().stream();
        auto lattice = PermutohedralLatticeGPU<T, pd, vd>(num_hypervoxels, &allocator, stream);
        lattice.filter(output_image, input_image, position_vectors, reverse);
        check_cuda_error();
    }

    void operator()(T* output_image, const T *input_image, const T *reference_image){
        compute_position_vectors(reference_image);
        filter(output_image, input_image);
    }

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
    T* position_vectors;
};


// Explicitly instantiate functors for the types of OpKernels registered.
//Bilateral
template struct LatticeFilter<GPUDevice, float, SPATIAL_DIMS + REFERENCE_CHANNELS, INPUT_CHANNELS + 1>;
template struct LatticeFilter<GPUDevice, double, SPATIAL_DIMS + REFERENCE_CHANNELS, INPUT_CHANNELS + 1>;
//Bilateral theta_color grad
template struct LatticeFilter<GPUDevice, float, SPATIAL_DIMS + REFERENCE_CHANNELS, REFERENCE_CHANNELS + 1>;
template struct LatticeFilter<GPUDevice, double, SPATIAL_DIMS + REFERENCE_CHANNELS, REFERENCE_CHANNELS + 1>;
//Bilateral theta_spatial grad
template struct LatticeFilter<GPUDevice, float, SPATIAL_DIMS + REFERENCE_CHANNELS, SPATIAL_DIMS + 1>;
template struct LatticeFilter<GPUDevice, double, SPATIAL_DIMS + REFERENCE_CHANNELS, SPATIAL_DIMS + 1>;
//Gaussian
template struct LatticeFilter<GPUDevice, float, SPATIAL_DIMS, INPUT_CHANNELS + 1>;
template struct LatticeFilter<GPUDevice, double, SPATIAL_DIMS, INPUT_CHANNELS + 1>;
//Gaussian theta_spatial grad
template struct LatticeFilter<GPUDevice, float, SPATIAL_DIMS, SPATIAL_DIMS + 1>;
template struct LatticeFilter<GPUDevice, double, SPATIAL_DIMS, SPATIAL_DIMS + 1>;


#endif  // GOOGLE_CUDA