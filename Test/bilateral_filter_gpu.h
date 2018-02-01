//
// Created by Miguel Monteiro on 29/01/2018.
//


#include <cuda_runtime.h>
//#include <unsupported/Eigen/CXX11/Tensor>
//#include "tensorflow/core/framework/op_kernel.h"
//#include "tensorflow/core/util/cuda_kernel_helper.h"
extern "C++" void lattice_filter_gpu(float * output, const float *input, const float *positions, int pd, int vd, int n);
extern "C++" void compute_bilateral_kernel_gpu(const float * reference,
                                                 float * positions,
                                                 int num_super_pixels,
                                                 int n_reference_channels,
                                                 int n_spatial_dims,
                                                 const int *spatial_dims,
                                                 float theta_alpha,
                                                 float theta_beta);

//for the case where the image is its own reference image
void bilateral_filter_gpu(float *input,
                          int n_input_channels,
                          int n_sdims,
                          int * spatial_dims,
                          int num_super_pixels,
                          float theta_alpha,
                          float theta_beta) {
    /*
    {
        Eigen::GpuDevice d;
        float * ptr = d.allocate(9);

    }*/

    int pd = n_input_channels + n_sdims;
    int vd = n_input_channels + 1;
    int n = num_super_pixels;

    //allocate memory in the GPU, ideally this will be handled by tensorflow later
    float * input_gpu = nullptr;
    float * positions_gpu = nullptr;
    int * sdims_gpu = nullptr;
    cudaMalloc((void**)&(input_gpu), n*(vd-1)*sizeof(float));
    cudaMemcpy(input_gpu, input, n*(vd-1)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&(positions_gpu), n*pd*sizeof(float));
    cudaMalloc((void**)&(sdims_gpu), n_sdims*sizeof(int));
    cudaMemcpy(sdims_gpu, spatial_dims, n_sdims*sizeof(int), cudaMemcpyHostToDevice);

    float * output_gpu = nullptr;
    cudaMalloc((void**)&(output_gpu), n*(vd-1)*sizeof(float));

    printf("Constructing kernel...\n");
    compute_bilateral_kernel_gpu(input_gpu, positions_gpu, n, n_input_channels, n_sdims, sdims_gpu, theta_alpha, theta_beta);

    printf("Calling filter...\n");
    lattice_filter_gpu(output_gpu, input_gpu, positions_gpu, pd, vd, n);

    cudaMemcpy(input, output_gpu, n*(vd-1)*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(input_gpu);
    cudaFree(positions_gpu);
    cudaFree(output_gpu);
    cudaFree(sdims_gpu);
}
