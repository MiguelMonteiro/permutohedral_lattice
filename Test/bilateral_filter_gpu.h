//
// Created by Miguel Monteiro on 29/01/2018.
//


#include <cuda_runtime.h>

extern "C++" void lattice_filter_gpu(float *input, float *positions, int pd, int vd, int n);
extern "C++" void compute_bilateral_kernel_gpu(const float * reference,
                                                 float * positions,
                                                 int num_super_pixels,
                                                 int reference_channels,
                                                 int n_sdims,
                                                 const int *sdims,
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

    printf("Constructing kernel...\n");
    compute_bilateral_kernel_gpu(input_gpu, positions_gpu, n, n_input_channels, n_sdims, sdims_gpu, theta_alpha, theta_beta);

    printf("Calling filter...\n");
    lattice_filter_gpu(input_gpu, positions_gpu, pd, vd, n);

    cudaMemcpy(input, input_gpu, n*(vd-1)*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(input_gpu);
    cudaFree(positions_gpu);
    cudaFree(sdims_gpu);
}
