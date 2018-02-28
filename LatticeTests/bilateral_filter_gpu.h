//
// Created by Miguel Monteiro on 29/01/2018.
//


#include <cuda_runtime.h>

extern "C++" void lattice_filter_gpu(float *output, const float *input, const float *positions, int pd, int vd, int n);

extern "C++" void
lattice_filter_gpu(double *output, const double *input, const double *positions, int pd, int vd, int n);

extern "C++" void
compute_bilateral_kernel_gpu(const float *reference, float *positions, int num_super_pixels, int n_reference_channels,
                             int n_spatial_dims, const int *spatial_dims, float theta_alpha, float theta_beta);

extern "C++" void
compute_bilateral_kernel_gpu(const double *reference, double *positions, int num_super_pixels, int n_reference_channels,
                             int n_spatial_dims, const int *spatial_dims, double theta_alpha, double theta_beta);


//for the case where the image is its own reference image
template<typename T>
void bilateral_filter_gpu(T *input,
                          int n_input_channels,
                          int n_sdims,
                          int *spatial_dims,
                          int num_super_pixels,
                          T theta_alpha,
                          T theta_beta) {

    int pd = n_input_channels + n_sdims;
    int vd = n_input_channels + 1;
    int n = num_super_pixels;

    //allocate memory in the GPU, ideally this will be handled by tensorflow later
    T *input_gpu = nullptr;
    T *positions_gpu = nullptr;
    int *sdims_gpu = nullptr;
    cudaMalloc((void **) &(input_gpu), n * (vd - 1) * sizeof(T));
    cudaMemcpy(input_gpu, input, n * (vd - 1) * sizeof(T), cudaMemcpyHostToDevice);
    cudaMalloc((void **) &(positions_gpu), n * pd * sizeof(T));
    cudaMalloc((void **) &(sdims_gpu), n_sdims * sizeof(int));
    cudaMemcpy(sdims_gpu, spatial_dims, n_sdims * sizeof(int), cudaMemcpyHostToDevice);

    T *output_gpu = nullptr;
    cudaMalloc((void **) &(output_gpu), n * (vd - 1) * sizeof(T));

    printf("Constructing kernel...\n");
    compute_bilateral_kernel_gpu(input_gpu, positions_gpu, n, n_input_channels, n_sdims, sdims_gpu, theta_alpha,
                                 theta_beta);

    printf("Calling filter...\n");
    lattice_filter_gpu(output_gpu, input_gpu, positions_gpu, pd, vd, n);

    cudaMemcpy(input, output_gpu, n * (vd - 1) * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(input_gpu);
    cudaFree(positions_gpu);
    cudaFree(output_gpu);
    cudaFree(sdims_gpu);
}
