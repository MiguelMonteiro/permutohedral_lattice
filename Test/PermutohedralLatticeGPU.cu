//#include "cuda_code_indexing.h"
#include "cuda_runtime.h"
#include "../PermutohedralLatticeGPU.cuh"

//input and positions should be device pointers by this point
void lattice_filter_gpu(float * output, const float *input, const float *positions, int pd, int vd, int n) {
    //vd = image_channels + 1
    if(pd == 5 && vd == 4)
        filter<float, 5, 4>(output, input, positions, n, false);
    else
        return;
}

void compute_bilateral_kernel_gpu(const float * reference,
                                  float * positions,
                                  int num_super_pixels,
                                  int n_reference_channels,
                                  int n_spatial_dims,
                                  const int *spatial_dims,
                                  float theta_alpha,
                                  float theta_beta){

    dim3 blocks((num_super_pixels - 1) / BLOCK_SIZE + 1, 1, 1);
    dim3 blockSize(BLOCK_SIZE, 1, 1);
    compute_kernel<float><<<blocks, blockSize>>>(reference, positions, num_super_pixels, n_reference_channels, n_spatial_dims, spatial_dims, theta_alpha, theta_beta);
};

void compute_spatial_kernel_gpu(float * positions,
                                int num_super_pixels,
                                int n_spatial_dims,
                                const int *spatial_dims,
                                float theta_gamma){

    dim3 blocks((num_super_pixels - 1) / BLOCK_SIZE + 1, 1, 1);
    dim3 blockSize(BLOCK_SIZE, 1, 1);
    compute_kernel<float><<<blocks, blockSize>>>(nullptr, positions, num_super_pixels, 0, n_spatial_dims, spatial_dims, theta_gamma, 0);
};


