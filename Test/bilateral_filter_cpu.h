//
// Created by Miguel Monteiro on 29/01/2018.
//
#include "../PermutohedralLatticeCPU.h"
#include "memory"
//for the case where the image is its own reference image
static void bilateral_filter_cpu(float *input,
                                 int n_input_channels,
                                 int n_sdims,
                                 int * sdims,
                                 int num_super_pixels,
                                 float theta_alpha,
                                 float theta_beta) {

    int pd = n_input_channels + n_sdims;
    int vd = n_input_channels + 1;
    int n = num_super_pixels;

    printf("Constructing kernel...\n");
    auto positions= new float[num_super_pixels * pd];
    compute_bilateral_kernel_cpu(input, positions, n, 3, 2, sdims, theta_alpha, theta_beta);

    printf("Calling filter...\n");
    auto output = new float[num_super_pixels*n_input_channels]{0};

    lattice_filter_cpu(output, input, positions, pd, vd, n);

    //std::swap(input, output);
    std::memcpy(input, output, num_super_pixels*n_input_channels* sizeof(float));
    delete[] output;
    delete[] positions;
}