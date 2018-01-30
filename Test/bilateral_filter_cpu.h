//
// Created by Miguel Monteiro on 29/01/2018.
//
#include "../PermutohedralLatticeCPU.h"

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

    auto positions = compute_bilateral_kernel(input, n, 3, 2, sdims, theta_alpha, theta_beta);

    lattice_filter_cpu(input, positions, pd, vd, n);

    delete[] positions;
}