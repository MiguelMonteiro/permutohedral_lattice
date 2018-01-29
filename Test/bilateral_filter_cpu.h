//
// Created by Miguel Monteiro on 29/01/2018.
//
#include "../PermutohedralLatticeCPU.h"

static void bilateral_filter_cpu(float *input, float *positions, int reference_channels, int input_channels, int n) {
    int pd = reference_channels;
    int vd = input_channels + 1;
    lattice_filter_cpu(input, positions, pd, vd, n);
}