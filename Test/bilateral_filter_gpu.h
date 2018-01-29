//
// Created by Miguel Monteiro on 29/01/2018.
//


#include <cuda_runtime.h>

//extern "C++" template<int pd, int vd> void permutohedral::filter(float *values, float *positions, int n);
extern "C++" void lattice_filter_gpu(float *input, float *positions, int pd, int vd, int n);

void bilateral_filter_gpu(float *input, float *positions, int reference_channels, int input_channels, int n) {
    int pd = reference_channels;
    int vd = input_channels + 1;

    //allocate memory in the GPU, ideally this will be handled by tensorflow later
    float * input_gpu = nullptr;
    float * positions_gpu = nullptr;

    cudaMalloc((void**)&(input_gpu), n*(vd-1)*sizeof(float));
    cudaMemcpy(input_gpu, input, n*(vd-1)*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&(positions_gpu), n*pd*sizeof(float));
    cudaMemcpy(positions_gpu, positions, n*pd*sizeof(float), cudaMemcpyHostToDevice);

    lattice_filter_gpu(input_gpu, positions_gpu, pd, vd, n);

    cudaMemcpy(input, input_gpu, n*(vd-1)*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(input_gpu);
    cudaFree(positions_gpu);
}
