
#ifndef CUDA_MEMORY_H
#define CUDA_MEMORY_H

size_t GPU_MEMORY_ALLOCATION = 0;

void allocateCudaMemory(void ** pointer, size_t size) {
  cudaMalloc(pointer, size);
  GPU_MEMORY_ALLOCATION += size;
}

#endif
