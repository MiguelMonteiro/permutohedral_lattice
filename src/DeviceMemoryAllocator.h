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

//This class is responsible for managing GPU memory for GOOGLE_CUDA (tensorflow) or simply CUDA in C++

#ifndef PERMUTOHEDRAL_LATTICE_DEVICEMEMORYALLOCATOR_H
#define PERMUTOHEDRAL_LATTICE_DEVICEMEMORYALLOCATOR_H


#include <cuda_runtime.h>
// function for debugging cuda calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if(code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#ifdef GOOGLE_CUDA

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"


using namespace tensorflow;

class DeviceMemoryAllocator {

    OpKernelContext* context;

    //allocator has a capacity to store 100 tensors
    Tensor tensors[10];
    int filled;

public:

    DeviceMemoryAllocator(OpKernelContext* context_): context(context_), filled(0){}

    template<typename t>
    void allocate_device_memory(void ** ptr_address, int num_elements){
        DataType dataType = DT_UINT8;
        num_elements *= sizeof(t);
        auto tensor_ptr = &(tensors[filled]);
        filled++;
        //OP_REQUIRES_OK(context, context->allocate_temp(dataType, TensorShape({num_elements}), tensor_ptr));
        auto status = context->allocate_temp(dataType, TensorShape({num_elements}), tensor_ptr);
        if(!status.ok()){
            LOG(FATAL) << "GPU memory allocation failed (might be due to insufficient memory)\n";
        }
        *ptr_address = reinterpret_cast<t*>((*tensor_ptr).flat<unsigned char>().data());
    }

    template<typename t> void memset(void * ptr, t value, int num_elements){
        cudaMemset(ptr, value, num_elements * sizeof(t));
    }

    template<typename t> void memcpy(void * device_ptr, void * host_ptr, int num_elements){
        cudaMemcpy(device_ptr, host_ptr, num_elements * sizeof(t), cudaMemcpyHostToDevice);
    }

    /*template<typename t>
    void allocate_device_memory(void ** ptr_address, Tensor *tensor_ptr, int num_elements){
        DataType dataType = DataTypeToEnum<t>::v();
        OP_REQUIRES_OK(context, context->allocate_temp(dataType, TensorShape({num_elements}), tensor_ptr));
        *ptr_address = (*tensor_ptr).flat<t>().data();
    }*/
};
#else

class DeviceMemoryAllocator {

    //allocator has a capacity to store 10 pointers
    void** ptr_addresses[10];
    int filled;

public:

    DeviceMemoryAllocator(): filled(0){}

    ~DeviceMemoryAllocator(){
        for(int i=0; i < filled; i++)
            gpuErrchk(cudaFree(*ptr_addresses[i]));
    }

    template<typename t>
    void allocate_device_memory(void ** ptr_address, int num_elements){
        gpuErrchk(cudaMalloc(ptr_address, num_elements*sizeof(t)));
        ptr_addresses[filled] = ptr_address;
        filled++;
    }

    template<typename t> void memset(void * ptr, t value, int num_elements){
        gpuErrchk(cudaMemset(ptr, value, num_elements * sizeof(t)));
    }

    template<typename t> void memcpy(void * device_ptr, void * host_ptr, int num_elements){
        gpuErrchk(cudaMemcpy(device_ptr, host_ptr, num_elements * sizeof(t), cudaMemcpyHostToDevice));
    }

};

#endif //GOOGLE_CUDA

#endif //PERMUTOHEDRAL_LATTICE_DEVICEMEMORYALLOCATOR_H

