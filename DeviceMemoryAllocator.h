//
// Created by Miguel Monteiro on 01/03/2018.
//
//This class is responsible for managing GPU memory for GOOGLE_CUDA (tensorflow) or simply CUDA in C++

#ifndef PERMUTOHEDRAL_LATTICE_DEVICEMEMORYALLOCATOR_H
#define PERMUTOHEDRAL_LATTICE_DEVICEMEMORYALLOCATOR_H

#include <cuda_runtime.h>

#define GOOGLE_CUDA
#ifdef GOOGLE_CUDA

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"

using namespace tensorflow;

class DeviceMemoryAllocator {

    OpKernelContext* context;

    //allocator has a capacity to store 100 tensors
    Tensor tensors[100];
    int filled;

    DeviceMemoryAllocator(OpKernelContext* context_): context(context_), filled(0){}

public:
    template<typename t>
    void allocate_device_memory(void ** ptr_address, int num_elements){
        DataType dataType = DT_UINT8;
        num_elements *= sizeof(t);
        auto tensor_ptr = &(tensors[filled]);
        filled++;
        OP_REQUIRES_OK(context, context->allocate_temp(dataType, TensorShape({num_elements}), tensor_ptr));
        *ptr_address = reinterpret_cast<t*>((*tensor_ptr).flat<unsigned char>().data());
    }

    template<typename t> void fill(void * ptr, t value, int num_elements){
        cudaMemset(ptr, value, num_elements * sizeof(t));
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

    //allocator has a capacity to store 100 pointers
    void** ptr_addresses[100];
    int filled;

    DeviceMemoryAllocator(): filled(0){}

    ~DeviceMemoryAllocator(){
        for(int i=0; i < filled; i++)
            cudaFree(*ptr_addresses[i]);
    }

public:
    template<typename t>
    void allocate_device_memory(void ** ptr_address, int num_elements){
        cudaMalloc(ptr_address, num_elements*sizeof(t));
        ptr_address[filled] = ptr_address;
        filled++;
    }

    template<typename t> void fill(void * ptr, t value, int num_elements){
        cudaMemset(ptr, value, num_elements * sizeof(t));
    }


};

#endif //GOOGLE_CUDA

#endif //PERMUTOHEDRAL_LATTICE_DEVICEMEMORYALLOCATOR_H
