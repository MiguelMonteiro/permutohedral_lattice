#ifndef MIRRORED_ARRAY_H
#define MIRRORED_ARRAY_H

template<typename T>
class MirroredArray {
public:

    T *host;
    T *device;
    size_t size;
    bool owner;

    MirroredArray(size_t len) {
	size = len;
	host = new T[len];
	owner = true;	
#ifdef CUDA_MEMORY_H
	allocateCudaMemory((void**)&(device), len*sizeof(T));
#else
	cudaMalloc((void**)&(device), len*sizeof(T));
#endif
    }

    MirroredArray(T *data, size_t len) {
	size = len;
	host = data;
	owner = false;
	cudaMalloc((void**)&(device), len*sizeof(T));
	hostToDevice();
    }

    void hostToDevice() {
	cudaMemcpy(device, host, size*sizeof(T), cudaMemcpyHostToDevice);
    }
    
    void deviceToHost() {
	cudaMemcpy(host, device, size*sizeof(T), cudaMemcpyDeviceToHost);
	printf("Save: %s\n", cudaGetErrorString(cudaGetLastError()));
    }

    ~MirroredArray() {
	if (owner)
		delete[] host;
	cudaFree(device);
    }
};

#endif
