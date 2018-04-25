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

#ifndef PERMUTOHEDRAL_CU
#define PERMUTOHEDRAL_CU

#define BLOCK_SIZE 256

#include <cstdio>
#include <utility>
#include "cuda_code_indexing.h"
#include "cuda_runtime.h"
#include "DeviceMemoryAllocator.h"

//64 bit implementation not implemented for compute capability < 6.0
// none trivial performance cost for compute capability < 6.0
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

void cudaErrorCheck(){
    auto code = cudaGetLastError();
    if(cudaSuccess != code){
        fprintf(stderr,"GPU Error: %s\n", cudaGetErrorString(code));
        exit(code);
    }
}

template<typename T, int pd, int vd>class HashTableGPU{
public:
    int capacity;
    T * values;
    short * keys;
    int * entries;

    HashTableGPU(int capacity_, DeviceMemoryAllocator* allocator): capacity(capacity_), values(nullptr), keys(nullptr), entries(nullptr){

        allocator->allocate_device_memory<T>((void**)&values, capacity * vd);
        allocator->memset<T>((void*)values, 0, capacity * vd);

        allocator->allocate_device_memory<int>((void**)&entries, capacity * 2);
        allocator->memset<int>((void*)entries, -1, capacity * 2);

        allocator->allocate_device_memory<short>((void**)&keys, capacity * pd);
        allocator->memset<short>((void*)keys, 0, capacity * pd);
    }

    __device__ int modHash(unsigned int n){
        return(n % (2 * capacity));
    }

    __device__ unsigned int hash(short *key) {
        unsigned int k = 0;
        for (int i = 0; i < pd; i++) {
            k += key[i];
            k = k * 2531011;
        }
        return k;
    }

    __device__ int insert(short *key, unsigned int slot) {
        int h = modHash(hash(key));
        while (1) {
            int *e = entries + h;

            // If the cell is empty (-1), lock it (-2)
            int contents = atomicCAS(e, -1, -2);

            if (contents == -2){
                // If it was locked already, move on to the next cell
            }else if (contents == -1) {
                // If it was empty, we successfully locked it. Write our key.
                for (int i = 0; i < pd; i++) {
                    keys[slot * pd + i] = key[i];
                }
                // Unlock
                atomicExch(e, slot);
                return h;
            } else {
                // The cell is unlocked and has a key in it, check if it matches
                bool match = true;
                for (int i = 0; i < pd && match; i++) {
                    match = (keys[contents*pd+i] == key[i]);
                }
                if (match)
                    return h;
            }
            // increment the bucket with wraparound
            h++;
            if (h == capacity*2)
                h = 0;
        }
    }

    __device__ int retrieve(short *key) {

        int h = modHash(hash(key));
        while (1) {
            int *e = entries + h;

            if (*e == -1)
                return -1;

            bool match = true;
            for (int i = 0; i < pd && match; i++) {
                match = (keys[(*e)*pd+i] == key[i]);
            }
            if (match)
                return *e;

            h++;
            if (h == capacity*2)
                h = 0;
        }
    }
};

template<typename T> struct MatrixEntry {
    int index;
    T weight;
};

template<typename T, int pd, int vd>
__global__ static void createLattice(const int n,
                                     const T *positions,
                                     const T *scaleFactor,
                                     MatrixEntry<T> *matrix,
                                     HashTableGPU<T, pd, vd> table) {

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n)
        return;

    T elevated[pd + 1];
    const T *position = positions + idx * pd;
    int rem0[pd + 1];
    int rank[pd + 1];

    // embed position vector into the hyperplane
    // first rotate position into the (pd+1)-dimensional hyperplane
    // sm contains the sum of 1..n of our feature vector
    T sm = 0;
    for (int i = pd; i > 0; i--) {
        T cf = position[i - 1] * scaleFactor[i - 1];
        elevated[i] = sm - i * cf;
        sm += cf;
    }
    elevated[0] = sm;


    // Find the closest 0-colored simplex through rounding
    // greedily search for the closest zero-colored lattice point
    short sum = 0;
    for (int i = 0; i <= pd; i++) {
        T v = elevated[i] * (1.0 / (pd + 1));
        T up = ceil(v) * (pd + 1);
        T down = floor(v) * (pd + 1);
        if (up - elevated[i] < elevated[i] - down) {
            rem0[i] = (short) up;
        } else {
            rem0[i] = (short) down;
        }
        sum += rem0[i];
    }
    sum /= pd + 1;


    // Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the features values)
    for (int i = 0; i <= pd; i++)
        rank[i] = 0;
    for (int i = 0; i < pd; i++) {
        double di = elevated[i] - rem0[i];
        for (int j = i + 1; j <= pd; j++)
            if (di < elevated[j] - rem0[j])
                rank[i]++;
            else
                rank[j]++;
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    for (int i = 0; i <= pd; i++) {
        rank[i] += sum;
        if (rank[i] < 0) {
            rank[i] += pd + 1;
            rem0[i] += pd + 1;
        } else if (rank[i] > pd) {
            rank[i] -= pd + 1;
            rem0[i] -= pd + 1;
        }
    }


    T barycentric[pd + 2]{0};
    // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
    for (int i = 0; i <= pd; i++) {
        T delta = (elevated[i] - rem0[i]) * (1.0 / (pd + 1));
        barycentric[pd - rank[i]] += delta;
        barycentric[pd + 1 - rank[i]] -= delta;
    }
    // Wrap around
    barycentric[0] += 1.0 + barycentric[pd + 1];


    short key[pd];
    for (int remainder = 0; remainder <= pd; remainder++) {
        // Compute the location of the lattice point explicitly (all but
        // the last coordinate - it's redundant because they sum to zero)
        for (int i = 0; i < pd; i++) {
            key[i] = static_cast<short>(rem0[i] + remainder);
            if (rank[i] > pd - remainder)
                key[i] -= (pd + 1);
        }

        MatrixEntry<T> r;
        unsigned int slot = static_cast<unsigned int>(idx * (pd + 1) + remainder);
        r.index = table.insert(key, slot);
        r.weight = barycentric[remainder];
        matrix[idx * (pd + 1) + remainder] = r;
    }
}

template<typename T, int pd, int vd>
__global__ static void cleanHashTable(int n, HashTableGPU<T, pd, vd> table) {

    const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;

    if (idx >= n)
        return;

    // find my hash table entry
    int *e = table.entries + idx;

    // Check if I created my own key in the previous phase
    if (*e >= 0) {
        // Rehash my key and reset the pointer in order to merge with
        // any other pixel that created a different entry under the
        // same key. If the computation was serial this would never
        // happen, but sometimes race conditions can make the same key
        // be inserted twice. hashTableRetrieve always returns the
        // earlier, so it's no problem as long as we rehash now.
        *e = table.retrieve(table.keys + *e * pd);
    }
}

template<typename T, int pd, int vd>
__global__ static void splatCache(const int n, const T *values, MatrixEntry<T> *matrix, HashTableGPU<T, pd, vd> table) {

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int threadId = threadIdx.x;
    const int color = blockIdx.y;
    const bool outOfBounds = (idx >= n);

    __shared__ int sharedOffsets[BLOCK_SIZE];
    __shared__ T sharedValues[BLOCK_SIZE * vd];
    int myOffset = -1;
    T *myValue = sharedValues + threadId * vd;

    if (!outOfBounds) {

        T *value = const_cast<T *>(values + idx * (vd - 1));

        MatrixEntry<T> r = matrix[idx * (pd + 1) + color];

        // convert the matrix entry from a pointer into the entries array to a pointer into the keys/values array
        matrix[idx * (pd + 1) + color].index = r.index = table.entries[r.index];
        // record the offset into the keys/values array in shared space
        myOffset = sharedOffsets[threadId] = r.index * vd;

        for (int j = 0; j < vd - 1; j++) {
            myValue[j] = value[j] * r.weight;
        }
        myValue[vd - 1] = r.weight;

    } else {
        sharedOffsets[threadId] = -1;
    }

    __syncthreads();

    // am I the first thread in this block to care about this key?
    if (outOfBounds)
        return;

    for (int i = 0; i < BLOCK_SIZE; i++) {
        if (i < threadId) {
            if (myOffset == sharedOffsets[i]) {
                // somebody else with higher priority cares about this key
                return;
            }
        } else if (i > threadId) {
            if (myOffset == sharedOffsets[i]) {
                // someone else with lower priority cares about this key, accumulate it into mine
                for (int j = 0; j < vd; j++) {
                    sharedValues[threadId * vd + j] += sharedValues[i * vd + j];
                }
            }
        }
    }

    // only the threads with something to write to main memory are still going
    T *val = table.values + myOffset;
    for (int j = 0; j < vd; j++) {
        atomicAdd(val + j, myValue[j]);
    }
}

template<typename T, int pd, int vd>
__global__ static void blur(int n, T *newValues, MatrixEntry<T> *matrix, int color, HashTableGPU<T, pd, vd> table) {

    const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;
    if (idx >= n)
        return;

    // Check if I'm valid
    if (matrix[idx].index != idx)
        return;


    // find my key and the keys of my neighbors
    short myKey[pd + 1];
    short np[pd + 1];
    short nm[pd + 1];


    for (int i = 0; i < pd; i++) {
        myKey[i] = table.keys[idx * pd + i];
        np[i] = myKey[i] + 1;
        nm[i] = myKey[i] - 1;
    }
    np[color] -= pd + 1;
    nm[color] += pd + 1;

    int offNp = table.retrieve(np);
    int offNm = table.retrieve(nm);

    T *valMe = table.values + vd * idx;
    T *valOut = newValues + vd * idx;

    //in case neighbours don't exist (lattice edges) offNp and offNm are -1
    T zeros[vd]{0};
    T *valNp = zeros; //or valMe? for edges?
    T *valNm = zeros;
    if(offNp >= 0)
        valNp = table.values + vd * offNp;
    if(offNm >= 0)
        valNm = table.values + vd * offNm;


    for (int i = 0; i < vd; i++)
        valOut[i] = 0.25 * valNp[i] + 0.5 * valMe[i] + 0.25 * valNm[i];
    //valOut[i] = 0.5f * valNp[i] + 1.0f * valMe[i] + 0.5f * valNm[i];
}

template<typename T, int pd, int vd>
__global__ static void slice(const int n, T *values, MatrixEntry<T> *matrix, HashTableGPU<T, pd, vd> table) {

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n)
        return;

    T value[vd-1]{0};
    T weight = 0;

    for (int i = 0; i <= pd; i++) {
        MatrixEntry<T> r = matrix[idx * (pd + 1) + i];
        T *val = table.values + r.index * vd;
        for (int j = 0; j < vd - 1; j++) {
            value[j] += r.weight * val[j];
        }
        weight += r.weight * val[vd - 1];
    }

    weight = 1.0 / weight;
    for (int j = 0; j < vd - 1; j++)
        values[idx * (vd - 1) + j] = value[j] * weight;
}

template<typename T, int pd, int vd>class PermutohedralLatticeGPU{
public:
    int n; //number of pixels/voxels etc..
    T * scaleFactor;
    MatrixEntry<T>* matrix;
    HashTableGPU<T, pd, vd> hashTable;
    cudaStream_t stream;
    T * newValues; // auxiliary array for blur stage

    void init_scaleFactor(DeviceMemoryAllocator* allocator){
        T hostScaleFactor[pd];
        T invStdDev = (pd + 1) * sqrt(2.0f / 3);
        for (int i = 0; i < pd; i++) {
            hostScaleFactor[i] = 1.0f / (sqrt((T) (i + 1) * (i + 2))) * invStdDev;
        }
        allocator->allocate_device_memory<T>((void**)&scaleFactor, pd);
        allocator->memcpy<T>((void*)scaleFactor, (void*)hostScaleFactor, pd);
    }

    void init_matrix(DeviceMemoryAllocator* allocator){
        allocator->allocate_device_memory<MatrixEntry<T>>((void**)&matrix, n * (pd + 1));
    }

    void init_newValues(DeviceMemoryAllocator* allocator){
        allocator->allocate_device_memory<T>((void**)&newValues,  n * (pd + 1) * vd);
        allocator->memset<T>((void *)newValues, 0, n * (pd + 1) * vd);
    }

    PermutohedralLatticeGPU(int n_, DeviceMemoryAllocator* allocator, cudaStream_t stream_=0):
            n(n_),
            scaleFactor(nullptr),
            matrix(nullptr),
            newValues(nullptr),
            hashTable(HashTableGPU<T, pd, vd>(n * (pd + 1), allocator)),
            stream(stream_){

        if (n >= 65535 * BLOCK_SIZE) {
            printf("Not enough GPU memory (on x axis, you can change the code to use other grid dims)\n");
            //this should crash the program
        }

        // initialize device memory
        init_scaleFactor(allocator);
        init_matrix(allocator);
        init_newValues(allocator);
    }

    // values and position must already be device pointers
    void filter(T* output, const T* inputs, const T*  positions, bool reverse){

        dim3 blocks((n - 1) / BLOCK_SIZE + 1, 1, 1);
        dim3 blockSize(BLOCK_SIZE, 1, 1);
        int cleanBlockSize = 128;
        dim3 cleanBlocks((n - 1) / cleanBlockSize + 1, 2 * (pd + 1), 1);

        createLattice<T, pd, vd> <<<blocks, blockSize, 0, stream>>>(n, positions, scaleFactor, matrix, hashTable);
        cudaErrorCheck();

        cleanHashTable<T, pd, vd> <<<cleanBlocks, cleanBlockSize, 0, stream>>>(2 * n * (pd + 1), hashTable);
        cudaErrorCheck();

        blocks.y = pd + 1;
        splatCache<T, pd, vd><<<blocks, blockSize, 0, stream>>>(n, inputs, matrix, hashTable);
        cudaErrorCheck();

        for (int remainder=reverse?pd:0; remainder >= 0 && remainder <= pd; reverse?remainder--:remainder++) {
            blur<T, pd, vd><<<cleanBlocks, cleanBlockSize, 0, stream>>>(n * (pd + 1), newValues, matrix, remainder, hashTable);
            cudaErrorCheck();
            std::swap(hashTable.values, newValues);
        }
        blockSize.y = 1;
        slice<T, pd, vd><<<blocks, blockSize, 0, stream>>>(n, output, matrix, hashTable);
        cudaErrorCheck();
    }
};

template<typename T>
__global__ static void compute_kernel(const T * reference,
                                      T * positions,
                                      int num_super_pixels,
                                      int reference_channels,
                                      int n_sdims,
                                      const int *sdims,
                                      T spatial_std,
                                      T feature_std){

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_super_pixels)
        return;

    int num_dims = n_sdims + reference_channels;
    int divisor = 1;
    for(int sdim = n_sdims - 1; sdim >= 0; sdim--){
        positions[num_dims * idx + sdim] = ((idx / divisor) % sdims[sdim]) / spatial_std;
        divisor *= sdims[sdim];
    }

    for(int channel = 0; channel < reference_channels; channel++){
        positions[num_dims * idx + n_sdims + channel] = reference[idx * reference_channels + channel] / feature_std;
    }
}

#endif //PERMUTOHEDRAL_CU