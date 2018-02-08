#ifndef PERMUTOHEDRAL_CU
#define PERMUTOHEDRAL_CU

#define BLOCK_SIZE 256
#define DEBUG 0


#include <cstdio>
//#include "cuda_code_indexing.h"
#include "cuda_runtime.h"
#include <stdexcept>

template<int pd, int vd>class HashTableGPU{
public:
    int capacity;
    float * values;
    signed short * keys;
    int * entries;
    bool original; //is this the original table or a copy?

    HashTableGPU(int capacity_): capacity(capacity_), values(nullptr), keys(nullptr), entries(nullptr), original(true){

        cudaMalloc((void**)&values, capacity*vd*sizeof(float));
        cudaMemset((void *)values, 0, capacity*vd*sizeof(float));

        cudaMalloc((void **)&entries, capacity*2*sizeof(int));
        cudaMemset((void *)entries, -1, capacity*2*sizeof(int));

        cudaMalloc((void **)&keys, capacity*pd*sizeof(signed short));
        cudaMemset((void *)keys, 0, capacity*pd*sizeof(signed short));
    }

    HashTableGPU(const HashTableGPU& table):capacity(table.capacity), values(table.values), keys(table.keys), entries(table.entries), original(false){}

    ~HashTableGPU(){
        // only free if it is the original table
        if(original){
            cudaFree(values);
            cudaFree(entries);
            cudaFree(keys);
        }
    }

    void resetHashTable() {
        cudaMemset((void*)values, 0, capacity*vd*sizeof(float));
    }

    __device__ int modHash(unsigned int n){
        return(n % (2 * capacity));
    }

    __device__ unsigned int hash(signed short *key) {
        unsigned int k = 0;
        for (int i = 0; i < pd; i++) {
            k += key[i];
            k = k * 2531011;
        }
        return k;
    }

    __device__ int insert(signed short *key, unsigned int slot) {
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
                    keys[slot*pd+i] = key[i];
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

    __device__ int retrieve(signed short *key) {

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



struct MatrixEntry {
    int index;
    float weight;
};

template<int pd, int vd>
__global__ static void createLattice(const int n,
                                     const float *positions,
                                     const float *scaleFactor,
                                     const int * canonical,
                                     MatrixEntry *matrix,
                                     HashTableGPU<pd, vd> table) {

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n)
        return;


    float elevated[pd + 1];
    const float *position = positions + idx * pd;
    int rem0[pd + 1];
    int rank[pd + 1];


    float sm = 0;
    for (int i = pd; i > 0; i--) {
        float cf = position[i - 1] * scaleFactor[i - 1];
        elevated[i] = sm - i * cf;
        sm += cf;
    }
    elevated[0] = sm;


    // Find the closest 0-colored simplex through rounding
    // greedily search for the closest zero-colored lattice point
    signed short sum = 0;
    for (int i = 0; i <= pd; i++) {
        float v = elevated[i] * (1.0f / (pd + 1));
        float up = ceilf(v) * (pd + 1);
        float down = floorf(v) * (pd + 1);
        if (up - elevated[i] < elevated[i] - down) {
            rem0[i] = (signed short) up;
        } else {
            rem0[i] = (signed short) down;
        }
        sum += rem0[i];
    }
    sum /= pd + 1;

    /*
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
    */

    // sort differential to find the permutation between this simplex and the canonical one
    for (int i = 0; i <= pd; i++) {
        rank[i] = 0;
        for (int j = 0; j <= pd; j++) {
            if (elevated[i] - rem0[i] < elevated[j] - rem0[j] || (elevated[i] - rem0[i] == elevated[j] - rem0[j] && i > j)) {
                rank[i]++;
            }
        }
    }

    if (sum > 0) { // sum too large, need to bring down the ones with the smallest differential
        for (int i = 0; i <= pd; i++) {
            if (rank[i] >= pd + 1 - sum) {
                rem0[i] -= pd + 1;
                rank[i] += sum - (pd + 1);
            } else {
                rank[i] += sum;
            }
        }
    } else if (sum < 0) { // sum too small, need to bring up the ones with largest differential
        for (int i = 0; i <= pd; i++) {
            if (rank[i] < -sum) {
                rem0[i] += pd + 1;
                rank[i] += (pd + 1) + sum;
            } else {
                rank[i] += sum;
            }
        }
    }

    float barycentric[pd + 2]{0};

    // turn delta into barycentric coords

    for (int i = 0; i <= pd; i++) {
        float delta = (elevated[i] - rem0[i]) * (1.0f / (pd + 1));
        barycentric[pd - rank[i]] += delta;
        barycentric[pd + 1 - rank[i]] -= delta;
    }
    barycentric[0] += 1.0f + barycentric[pd + 1];


    short key[pd];
    for (int remainder = 0; remainder <= pd; remainder++) {
        // Compute the location of the lattice point explicitly (all but
        // the last coordinate - it's redundant because they sum to zero)

        /*for (int i = 0; i < pd; i++)
            key[i] = static_cast<short>(rem0[i] + canonical[remainder * (pd + 1) + rank[i]]);*/
        for (int i = 0; i < pd; i++) {
            key[i] = static_cast<short>(rem0[i] + remainder);
            if (rank[i] > pd - remainder)
                key[i] -= (pd + 1);
        }

        MatrixEntry r;
        unsigned int slot = static_cast<unsigned int>(idx * (pd + 1) + remainder);
        r.index = table.insert(key, slot);
        r.weight = barycentric[remainder];
        matrix[idx * (pd + 1) + remainder] = r;
    }
}

template<int pd, int vd>
__global__ static void cleanHashTable(int n, HashTableGPU<pd, vd> table) {

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

template<int pd, int vd>
__global__ static void splatCache(const int n, const float *values, MatrixEntry *matrix, HashTableGPU<pd, vd> table) {

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int threadId = threadIdx.x;
    const int color = blockIdx.y;
    const bool outOfBounds = (idx >= n);

    __shared__ int sharedOffsets[BLOCK_SIZE];
    __shared__ float sharedValues[BLOCK_SIZE * vd];
    int myOffset = -1;
    float *myValue = sharedValues + threadId * vd;

    if (!outOfBounds) {

        float *value = const_cast<float *>(values + idx * (vd - 1));

        MatrixEntry r = matrix[idx * (pd + 1) + color];

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
    float *val = table.values + myOffset;
    for (int j = 0; j < vd; j++) {
        atomicAdd(val + j, myValue[j]);
    }
}

template<int pd, int vd>
__global__ static void blur(int n, float *newValues, MatrixEntry *matrix, int color, HashTableGPU<pd, vd> table) {

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

    //in case neighbours don't exist (lattice edges) offNp and offNm are -1
    float zeros[vd]{0};
    float *valNp = zeros;
    float *valNm = zeros;
    if(offNp >= 0)
        valNp = table.values + vd * offNp;
    if(offNm >= 0)
        valNm = table.values + vd * offNm;

    float *valMe = table.values + vd * idx;
    float *valOut = newValues + vd * idx;

    for (int i = 0; i < vd; i++)
        valOut[i] = 0.25f * valNp[i] + 0.5f * valMe[i] + 0.25f * valNm[i];

}

template<int pd, int vd>
__global__ static void slice(const int n, float *values, MatrixEntry *matrix, HashTableGPU<pd, vd> table) {

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n)
        return;

    float value[vd-1]{0};
    float weight = 0;

    for (int i = 0; i <= pd; i++) {
        MatrixEntry r = matrix[idx * (pd + 1) + i];
        float *val = table.values + r.index * vd;
        for (int j = 0; j < vd - 1; j++) {
            value[j] += r.weight * val[j];
        }
        weight += r.weight * val[vd - 1];
    }

    weight = 1.0f / weight;
    for (int j = 0; j < vd - 1; j++)
        values[idx * (vd - 1) + j] = value[j] * weight;

}


template<int pd, int vd>class PermutohedralLatticeGPU{
public:
    int n; //number of pixels/voxels etc..
    int * canonical;
    float * scaleFactor;
    MatrixEntry* matrix;
    HashTableGPU<pd, vd> hashTable;

    float * newValues; // auxiliary array for blur stage
    //number of blocks and threads per block
    //dim3 blocks;
    //dim3 blockSize;
    //dim3 cleanBlocks;
    //unsigned int cleanBlockSize;

    void init_canonical(){
        int hostCanonical[(pd + 1) * (pd + 1)];
        //auto canonical = new int[(pd + 1) * (pd + 1)];
        // compute the coordinates of the canonical simplex, in which
        // the difference between a contained point and the zero
        // remainder vertex is always in ascending order. (See pg.4 of paper.)
        for (int i = 0; i <= pd; i++) {
            for (int j = 0; j <= pd - i; j++)
                hostCanonical[i * (pd + 1) + j] = i;
            for (int j = pd - i + 1; j <= pd; j++)
                hostCanonical[i * (pd + 1) + j] = i - (pd + 1);
        }
        size_t size =  ((pd + 1) * (pd + 1))*sizeof(int);
        cudaMalloc((void**)&(canonical), size);
        cudaMemcpy(canonical, hostCanonical, size, cudaMemcpyHostToDevice);
    }


    void init_scaleFactor(){
        float hostScaleFactor[pd];
        float inv_std_dev = (pd + 1) * sqrtf(2.0f / 3);
        for (int i = 0; i < pd; i++) {
            hostScaleFactor[i] = 1.0f / (sqrtf((float) (i + 1) * (i + 2))) * inv_std_dev;
        }
        size_t size =  pd*sizeof(float);
        cudaMalloc((void**)&(scaleFactor), size);
        cudaMemcpy(scaleFactor, hostScaleFactor, size, cudaMemcpyHostToDevice);
    }

    void init_matrix(){
        cudaMalloc((void**)&(matrix), n * (pd + 1) * sizeof(MatrixEntry));
    }

    void init_newValues(){
        cudaMalloc((void **) &(newValues), n * (pd + 1) * vd * sizeof(float));
        cudaMemset((void *) newValues, 0, n * (pd + 1) * vd * sizeof(float));
    }


    PermutohedralLatticeGPU(int n_): n(n_), canonical(nullptr), scaleFactor(nullptr), matrix(nullptr), newValues(nullptr), hashTable(HashTableGPU<pd, vd>(n * (pd + 1))){

        if (n >= 65535 * BLOCK_SIZE) {
            printf("Not enough GPU memory (on x axis, you can change the code to use other grid dims)\n");
            //this should crash the program
        }

        // initialize device memory
        init_canonical();
        init_scaleFactor();
        init_matrix();
        init_newValues();
        //
        //blocks = dim3((n - 1) / BLOCK_SIZE + 1, 1, 1);
        //blockSize = dim3(BLOCK_SIZE, 1, 1);
        //cleanBlockSize = 32;
        //cleanBlocks = dim3((n - 1) / cleanBlockSize + 1, 2 * (pd + 1), 1);
    }

    ~PermutohedralLatticeGPU(){
        cudaFree(canonical);
        cudaFree(scaleFactor);
        cudaFree(matrix);
        cudaFree(newValues);
    }

#ifndef DEBUG
    // values and position must already be device pointers
    void filter(float * inputs, float*  positions){
        createLattice<pd, vd> <<<blocks, blockSize>>>(n, positions, scaleFactor, canonical, matrix, hashTable);
        cleanHashTable<pd, vd> <<<cleanBlocks, cleanBlockSize>>>(2 * n * (pd + 1), hashTable);
        blocks.y = pd + 1;
        splatCache<pd, vd><<<blocks, blockSize>>>(n, inputs, matrix, hashTable);
        for (int remainder = 0; remainder <= pd; remainder++) {
            blur<pd, vd><<<cleanBlocks, cleanBlockSize>>>(n * (pd + 1), newValues, matrix, remainder, hashTable);
            std::swap(hashTable.values, newValues);
        }
        blockSize.y = 1;
        slice<pd, vd><<< blocks, blockSize>>>(n, inputs, matrix, hashTable);
    }
#else
    // values and position must already be device pointers
    void filter(float* output, const float* inputs, const float*  positions){

        dim3 blocks((n - 1) / BLOCK_SIZE + 1, 1, 1);
        dim3 blockSize(BLOCK_SIZE, 1, 1);
        int cleanBlockSize = 32;
        dim3 cleanBlocks((n - 1) / cleanBlockSize + 1, 2 * (pd + 1), 1);

        createLattice<pd, vd> <<<blocks, blockSize>>>(n, positions, scaleFactor, canonical, matrix, hashTable);
        printf("Create Lattice: %s\n", cudaGetErrorString(cudaGetLastError()));

        cleanHashTable<pd, vd> <<<cleanBlocks, cleanBlockSize>>>(2 * n * (pd + 1), hashTable);
        printf("Clean Hash Table: %s\n", cudaGetErrorString(cudaGetLastError()));

        blocks.y = pd + 1;
        splatCache<pd, vd><<<blocks, blockSize>>>(n, inputs, matrix, hashTable);
        printf("Splat: %s\n", cudaGetErrorString(cudaGetLastError()));

        for (int remainder = 0; remainder <= pd; remainder++) {
            blur<pd, vd><<<cleanBlocks, cleanBlockSize>>>(n * (pd + 1), newValues, matrix, remainder, hashTable);
            printf("Blur %d: %s\n", remainder, cudaGetErrorString(cudaGetLastError()));
            std::swap(hashTable.values, newValues);
        }
        blockSize.y = 1;
        slice<pd, vd><<< blocks, blockSize>>>(n, output, matrix, hashTable);
        printf("Slice: %s\n", cudaGetErrorString(cudaGetLastError()));
    }

#endif
};


template<int pd, int vd>
void filter_(float* output, const float *input, const float *positions, int n) {
    auto lattice = PermutohedralLatticeGPU<pd, vd>(n);
    lattice.filter(output, input, positions);

}

__global__ static void compute_bilateral_kernel(const float * reference,
                                                float * positions,
                                                int num_super_pixels,
                                                int reference_channels,
                                                int n_sdims,
                                                const int *sdims,
                                                float theta_alpha,
                                                float theta_beta){

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_super_pixels)
        return;

    int num_dims = n_sdims + reference_channels;
    int divisor = 1;
    for(int sdim = n_sdims - 1; sdim >= 0; sdim--){
        positions[num_dims * idx + sdim] = ((idx / divisor) % sdims[sdim]) / theta_alpha;
        divisor *= sdims[sdim];
    }

    for(int channel = 0; channel < reference_channels; channel++){
        positions[num_dims * idx + n_sdims + channel] = reference[idx * reference_channels + channel] / theta_beta;
    }

}


#ifdef LIBRARY
extern "C++"
#ifdef WIN32
__declspec(dllexport)
#endif
#endif


//input and positions should be device pointers by this point
void lattice_filter_gpu(float * output, const float *input, const float *positions, int pd, int vd, int n) {
    //vd = image_channels + 1
    if(pd == 5 && vd == 4)
        filter_<5, 4>(output, input, positions, n);
    else
        return;
        //throw std::invalid_argument( "filter not implemented" ); //LOG(FATAL);
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
    compute_bilateral_kernel<<<blocks, blockSize>>>(reference, positions, num_super_pixels, n_reference_channels, n_spatial_dims, spatial_dims, theta_alpha, theta_beta);
};


#endif //PERMUTOHEDRAL_CU


