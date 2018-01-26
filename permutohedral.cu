#ifndef PERMUTOHEDRAL_CU
#define PERMUTOHEDRAL_CU

#define BLOCK_SIZE 256

#include <cuda_runtime.h>
#include <stdio.h>
#include <string>
#include "cuda_code_indexing.h"

#include <sys/time.h>

#include "MirroredArray.h"
#include "hash_table.cu"
#include "cuda_code_indexing.h"

struct MatrixEntry {
    int index;
    float weight;
};

template<int pd, int vd>
__global__ static void createMatrix(const int n, const float *positions, const float *scaleFactor, const float * canonical, MatrixEntry *matrix,
                                    HashTable_1<pd, vd> table) {

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
            key[i] = rem0[i] + remainder;
            if (rank[i] > pd - remainder)
                key[i] -= (pd + 1);
        }

        MatrixEntry r;
        r.index = table.insert(key, idx * (pd + 1) + remainder);
        r.weight = barycentric[remainder];
        matrix[idx * (pd + 1) + remainder] = r;
    }
}

template<int pd, int vd>
__global__ static void cleanHashTable(int n, HashTable_1<pd, vd> table) {

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
__global__ static void splatCache(const int n, float *values, MatrixEntry *matrix, HashTable_1<pd, vd> table) {

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int threadId = threadIdx.x;
    const int color = blockIdx.y;
    const bool outOfBounds = (idx >= n);

    __shared__ int sharedOffsets[BLOCK_SIZE];
    __shared__ float sharedValues[BLOCK_SIZE * vd];
    int myOffset = -1;
    float *myValue = sharedValues + threadId * vd;

    if (!outOfBounds) {

        float *value = values + idx * (vd - 1);

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
__global__ static void blur(int n, float *newValues, MatrixEntry *matrix, int color, HashTable_1<pd, vd> table) {

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

    float *valMe = table.values + vd * idx;
    float *valNp = table.values + vd * offNp;
    float *valNm = table.values + vd * offNm;
    float *valOut = newValues + vd * idx;

    for (int i = 0; i < vd; i++)
        valOut[i] = 0.25 * valNp[i] + 0.5 * valMe[i] + 0.25 * valNm[i];
}

template<int pd, int vd>
__global__ static void slice(const int n, float *values, MatrixEntry *matrix, HashTable_1<pd, vd> table) {

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


template<int pd, int vd>class PermutohedralLattice{
    int * canonical;
    float * scaleFactor;
    HashTable_1<pd, vd> hashTable;
    MatrixEntry* matrix;
    float * ref;
    float * values;


    PermutohedralLattice(): canonical(nullptr), scaleFactor(nullptr), hashTable(HashTable_1<pd, vd>(n * (pd + 1))){
        if (n >= 65535 * BLOCK_SIZE) {
            printf("Not enough GPU memory (on x axis, you can change the code to use other grid dims)\n");
            return;
        }

    }

    void init_canonical(){
        int localCanonical[(pd + 1) * (pd + 1)];
        //auto canonical = new int[(pd + 1) * (pd + 1)];
        // compute the coordinates of the canonical simplex, in which
        // the difference between a contained point and the zero
        // remainder vertex is always in ascending order. (See pg.4 of paper.)
        for (int i = 0; i <= pd; i++) {
            for (int j = 0; j <= pd - i; j++)
                localCanonical[i * (pd + 1) + j] = i;
            for (int j = pd - i + 1; j <= pd; j++)
                localCanonical[i * (pd + 1) + j] = i - (pd + 1);
        }
        cudaMalloc((void**)&(canonical), ((pd + 1) * (pd + 1))*sizeof(int));
    }

    void init_scaleFactor(){
        float hostScaleFactor[pd];
        float inv_std_dev = (pd + 1) * sqrtf(2.0f / 3);
        for (int i = 0; i < pd; i++) {
            hostScaleFactor[i] = 1.0f / (sqrtf((float) (i + 1) * (i + 2))) * inv_std_dev;
        }
        cudaMalloc((void**)&(scaleFactor), pd*sizeof(float));
    }

    void init_matrix(){
        cudaMalloc((void**)&(matrix), n * (pd + 1) * sizeof(MatrixEntry));
    }

    void filter(){
       return;
    }
};


template<int pd, int vd>
void filter_(float *im, float *ref, int n) {

    timeval t[9];
    gettimeofday(t + 0, NULL);

    if (n >= 65535 * BLOCK_SIZE) {
        printf("Not enough GPU memory (on x axis, you can change the code to use other grid dims)\n");
        return;
    }




    MirroredArray<float> positions(ref, static_cast<size_t>(n * pd));
    MirroredArray<MatrixEntry> matrix(static_cast<size_t>(n * (pd + 1)));
    MirroredArray<float> values(im, static_cast<size_t>(n * (vd-1)));

    float *newValues;
    cudaMalloc((void **) &(newValues), n * (pd + 1) * vd * sizeof(float));
    cudaMemset((void *) newValues, 0, n * (pd + 1) * vd * sizeof(float));

    HashTable_1<pd, vd> table(n * (pd + 1));

    dim3 blocks((n - 1) / BLOCK_SIZE + 1, 1, 1);
    dim3 blockSize(BLOCK_SIZE, 1, 1);

    gettimeofday(t + 1, NULL);


    createMatrix<pd, vd><<<blocks, blockSize>>>(n, positions.device, scaleFactor.device, canonical.device, matrix.device, table);
    gettimeofday(t + 2, NULL);

    // fix duplicate hash table entries
    int cleanBlockSize = 32;
    dim3 cleanBlocks((n - 1) / cleanBlockSize + 1, 2 * (pd + 1), 1);
    cleanHashTable<pd, vd> <<<cleanBlocks, cleanBlockSize>>>(2 * n * (pd + 1), table);
    gettimeofday(t + 3, NULL);

    // splat splits by color, so extend the y coordinate to our blocks to represent that
    blocks.y = pd + 1;
    splatCache<pd, vd><<<blocks, blockSize>>>(n, values.device, matrix.device, table);
    gettimeofday(t + 4, NULL);

    for (int color = 0; color <= pd; color++) {
        blur<pd, vd><<<cleanBlocks, cleanBlockSize>>>(n * (pd + 1), newValues, matrix.device, color, table);
        std::swap(table.values, newValues);
    }
    gettimeofday(t + 5, NULL);


    blockSize.y = 1;
    slice<pd, vd><<< blocks, blockSize>>>(n, values.device, matrix.device, table);
    gettimeofday(t + 6, NULL);

    values.deviceToHost();
    cudaFree(newValues);
    gettimeofday(t + 7, NULL);

    double total = (t[7].tv_sec - t[0].tv_sec) * 1000.0 + (t[7].tv_usec - t[0].tv_usec) / 1000.0;
    printf("Total time: %3.3f ms\n", total);
    printf("%s: %3.3f ms\n", "Init", (t[1].tv_sec - t[0].tv_sec) * 1000.0 + (t[1].tv_usec - t[0].tv_usec) / 1000.0);
    printf("%s: %3.3f ms\n", "Create", (t[2].tv_sec - t[1].tv_sec) * 1000.0 + (t[2].tv_usec - t[1].tv_usec) / 1000.0);
    printf("%s: %3.3f ms\n", "Clean", (t[3].tv_sec - t[2].tv_sec) * 1000.0 + (t[3].tv_usec - t[2].tv_usec) / 1000.0);
    printf("%s: %3.3f ms\n", "Splat", (t[4].tv_sec - t[3].tv_sec) * 1000.0 + (t[4].tv_usec - t[3].tv_usec) / 1000.0);
    printf("%s: %3.3f ms\n", "Blur", (t[5].tv_sec - t[4].tv_sec) * 1000.0 + (t[5].tv_usec - t[4].tv_usec) / 1000.0);
    printf("%s: %3.3f ms\n", "Slice", (t[6].tv_sec - t[4].tv_sec) * 1000.0 + (t[6].tv_usec - t[5].tv_usec) / 1000.0);
    printf("%s: %3.3f ms\n", "Free", (t[7].tv_sec - t[6].tv_sec) * 1000.0 + (t[7].tv_usec - t[6].tv_usec) / 1000.0);


}

#ifdef LIBRARY
extern "C++"
#ifdef WIN32
__declspec(dllexport)
#endif
#endif

void filter(float *im, float *ref, int n) {
    //vd = image_channels + 1
    filter_<5, 4>(im, ref, n);
}


#endif //PERMUTOHEDRAL_CU


