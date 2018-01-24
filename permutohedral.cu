
#define BLOCK_SIZE 256

#include <cuda_runtime.h>
#include <stdio.h>
#include <string>
#include "cuda_memory.h"
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
__global__ static void createMatrix(const int n, const float *positions, const float *scaleFactor, MatrixEntry *matrix, HashTable<pd, vd> table) {

	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx >= n)
		return;


	float elevated[pd+1];
	const float *position = positions + idx*pd;
	int rem0[pd+1];
	int rank[pd+1];


	float sm = 0;
	for (int i = pd; i > 0; i--) {
		float cf = position[i - 1] * scaleFactor[i - 1];
		elevated[i] = sm - i * cf;
		sm += cf;
	}
	elevated[0] = sm;


	// find the closest zero-colored lattice point

	// greedily search for the closest zero-colored lattice point
	signed short sum = 0;
	for (int i = 0; i <= pd; i++) {
		float v = elevated[i]*(1.0f/(pd+1));
		float up = ceilf(v) * (pd+1);
		float down = floorf(v) * (pd+1);
		if (up - elevated[i] < elevated[i] - down) {
			rem0[i] = (signed short)up;
		} else {
			rem0[i] = (signed short)down;
		}
		sum += rem0[i];
	}
	sum /= pd+1;

	// sort differential to find the permutation between this simplex and the canonical one
	for (int i = 0; i <= pd; i++) {
		rank[i] = 0;
		for (int j = 0; j <= pd; j++) {
			if (elevated[i] - rem0[i] < elevated[j] - rem0[j] ||
					(elevated[i] - rem0[i] == elevated[j] - rem0[j] && i > j)) {
				rank[i]++;
			}
		}
	}

	if (sum > 0) { // sum too large, need to bring down the ones with the smallest differential
		for (int i = 0; i <= pd; i++) {
			if (rank[i] >= pd + 1 - sum) {
				rem0[i] -= pd+1;
				rank[i] += sum - (pd+1);
			} else {
				rank[i] += sum;
			}
		}
	} else if (sum < 0) { // sum too small, need to bring up the ones with largest differential
		for (int i = 0; i <= pd; i++) {
			if (rank[i] < -sum) {
				rem0[i] += pd+1;
				rank[i] += (pd+1) + sum;
			} else {
				rank[i] += sum;
			}
		}
	}

	float barycentric[pd+2]{0};

	// turn delta into barycentric coords

	for (int i = 0; i <= pd; i++) {
		float delta = (elevated[i] - rem0[i]) * (1.0f/(pd+1));
		barycentric[pd-rank[i]] += delta;
		barycentric[pd+1-rank[i]] -= delta;
	}
	barycentric[0] += 1.0f + barycentric[pd+1];

	short key[pd];
	for (int color = 0; color <= pd; color++) {
		// Compute the location of the lattice point explicitly (all but
		// the last coordinate - it's redundant because they sum to zero)

		for (int i = 0; i < pd; i++) {
			key[i] = rem0[i] + color;
			if (rank[i] > pd-color)
				key[i] -= (pd+1);
		}

		MatrixEntry r;
		r.index = table.insert(key, idx*(pd+1)+color);
		r.weight = barycentric[color];
		matrix[idx*(pd+1) + color] = r;
	}

}

template<int pd, int vd>
__global__ static void cleanHashTable(int n, MatrixEntry *matrix, HashTable<pd, vd> table) {

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
		*e = table.retrieve(table.keys + *e*pd);
	}
}

template<int pd, int vd>
__global__ static void splatCache(const int n, float *values, MatrixEntry *matrix, HashTable<pd, vd> table) {

	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int threadId = threadIdx.x;
	const int color = blockIdx.y;
	const bool outOfBounds = (idx>=n);

	__shared__ int sharedOffsets[BLOCK_SIZE];
	__shared__ float sharedValues[BLOCK_SIZE*(vd+1)];
	int myOffset = -1;
	float *myValue = sharedValues + threadId*(vd+1);

	if (!outOfBounds) {

		float *value = values + idx*vd;

		MatrixEntry r = matrix[idx*(pd+1)+color];

		// convert the matrix entry from a pointer into the entries array to a pointer into the keys/values array
		matrix[idx*(pd+1)+color].index = r.index = table.entries[r.index];
		// record the offset into the keys/values array in shared space
		myOffset = sharedOffsets[threadId] = r.index*(vd+1);

		for (int j = 0; j < vd; j++) {
			myValue[j] = value[j]*r.weight;
		}
		myValue[vd] = r.weight;

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
				for (int j = 0; j <= vd; j++) {
					sharedValues[threadId*(vd+1) + j] += sharedValues[i*(vd+1) + j];
				}
			}
		}
	}

	// only the threads with something to write to main memory are still going
	float *val = table.values + myOffset;
	for (int j = 0; j <= vd; j++) {
		atomicAdd(val+j, myValue[j]);
	}
}

template<int pd, int vd>
__global__ static void blur(int n, float *newValues, MatrixEntry *matrix, int color, HashTable<pd, vd> table) {

	const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;
	if (idx >= n)
		return;

	// Check if I'm valid
	if (matrix[idx].index != idx)
		return;


	// find my key and the keys of my neighbors
	short myKey[pd+1];
	short np[pd+1];
	short nm[pd+1];


	for (int i = 0; i < pd; i++) {
		myKey[i] = table.keys[idx*pd+i];
		np[i] = myKey[i]+1;
		nm[i] = myKey[i]-1;
	}
	np[color] -= pd+1;
	nm[color] += pd+1;


	int offNp = table.retrieve(np);
	int offNm = table.retrieve(nm);

	float *valMe = table.values + (vd+1)*idx;
	float *valNp = table.values + (vd+1)*offNp;
	float *valNm = table.values + (vd+1)*offNm;
	float *valOut = newValues + (vd+1)*idx;

	for (int i = 0; i <= vd; i++)
		valOut[i] = 0.25 * valNp[i] + 0.5* valMe[i] + 0.25*valNm[i];
}

template<int pd, int vd>
__global__ static void slice(const int n, float *values, MatrixEntry *matrix, HashTable<pd, vd> table) {

	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx >= n)
		return;

	float value[vd]{0};
	float weight = 0;

	for (int i = 0; i <= pd; i++) {
		MatrixEntry r = matrix[idx*(pd+1) + i];
		float *val = table.values + r.index*(vd+1);
		for (int j = 0; j < vd; j++) {
			value[j] += r.weight*val[j];
		}
		weight += r.weight*val[vd];
	}

	weight = 1.0f/weight;
	for (int j = 0; j < vd; j++)
		values[idx*vd + j] = value[j]*weight;

}


#ifdef LIBRARY
extern "C++"
#ifdef WIN32
__declspec(dllexport)
#endif
#endif

template<int vd, int pd>
void filter(float *im, float *ref, int n) {

	timeval t[9];
	gettimeofday(t+0, NULL);

	if(n >= 65535 * BLOCK_SIZE){
		printf("Not enough GPU memory (on x axis, you can change the code to use other grid dims)\n");
		return;
	}

	MirroredArray<float> scaleFactor(pd);

	float inv_std_dev = (pd + 1) * sqrtf(2.0f / 3);
	for (int i = 0; i < pd; i++) {
		scaleFactor.host[i] = 1.0f / (sqrtf((float) (i + 1) * (i + 2))) * inv_std_dev;
	}
	scaleFactor.hostToDevice();


	MirroredArray<float> positions(ref, n*pd);
	MirroredArray<MatrixEntry> matrix(n*(pd+1));
	MirroredArray<float> values(im, n*vd);

	float *newValues;
	allocateCudaMemory((void**)&(newValues), n*(pd+1)*(vd+1)*sizeof(float));
	cudaMemset((void *)newValues, 0, n*(pd+1)*(vd+1)*sizeof(float));

	HashTable<pd, vd> table(n*(pd+1));

	dim3 blocks((n-1)/BLOCK_SIZE+1, 1, 1);
	dim3 blockSize(BLOCK_SIZE, 1, 1);

	gettimeofday(t+1, NULL);


	createMatrix<pd, vd><<<blocks, blockSize>>>(n, positions.device, scaleFactor.device, matrix.device, table);
	printf("Create Matrix %s\n", cudaGetErrorString(cudaGetLastError()));
	gettimeofday(t+2, NULL);

	// fix duplicate hash table entries
	int cleanBlockSize = 32;
	dim3 cleanBlocks((n-1)/cleanBlockSize+1, 2*(pd+1), 1);
	cleanHashTable<pd, vd><<<cleanBlocks, cleanBlockSize>>>(2*n*(pd+1), matrix.device, table);
	printf("Clean Hash Table %s\n", cudaGetErrorString(cudaGetLastError()));
	gettimeofday(t+3, NULL);

	// splat splits by color, so extend the y coordinate to our blocks to represent that
	blocks.y = pd+1;
	splatCache<pd, vd><<<blocks, blockSize>>>(n, values.device, matrix.device, table);
	gettimeofday(t+4, NULL);

	for (int color = 0; color <= pd; color++) {
		blur<pd, vd><<<cleanBlocks, cleanBlockSize>>>(n*(pd+1), newValues, matrix.device, color, table);
		printf("Blur 1.%d %s\n", color, cudaGetErrorString(cudaGetLastError()));
		std::swap(table.values, newValues);
	}
	gettimeofday(t+5, NULL);


	blockSize.y = 1;
	slice<pd, vd><<<blocks, blockSize>>>(n, values.device, matrix.device, table);
	printf("Slice %s\n", cudaGetErrorString(cudaGetLastError()));
	gettimeofday(t+6, NULL);

	values.deviceToHost();
	cudaFree(newValues);
	gettimeofday(t+7, NULL);

	double total = (t[7].tv_sec - t[0].tv_sec)*1000.0 + (t[7].tv_usec - t[0].tv_usec)/1000.0;
	printf("Total time: %3.3f ms\n", total);
	printf("%s: %3.3f ms\n", "Init", (t[1].tv_sec - t[0].tv_sec)*1000.0 + (t[1].tv_usec - t[0].tv_usec)/1000.0);
	printf("%s: %3.3f ms\n", "Create", (t[2].tv_sec - t[1].tv_sec)*1000.0 + (t[2].tv_usec - t[1].tv_usec)/1000.0);
	printf("%s: %3.3f ms\n", "Clean", (t[3].tv_sec - t[2].tv_sec)*1000.0 + (t[3].tv_usec - t[2].tv_usec)/1000.0);
	printf("%s: %3.3f ms\n", "Splat", (t[4].tv_sec - t[3].tv_sec)*1000.0 + (t[4].tv_usec - t[3].tv_usec)/1000.0);
	printf("%s: %3.3f ms\n", "Blur", (t[5].tv_sec - t[4].tv_sec)*1000.0 + (t[5].tv_usec - t[4].tv_usec)/1000.0);
	printf("%s: %3.3f ms\n", "Slice", (t[6].tv_sec - t[4].tv_sec)*1000.0 + (t[6].tv_usec - t[5].tv_usec)/1000.0);
	printf("%s: %3.3f ms\n", "Free", (t[7].tv_sec - t[6].tv_sec)*1000.0 + (t[7].tv_usec - t[6].tv_usec)/1000.0);

	printf("Total GPU memory usage: %u bytes\n", (unsigned int)GPU_MEMORY_ALLOCATION);

}





