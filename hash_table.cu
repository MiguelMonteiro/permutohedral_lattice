#ifndef HASH_TABLE_CU_H
#define HASH_TABLE_CU_H

#include "cuda_code_indexing.h"


template<int pd, int vd>class HashTable{
public:
	int capacity;
	float * values;
	signed short * keys;
	int * entries;
	bool original; //is this the original table or a copy?

	HashTable(int capacity_): capacity(capacity_), values(nullptr), keys(nullptr), entries(nullptr), original(true){

        cudaMalloc((void**)&values, capacity*(vd+1)*sizeof(float));
		cudaMemset((void *)values, 0, capacity*(vd+1)*sizeof(float));

        cudaMalloc((void **)&entries, capacity*2*sizeof(int));
		cudaMemset((void *)entries, -1, capacity*2*sizeof(int));

        cudaMalloc((void **)&keys, capacity*pd*sizeof(signed short));
		cudaMemset((void *)keys, 0, capacity*pd*sizeof(signed short));
	}

	HashTable(const HashTable& table):capacity(table.capacity), values(table.values), keys(table.keys), entries(table.entries), original(false){}

	~HashTable(){
		// only free if it is the original table
		if(original){
			cudaFree(values);
			cudaFree(entries);
			cudaFree(keys);
		}
	}

	void resetHashTable() {
		cudaMemset((void*)values, 0, capacity*(vd+1)*sizeof(float));
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

#endif //HASH_TABLE_CU_H



