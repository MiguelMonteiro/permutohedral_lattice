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

#ifndef PERMUTOHEDRAL_LATTICE_CPU_H
#define PERMUTOHEDRAL_LATTICE_CPU_H

#include <cstring>
#include <memory>

/***************************************************************/
/* Hash table implementation for permutohedral lattice
 *
 * The lattice points are stored sparsely using a hash table.
 * The key for each point is its spatial location in the (pd+1)-
 * dimensional space.
 */
/***************************************************************/
template <typename T>class HashTableCPU {
public:
    short *keys;
    T *values;
    int *entries;
    size_t capacity, filled;
    int pd, vd;

    /* Hash function used in this implementation. A simple base conversion. */
    size_t hash(const short *key) {
        size_t k = 0;
        for (int i = 0; i < pd; i++) {
            k += key[i];
            k *= 2531011;
        }
        return k;
    }

    /* Returns the index into the hash table for a given key.
    *     key: a pointer to the position vector.
    *       h: hash of the position vector.
    *  create: a flag specifying whether an entry should be created,
    *          should an entry with the given key not found.
    */
    int lookupOffset(const short *key, size_t h, bool create = true) {

        // Double hash table size if necessary
        if (filled >= (capacity / 2) - 1) { grow(); }

        // Find the entry with the given key
        while (true) {
            int* e = entries + h;
            // check if the cell is empty
            if (*e == -1) {
                if (!create)
                    return -1; // Return not found.
                // need to create an entry. Store the given key.
                for (int i = 0; i < pd; i++)
                    keys[filled * pd + i] = key[i];
                *e = static_cast<int>(filled);
                filled++;
                return *e * vd;
            }

            // check if the cell has a matching key
            bool match = true;
            for (int i = 0; i < pd && match; i++)
                match = keys[*e*pd + i] == key[i];
            if (match)
                return *e * vd;

            // increment the bucket with wraparound
            h++;
            if (h == capacity)
                h = 0;
        }
    }

    /* Grows the size of the hash table */
    void grow() {
        printf("Resizing hash table\n");

        size_t oldCapacity = capacity;
        capacity *= 2;

        // Migrate the value vectors.
        auto newValues = new T[vd * capacity / 2]{0};
        std::memcpy(newValues, values, sizeof(T) * vd * filled);
        delete[] values;
        values = newValues;

        // Migrate the key vectors.
        auto newKeys = new short[pd * capacity / 2];
        std::memcpy(newKeys, keys, sizeof(short) * pd * filled);
        delete[] keys;
        keys = newKeys;

        auto newEntries = new int[capacity];
        memset(newEntries, -1, capacity*sizeof(int));

        // Migrate the table of indices.
        for (size_t i = 0; i < oldCapacity; i++) {
            if (entries[i] == -1)
                continue;
            size_t h = hash(keys + entries[i] * pd) % capacity;
            while (newEntries[h] != -1) {
                h++;
                if (h == capacity) h = 0;
            }
            newEntries[h] = entries[i];
        }
        delete[] entries;
        entries = newEntries;
    }

public:
    /* Constructor
     *  pd_: the dimensionality of the position vectors on the hyperplane.
     *  vd_: the dimensionality of the value vectors
     */
    HashTableCPU(int pd_, int vd_) : pd(pd_), vd(vd_) {
        capacity = 1 << 15;
        filled = 0;
        entries = new int[capacity];
        memset(entries, -1, capacity*sizeof(int));
        keys = new short[pd * capacity / 2];
        values = new T[vd * capacity / 2]{0};
    }

    ~HashTableCPU(){
        delete[](entries);
        delete[](keys);
        delete[](values);
    }

    // Returns the number of vectors stored.
    int size() { return filled; }

    // Returns a pointer to the keys array.
    short *getKeys() { return keys; }

    // Returns a pointer to the values array.
    T *getValues() { return values; }

    /* Looks up the value vector associated with a given key vector.
     *        k : pointer to the key vector to be looked up.
     *   create : true if a non-existing key should be created.
     */
    T *lookup(short *k, bool create = true) {
        size_t h = hash(k) % capacity;
        int offset = lookupOffset(k, h, create);
        if (offset < 0)
            return nullptr;
        else
            return values + offset;
    }
};


template<typename T> class PermutohedralLatticeCPU {

    int pd, vd, N;
    std::unique_ptr<T[]> scaleFactor;
    HashTableCPU<T> hashTable;
    std::unique_ptr<T[]> elevated;
    std::unique_ptr<T[]> rem0;
    std::unique_ptr<short[]> rank;
    std::unique_ptr<T[]> barycentric;
    // std::unique_ptr<short[]> key;
    std::unique_ptr<T[]> val;

    // slicing is done by replaying splatting (ie storing the sparse matrix)
    struct MatrixEntry {
        int offset; //idx * vd
        T weight;
    };
    std::unique_ptr<MatrixEntry[]> matrix;
    int idx;

    std::unique_ptr<T[]> compute_scale_factor() {
        auto scaleFactor = std::unique_ptr<T[]>(new T[pd]);

        /* We presume that the user would like to do a Gaussian blur of standard deviation
         * 1 in each dimension (or a total variance of pd, summed over dimensions.)
         * Because the total variance of the blur performed by this algorithm is not pd,
         * we must scale the space to offset this.
         *
         * The total variance of the algorithm is (See pg.6 and 10 of paper):
         *  [variance of splatting] + [variance of blurring] + [variance of splatting]
         *   = pd(pd+1)(pd+1)/12 + pd(pd+1)(pd+1)/2 + pd(pd+1)(pd+1)/12
         *   = 2d(pd+1)(pd+1)/3.
         *
         * So we need to scale the space by (pd+1)sqrt(2/3).
         */
        T invStdDev = (pd + 1) * sqrt(2.0 / 3);

        // Compute parts of the rotation matrix E. (See pg.4-5 of paper.)
        for (int i = 0; i < pd; i++) {
            // the diagonal entries for normalization
            scaleFactor[i] = 1.0 / (sqrt((i + 1) * (i + 2))) * invStdDev;
        }
        return scaleFactor;
    }


    void embed_position_vector(const T *position) {
        // embed position vector into the hyperplane
        // first rotate position into the (pd+1)-dimensional hyperplane
        // sm contains the sum of 1..n of our feature vector
        T sm = 0;
        for (int i = pd; i > 0; i--) {
            float cf = position[i - 1] * scaleFactor[i - 1];
            elevated[i] = sm - i * cf;
            sm += cf;
        }
        elevated[0] = sm;
    }

    void find_enclosing_simplex(){
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
            T di = elevated[i] - rem0[i];
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
    }

    void compute_barycentric_coordinates() {
        for(int i = 0; i < pd + 2; i++)
            barycentric[i]=0;
        // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
        for (int i = 0; i <= pd; i++) {
            T delta = (elevated[i] - rem0[i]) *  (1.0 / (pd + 1));
            barycentric[pd - rank[i]] += delta;
            barycentric[pd - rank[i] + 1] -= delta;
        }
        // Wrap around
        barycentric[0] += 1.0 + barycentric[pd + 1];
    }

    void splat_point(const T *position, const T * value) {

        embed_position_vector(position);

        find_enclosing_simplex();

        compute_barycentric_coordinates();

        auto key = new short[pd];
        for (int remainder = 0; remainder <= pd; remainder++) {
            // Compute the location of the lattice point explicitly (all but
            // the last coordinate - it's redundant because they sum to zero)
            for (int i = 0; i < pd; i++) {
                key[i] = static_cast<short>(rem0[i] + remainder);
                if (rank[i] > pd - remainder)
                    key[i] -= (pd + 1);
            }

            // Retrieve pointer to the value at this vertex.
            T *val = hashTable.lookup(key, true);
            // Accumulate values with barycentric weight.
            for (int i = 0; i < vd - 1; i++)
                val[i] += barycentric[remainder] * value[i];

            val[vd - 1] += barycentric[remainder]; //homogeneous coordinate (as if value[vd-1]=1)

            // Record this interaction to use later when slicing
            matrix[idx].offset = val - hashTable.getValues();
            matrix[idx].weight = barycentric[remainder];
            idx++;
        }
        delete[] key;
    }

    void splat(const T * positions, const T * values){
        for (int n = 0; n < N; n++) {
            splat_point(&(positions[n*pd]), &(values[n*(vd-1)]));
        }
    }


    /* Performs slicing out of position vectors. Note that the barycentric weights and the simplex
    * containing each position vector were calculated and stored in the splatting step.
    * We may reuse this to accelerate the algorithm. (See pg. 6 in paper.)
    */
    void slice_point(T* out, int n) {

        T* base = hashTable.getValues();

        for (int j = 0; j < vd; j++)
            val[j] = 0;

        for (int i = 0; i <= pd; i++) {
            MatrixEntry r = matrix[n * (pd + 1) + i];
            for (int j = 0; j < vd; j++) {
                val[j] += r.weight * base[r.offset + j];
            }
        }

        T scale = 1.0 / val[vd - 1];
        for (int j = 0; j < vd - 1; j++) {
            out[n * (vd - 1) + j] = val[j] * scale;
        }

    }

    void slice(T* out){
        for (int n = 0; n < N; n++) {
            slice_point(out, n);
        }
    }


    /* Performs a Gaussian blur along each projected axis in the hyperplane. */
    void blur(bool reverse) {

        // Prepare arrays
        auto n1_key = new short[pd + 1];
        auto n2_key = new short[pd + 1];

        //old and new values contain the lattice points before and after blur
        //auto new_values = new T[vd * hashTable.size()];
        auto new_values = new T[vd * hashTable.capacity];

        auto zero = new T[vd]{0};
        //for (int k = 0; k < vd; k++)
        //    zero[k] = 0;

        // For each of pd+1 axes,
        for (int remainder=reverse?pd:0; remainder >= 0 && remainder <= pd; reverse?remainder--:remainder++){
            // For each vertex in the lattice,
            for (int i = 0; i < hashTable.size(); i++) { // blur point i in dimension j

                short *key = hashTable.getKeys() + i * pd; // keys to current vertex
                for (int k = 0; k < pd; k++) {
                    n1_key[k] = key[k] + 1;
                    n2_key[k] = key[k] - 1;
                }

                n1_key[remainder] = key[remainder] - pd;
                n2_key[remainder] = key[remainder] + pd; // keys to the neighbors along the given axis.

                T *oldVal = hashTable.values + i * vd;
                T *newVal = new_values + i * vd;

                T *n1_value, *n2_value;

                n1_value = hashTable.lookup(n1_key, false); // look up first neighbor
                if (n1_value == nullptr)
                    n1_value = zero;

                n2_value = hashTable.lookup(n2_key, false); // look up second neighbor
                if (n2_value == nullptr)
                    n2_value = zero;

                // Mix values of the three vertices
                for (int k = 0; k < vd; k++)
                    newVal[k] = (0.25 * n1_value[k] + 0.5 * oldVal[k] + 0.25 * n2_value[k]);
            }
            // the freshest data is now in old_values, and new_values is ready to be written over
            std::swap(hashTable.values, new_values);
        }

        delete[](new_values);
        delete[] zero;
        delete[] n1_key;
        delete[] n2_key;
    }

public:

    PermutohedralLatticeCPU(int pd_, int vd_, int N_): pd(pd_), vd(vd_), N(N_), hashTable(pd_, vd_) {

        // Allocate storage for various arrays
        matrix = std::unique_ptr<MatrixEntry[]>(new MatrixEntry[N * (pd + 1)]);
        //matrix = new MatrixEntry[N * (pd + 1)];
        idx = 0;

        //lattice properties
        scaleFactor = compute_scale_factor();

        //arrays that are used in splatting and slicing, they are overwritten for each point but we only allocate once for speed
        // position embedded in subspace Hd
        elevated = std::unique_ptr<T[]>(new T[pd + 1]);
        // remainder-0 and rank describe the enclosing simplex of a point
        rem0 = std::unique_ptr<T[]>(new T[pd + 1]);
        rank = std::unique_ptr<short[]>(new short[pd + 1]);
        // barycentric coordinates of position
        barycentric = std::unique_ptr<T[]>(new T[pd + 2]);
        //val
        val = std::unique_ptr<T[]>(new T[vd]);

    }

    void filter(T * output, const T* input, const T* positions, bool reverse) {
        splat(positions, input);
        blur(reverse);
        slice(output);
    }

};



template <typename T>
static void compute_kernel_cpu(const T * reference,
                               T * positions,
                               int num_super_pixels,
                               int reference_channels,
                               int n_sdims,
                               const int *sdims,
                               T spatial_std,
                               T feature_std){

    int num_dims = n_sdims + reference_channels;

    for(int idx = 0; idx < num_super_pixels; idx++){
        int divisor = 1;
        for(int sdim = n_sdims - 1; sdim >= 0; sdim--){
            positions[num_dims * idx + sdim] = ((idx / divisor) % sdims[sdim]) / spatial_std;
            divisor *= sdims[sdim];
        }
        for(int channel = 0; channel < reference_channels; channel++){
            positions[num_dims * idx + n_sdims + channel] = reference[idx * reference_channels + channel] / feature_std;
        }
    }
};




#endif //PERMUTOHEDRAL_LATTICE_CPU_H
