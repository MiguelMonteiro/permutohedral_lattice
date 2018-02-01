//
// Created by Miguel Monteiro on 16/01/2018.
//

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
        auto *newValues = new T[vd * capacity / 2]{0};
        std::memcpy(newValues, values, sizeof(T) * vd * filled);
        delete[] values;
        values = newValues;

        // Migrate the key vectors.
        auto *newKeys = new short[pd * capacity / 2];
        std::memcpy(newKeys, keys, sizeof(short) * pd * filled);
        delete[] keys;
        keys = newKeys;

        auto *newEntries = new int[capacity];
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
    };


};




template<typename T> class PermutohedralLatticeCPU {
protected:

    int pd, vd, N;
    std::unique_ptr<int[]> canonical;
    std::unique_ptr<T[]> scaleFactor;
    HashTableCPU<T> hashTable;
    std::unique_ptr<T[]> elevated;
    std::unique_ptr<T[]> rem0;
    std::unique_ptr<short[]> rank;
    std::unique_ptr<T[]> barycentric;
    // std::unique_ptr<short[]> key;

    // slicing is done by replaying splatting (ie storing the sparse matrix)
    struct ReplayEntry {
        int offset;
        T weight;
    };
    std::unique_ptr<ReplayEntry[]> replay;
    int nReplay;


    std::unique_ptr<int[]> compute_canonical_simplex() {
        auto canonical = std::unique_ptr<int[]>(new int[(pd + 1) * (pd + 1)]);
        //auto canonical = new int[(pd + 1) * (pd + 1)];
        // compute the coordinates of the canonical simplex, in which
        // the difference between a contained point and the zero
        // remainder vertex is always in ascending order. (See pg.4 of paper.)
        for (int i = 0; i <= pd; i++) {
            for (int j = 0; j <= pd - i; j++)
                canonical[i * (pd + 1) + j] = i;
            for (int j = pd - i + 1; j <= pd; j++)
                canonical[i * (pd + 1) + j] = i - (pd + 1);
        }
        return canonical;
    }


    //this needs work with the floats
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
        T inv_std_dev = (pd + 1) * sqrt(2.0 / 3);

        // Compute parts of the rotation matrix E. (See pg.4-5 of paper.)
        for (int i = 0; i < pd; i++) {
            // the diagonal entries for normalization
            scaleFactor[i] = 1.0 / (sqrt((i + 1) * (i + 2))) * inv_std_dev;
        }
        return scaleFactor;
    }


    void embed_position_vector(const T *position) {
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

    //this also needs work the floats
    void find_enclosing_simplex(){
        // Find the closest 0-colored simplex through rounding
        // greedily search for the closest zero-colored lattice point
        signed short sum = 0;
        for (int i = 0; i <= pd; i++) {
            T v = elevated[i] * (1.0 / (pd + 1));
            T up = ceil(v) * (pd + 1);
            T down = floor(v) * (pd + 1);
            if (up - elevated[i] < elevated[i] - down) {
                rem0[i] = (signed short) up;
            } else {
                rem0[i] = (signed short) down;
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
        T down_factor = 1.0 / (pd + 1);
        for(int i = 0; i < pd + 2; i++)
            barycentric[i]=0;
        //memset(barycentric, 0, sizeof(float) * (pd + 2));
        // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
        for (int i = 0; i <= pd; i++) {
            T v = (elevated[i] - rem0[i]) * down_factor;
            barycentric[pd - rank[i]] += v;
            barycentric[pd - rank[i] + 1] -= v;
        }
        // Wrap around
        barycentric[0] += 1.0 + barycentric[pd + 1];
    }


    void splat_point(const T *position, const T * value) {

        embed_position_vector(position);

        find_enclosing_simplex();

        compute_barycentric_coordinates();

        //here key seems to be 1 too long
        auto key = new short[pd + 1];
        // Splat the value into each vertex of the simplex, with barycentric weights.
        for (int remainder = 0; remainder <= pd; remainder++) {
            // Compute the location of the lattice point explicitly (all but the last coordinate - it's redundant because they sum to zero)
            for (int i = 0; i < pd; i++)
                key[i] = static_cast<short>(rem0[i] + canonical[remainder * (pd + 1) + rank[i]]);

            // Retrieve pointer to the value at this vertex.
            T *val = hashTable.lookup(key, true);

            // Accumulate values with barycentric weight.
            for (int i = 0; i < vd; i++)
                val[i] += barycentric[remainder] * value[i];

            // Record this interaction to use later when slicing
            replay[nReplay].offset = val - hashTable.getValues();
            replay[nReplay].weight = barycentric[remainder];
            nReplay++;

        }

        delete[] key;

/*    // Compute all vertices and their offset
    for( int remainder=0; remainder<=pd; remainder++ ){
        for( int i=0; i<pd; i++ )
            key[i] = rem0[i] + canonical[ remainder*(pd+1) + rank[i] ];
        offset_[ k*(d_+1)+remainder ] = hash_table.find( key, true );
        rank_[ k*(d_+1)+remainder ] = rank[remainder];
        barycentric_[ k*(d_+1)+remainder ] = barycentric[ remainder ];
    }*/


    }


    void splat(const T * positions, const T * values){

        //auto col = std::unique_ptr<float[]>(new float[vd]);
        auto col = new T[vd];
        col[vd - 1] = 1; // homogeneous coordinate

        T *imPtr = const_cast<T *>(values);
        T *refPtr = const_cast<T *>(positions);
        for (int n = 0; n < N; n++) {

            for (int c = 0; c < vd - 1; c++) {
                col[c] = *imPtr++;
            }

            splat_point(refPtr, col);
            refPtr += pd;
        }
        delete[] col;
    }


    /* Performs slicing out of position vectors. Note that the barycentric weights and the simplex
    * containing each position vector were calculated and stored in the splatting step.
    * We may reuse this to accelerate the algorithm. (See pg. 6 in paper.)
    */
    /*void slice_point(float *col) {
        float *base = hashTable.getValues();
        for (int j = 0; j < vd; j++)
            col[j] = 0;
        for (int i = 0; i <= pd; i++) {
            ReplayEntry r = replay[nReplay++];
            for (int j = 0; j < vd; j++) {
                col[j] += r.weight * base[r.offset + j];
            }
        }
    }*/

    void slice(T * out){

        nReplay = 0;

        int im_channels = vd - 1;

        for (int n = 0; n < N; n++) {

            T *base = hashTable.getValues();
            auto col = new T[vd]{0};
            for (int i = 0; i <= pd; i++) {
                ReplayEntry r = replay[nReplay];
                nReplay++;
                for (int j = 0; j < vd; j++) {
                    col[j] += r.weight * base[r.offset + j];
                }
            }

            T scale = 1.0 / col[im_channels];

            for (int c = 0; c < im_channels; c++) {
                *out = col[c]* scale;
                out++;
            }
            delete[] col;
        }
    }


    /* Performs a Gaussian blur along each projected axis in the hyperplane. */
    void blur() {

        // Prepare arrays
        auto *n1_key = new short[pd + 1];
        auto *n2_key = new short[pd + 1];

        //old and new values contain the lattice points before and after blur
        auto new_values = new T[vd * hashTable.size()];
        T *old_values = hashTable.getValues();
        T *hashTableBase = old_values;

        auto *zero = new T[vd]{0};
        //for (int k = 0; k < vd; k++)
        //    zero[k] = 0;

        // For each of pd+1 axes,
        for (int j = 0; j <= pd; j++) {
            // For each vertex in the lattice,
            for (int i = 0; i < hashTable.size(); i++) { // blur point i in dimension j

                short *key = hashTable.getKeys() + i * (pd); // keys to current vertex
                for (int k = 0; k < pd; k++) {
                    n1_key[k] = key[k] + 1;
                    n2_key[k] = key[k] - 1;
                }
                n1_key[j] = key[j] - pd;
                n2_key[j] = key[j] + pd; // keys to the neighbors along the given axis.

                T *oldVal = old_values + i * vd;
                T *newVal = new_values + i * vd;

                T *n1_value, *n2_value;

                n1_value = hashTable.lookup(n1_key, false); // look up first neighbor
                if (n1_value)
                    n1_value = n1_value - hashTableBase + old_values;
                else
                    n1_value = zero;

                n2_value = hashTable.lookup(n2_key, false); // look up second neighbor
                if (n2_value)
                    n2_value = n2_value - hashTableBase + old_values;
                else
                    n2_value = zero;

                // Mix values of the three vertices
                for (int k = 0; k < vd; k++)
                    newVal[k] = (0.25 * n1_value[k] + 0.5 * oldVal[k] + 0.25 * n2_value[k]);
            }
            std::swap(old_values, new_values);
            // the freshest data is now in old_values, and new_values is ready to be written over
        }

        if(old_values != hashTableBase){
            std::swap(old_values, new_values);
        }
        delete[](new_values);
        /*
        // depending where we ended up, we may have to copy data
        if (old_values != hashTableBase) {
            memcpy(hashTableBase, old_values, hashTable.size() * vd * sizeof(T));
            delete[] old_values;
        } else {
            delete[] new_values;
        }
        printf("\n");*/

        delete[] zero;
        delete[] n1_key;
        delete[] n2_key;
    }

public:

    PermutohedralLatticeCPU(int pd_, int vd_, int N_): pd(pd_), vd(vd_), N(N_), hashTable(pd_, vd_) {

        // Allocate storage for various arrays
        replay = std::unique_ptr<ReplayEntry[]>(new ReplayEntry[N * (pd + 1)]);
        //replay = new ReplayEntry[N * (pd + 1)];
        nReplay = 0;

        //lattice properties
        canonical = compute_canonical_simplex();
        scaleFactor = compute_scale_factor();

        //arrays that are used in splatting, they are overwritten for each point but we only allocate once for speed
        // position embedded in subspace Hd
        elevated = std::unique_ptr<T[]>(new T[pd + 1]);
        // remainder-0 and rank describe the enclosing simplex of a point
        rem0 = std::unique_ptr<T[]>(new T[pd + 1]);
        rank = std::unique_ptr<short[]>(new short[pd + 1]);
        // barycentric coordinates of position
        barycentric = std::unique_ptr<T[]>(new T[pd + 2]);

    }

    void filter(T * output, const T* input, const T* positions) {
        splat(positions, input);
        blur();
        slice(output);
    }

};


template <typename T> static void compute_bilateral_kernel_cpu(const T * reference,
                                                               T * positions,
                                                               int num_super_pixels,
                                                               int reference_channels,
                                                               int n_sdims,
                                                               const int *sdims,
                                                               T theta_alpha,
                                                               T theta_beta){

    int num_dims = n_sdims + reference_channels;

    for(int p = 0; p < num_super_pixels; p++){
        int divisor = 1;
        for(int sdim = 0; sdim < n_sdims; sdim++){
            positions[num_dims * p + sdim] = ((p / divisor) % sdims[sdim]) / theta_alpha;
            divisor *= sdims[sdim];
        }
        for(int channel = 0; channel < reference_channels; channel++){
            positions[num_dims * p + n_sdims + channel] = reference[p * reference_channels + channel] / theta_beta;
        }
    }
};

template <typename T> static void lattice_filter_cpu(T * output, const T *input, const T *positions, int pd, int vd, int n){
    PermutohedralLatticeCPU<T> lattice(pd, vd, n);
    lattice.filter(output, input, positions);
}


#endif //PERMUTOHEDRAL_LATTICE_CPU_H
