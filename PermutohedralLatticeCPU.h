//
// Created by Miguel Monteiro on 16/01/2018.
//

#ifndef PERMUTOHEDRAL_LATTICE_BILATERAL_ORIGINAL_PERMUTOHEDRALLATTICE_H
#define PERMUTOHEDRAL_LATTICE_BILATERAL_ORIGINAL_PERMUTOHEDRALLATTICE_H

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
class HashTableCPU {
public:
    /* Constructor
     *  pd_: the dimensionality of the position vectors on the hyperplane.
     *  vd_: the dimensionality of the value vectors
     */
    HashTableCPU(int pd_, int vd_) : pd(pd_), vd(vd_) {
        capacity = 1 << 15;
        filled = 0;
        entries = new Entry[capacity];
        keys = new short[pd * capacity / 2];
        values = new float[vd * capacity / 2]{0};
    }

    // Returns the number of vectors stored.
    int size() { return filled; }

    // Returns a pointer to the keys array.
    short *getKeys() { return keys; }

    // Returns a pointer to the values array.
    float *getValues() { return values; }

    /* Returns the index into the hash table for a given key.
     *     key: a pointer to the position vector.
     *       h: hash of the position vector.
     *  create: a flag specifying whether an entry should be created,
     *          should an entry with the given key not found.
     */
    int lookupOffset(short *key, size_t h, bool create = true) {

        // Double hash table size if necessary
        if (filled >= (capacity / 2) - 1) { grow(); }

        // Find the entry with the given key
        while (true) {
            Entry e = entries[h];
            // check if the cell is empty
            if (e.keyIdx == -1) {
                if (!create)
                    return -1; // Return not found.
                // need to create an entry. Store the given key.
                for (int i = 0; i < pd; i++)
                    keys[filled * pd + i] = key[i];
                e.keyIdx = filled * pd;
                e.valueIdx = filled * vd;
                entries[h] = e;
                filled++;
                return e.valueIdx;
            }

            // check if the cell has a matching key
            bool match = true;
            for (int i = 0; i < pd && match; i++)
                match = keys[e.keyIdx + i] == key[i];
            if (match)
                return e.valueIdx;

            // increment the bucket with wraparound
            h++;
            if (h == capacity)
                h = 0;
        }
    }

    /* Looks up the value vector associated with a given key vector.
     *        k : pointer to the key vector to be looked up.
     *   create : true if a non-existing key should be created.
     */
    float *lookup(short *k, bool create = true) {
        size_t h = hash(k) % capacity;
        int offset = lookupOffset(k, h, create);
        if (offset < 0)
            return nullptr;
        else
            return values + offset;
    };

    /* Hash function used in this implementation. A simple base conversion. */
    size_t hash(const short *key) {
        size_t k = 0;
        for (int i = 0; i < pd; i++) {
            k += key[i];
            k *= 2531011;
        }
        return k;
    }

private:
    /* Grows the size of the hash table */
    void grow() {
        printf("Resizing hash table\n");

        size_t oldCapacity = capacity;
        capacity *= 2;

        // Migrate the value vectors.
        auto *newValues = new float[vd * capacity / 2]{0};
        std::memcpy(newValues, values, sizeof(float) * vd * filled);
        delete[] values;
        values = newValues;

        // Migrate the key vectors.
        auto *newKeys = new short[pd * capacity / 2];
        std::memcpy(newKeys, keys, sizeof(short) * pd * filled);
        delete[] keys;
        keys = newKeys;

        auto *newEntries = new Entry[capacity];

        // Migrate the table of indices.
        for (size_t i = 0; i < oldCapacity; i++) {
            if (entries[i].keyIdx == -1) continue;
            size_t h = hash(keys + entries[i].keyIdx) % capacity;
            while (newEntries[h].keyIdx != -1) {
                h++;
                if (h == capacity) h = 0;
            }
            newEntries[h] = entries[i];
        }
        delete[] entries;
        entries = newEntries;
    }

    // Private struct for the hash table entries.
    struct Entry {
        Entry() : keyIdx(-1), valueIdx(-1) {}

        int keyIdx;
        int valueIdx;
    };

    short *keys;
    float *values;
    Entry *entries;
    size_t capacity, filled;
    int pd, vd;
};


class PermutohedralLatticeCPU {
protected:

    int pd, vd, N;
    std::unique_ptr<int[]> canonical;
    std::unique_ptr<float[]> scaleFactor;

    std::unique_ptr<float[]> elevated;
    std::unique_ptr<float[]> rem0;
    std::unique_ptr<short[]> rank;
    std::unique_ptr<float[]> barycentric;
    // std::unique_ptr<short[]> key;


    // slicing is done by replaying splatting (ie storing the sparse matrix)
    struct ReplayEntry {
        int offset;
        float weight;
    } *replay;

    int nReplay;

    std::unique_ptr<int[]> compute_canonical_simplex();

    std::unique_ptr<float[]> compute_scale_factor();

    void embed_position_vector(const float *position);

    void find_enclosing_simplex();

    void compute_barycentric_coordinates();

    void splat_point(const float *position, const float *value);

    void slice_point(float *col);


public:

    HashTableCPU hashTable;

    PermutohedralLatticeCPU(int pd_, int vd_, int N_);

    void splat(float *positions, float *values);

    void blur();

    void slice(float *out);
};


static float *compute_kernel(float * image){}

static void filter_cpu(float * im, float* ref, int ref_channels, int im_channels, int num_points){

    timeval t[5];

    // Create lattice
    gettimeofday(t + 0, nullptr);
    PermutohedralLatticeCPU lattice(ref_channels, im_channels, num_points);

    // Splat into the lattice
    gettimeofday(t + 1, nullptr);
    printf("Splatting...\n");
    lattice.splat(ref, im);


    // Blur the lattice
    gettimeofday(t + 2, nullptr);
    printf("Blurring...");
    lattice.blur();

    // Slice from the lattice
    gettimeofday(t + 3, nullptr);
    printf("Slicing...\n");
    lattice.slice(im);


    // Print time elapsed for each step
    gettimeofday(t + 4, nullptr);
    const char *names[4] = {"Init  ", "Splat ", "Blur  ", "Slice "};
    for (int i = 1; i < 5; i++)
        printf("%s: %3.3f ms\n", names[i - 1], (t[i].tv_sec - t[i - 1].tv_sec) +
                                               (t[i].tv_usec - t[i - 1].tv_usec) / 1000000.0);
}


#endif //PERMUTOHEDRAL_LATTICE_BILATERAL_ORIGINAL_PERMUTOHEDRALLATTICE_H
