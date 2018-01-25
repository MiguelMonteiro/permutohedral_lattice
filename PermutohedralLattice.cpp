//
// Created by Miguel Monteiro on 16/01/2018.
//
#include <cmath>
#include <memory>
#include <iostream>
#include <memory>
#include "PermutohedralLattice.h"

std::unique_ptr<int[]> PermutohedralLattice::compute_canonical_simplex() {
    auto canonical = std::unique_ptr<int[]>(new int[(d + 1) * (d + 1)]);
    //auto canonical = new int[(d + 1) * (d + 1)];
    // compute the coordinates of the canonical simplex, in which
    // the difference between a contained point and the zero
    // remainder vertex is always in ascending order. (See pg.4 of paper.)
    for (int i = 0; i <= d; i++) {
        for (int j = 0; j <= d - i; j++)
            canonical[i * (d + 1) + j] = i;
        for (int j = d - i + 1; j <= d; j++)
            canonical[i * (d + 1) + j] = i - (d + 1);
    }
    return canonical;
}

std::unique_ptr<float[]> PermutohedralLattice::compute_scale_factor() {
    auto scaleFactor = std::unique_ptr<float[]>(new float[d]);

    /* We presume that the user would like to do a Gaussian blur of standard deviation
     * 1 in each dimension (or a total variance of d, summed over dimensions.)
     * Because the total variance of the blur performed by this algorithm is not d,
     * we must scale the space to offset this.
     *
     * The total variance of the algorithm is (See pg.6 and 10 of paper):
     *  [variance of splatting] + [variance of blurring] + [variance of splatting]
     *   = d(d+1)(d+1)/12 + d(d+1)(d+1)/2 + d(d+1)(d+1)/12
     *   = 2d(d+1)(d+1)/3.
     *
     * So we need to scale the space by (d+1)sqrt(2/3).
     */
    float inv_std_dev = (d + 1) * sqrtf(2.0f / 3);

    // Compute parts of the rotation matrix E. (See pg.4-5 of paper.)
    for (int i = 0; i < d; i++) {
        // the diagonal entries for normalization
        scaleFactor[i] = 1.0f / (sqrtf((float) (i + 1) * (i + 2))) * inv_std_dev;
    }
    return scaleFactor;
}

void PermutohedralLattice::embed_position_vector(const float *position) {
    // first rotate position into the (d+1)-dimensional hyperplane
    // sm contains the sum of 1..n of our feature vector
    float sm = 0;
    for (int i = d; i > 0; i--) {
        float cf = position[i - 1] * scaleFactor[i - 1];
        elevated[i] = sm - i * cf;
        sm += cf;
    }
    elevated[0] = sm;
}


void PermutohedralLattice::find_enclosing_simplex(){
    // Find the closest 0-colored simplex through rounding
    float down_factor = 1.0f / (d + 1);
    float up_factor = (d + 1);
    int sum = 0;
    for (int i = 0; i <= d; i++) {
        int rd2;
        float v = down_factor * elevated[i];
        float up = ceilf(v) * up_factor;
        float down = floorf(v) * up_factor;
        if (up - elevated[i] < elevated[i] - down)
            rd2 = (short) up;
        else
            rd2 = (short) down;

        rem0[i] = rd2;
        sum += rd2 * down_factor;
    }

    // Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the features values)
    for (int i = 0; i <= d; i++)
        rank[i] = 0;
    for (int i = 0; i < d; i++) {
        double di = elevated[i] - rem0[i];
        for (int j = i + 1; j <= d; j++)
            if (di < elevated[j] - rem0[j])
                rank[i]++;
            else
                rank[j]++;
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    for (int i = 0; i <= d; i++) {
        rank[i] += sum;
        if (rank[i] < 0) {
            rank[i] += d + 1;
            rem0[i] += d + 1;
        } else if (rank[i] > d) {
            rank[i] -= d + 1;
            rem0[i] -= d + 1;
        }
    }
}


void PermutohedralLattice::compute_barycentric_coordinates() {
    float down_factor = 1.0f / (d + 1);
    for(int i = 0; i < d + 2; i++)
        barycentric[i]=0;
    //memset(barycentric, 0, sizeof(float) * (d + 2));
    // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
    for (int i = 0; i <= d; i++) {
        float v = (elevated[i] - rem0[i]) * down_factor;
        barycentric[d - rank[i]] += v;
        barycentric[d - rank[i] + 1] -= v;
    }
    // Wrap around
    barycentric[0] += 1.0 + barycentric[d + 1];
}


void PermutohedralLattice::splat_point(const float *position, const float * value) {

    embed_position_vector(position);

    find_enclosing_simplex();

    compute_barycentric_coordinates();

    //here key seems to be 1 too long
    auto key = new short[d + 1];
    // Splat the value into each vertex of the simplex, with barycentric weights.
    for (int remainder = 0; remainder <= d; remainder++) {
        // Compute the location of the lattice point explicitly (all but the last coordinate - it's redundant because they sum to zero)
        for (int i = 0; i < d; i++)
            key[i] = rem0[i] + canonical[remainder * (d + 1) + rank[i]];

        // Retrieve pointer to the value at this vertex.
        float *val = hashTable.lookup(key, true);

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
    for( int remainder=0; remainder<=d; remainder++ ){
        for( int i=0; i<d; i++ )
            key[i] = rem0[i] + canonical[ remainder*(d+1) + rank[i] ];
        offset_[ k*(d_+1)+remainder ] = hash_table.find( key, true );
        rank_[ k*(d_+1)+remainder ] = rank[remainder];
        barycentric_[ k*(d_+1)+remainder ] = barycentric[ remainder ];
    }*/


}


void PermutohedralLattice::splat(float * positions, float * values){

    //auto col = std::unique_ptr<float[]>(new float[vd]);
    auto col = new float[vd];
    col[vd - 1] = 1; // homogeneous coordinate

    float *imPtr = values;
    float *refPtr = positions;
    for (int n = 0; n < N; n++) {

        for (int c = 0; c < vd - 1; c++) {
            col[c] = *imPtr++;
        }

        splat_point(refPtr, col);
        refPtr += d;
    }
    delete[] col;
}



/* Performs slicing out of position vectors. Note that the barycentric weights and the simplex
 * containing each position vector were calculated and stored in the splatting step.
 * We may reuse this to accelerate the algorithm. (See pg. 6 in paper.)
 */
void PermutohedralLattice::slice_point(float *col) {
    float *base = hashTable.getValues();
    for (int j = 0; j < vd; j++)
        col[j] = 0;
    for (int i = 0; i <= d; i++) {
        ReplayEntry r = replay[nReplay++];
        for (int j = 0; j < vd; j++) {
            col[j] += r.weight * base[r.offset + j];
        }
    }
}

void PermutohedralLattice::slice(float * out){

    nReplay = 0;

    int im_channels = vd -1;

    for (int n = 0; n < N; n++) {

        float *base = hashTable.getValues();
        auto col = new float[vd]{0};
        for (int i = 0; i <= d; i++) {
            ReplayEntry r = replay[nReplay];
            nReplay++;
            for (int j = 0; j < vd; j++) {
                col[j] += r.weight * base[r.offset + j];
            }
        }

        float scale = 1.0f / col[im_channels];

        for (int c = 0; c < im_channels; c++) {
            *out = col[c]* scale;
            out++;
        }
        delete[] col;
    }
}


/* Performs a Gaussian blur along each projected axis in the hyperplane. */
void PermutohedralLattice::blur() {

    // Prepare arrays
    auto *n1_key = new short[d + 1];
    auto *n2_key = new short[d + 1];

    //old and new values contain the lattice points before and after blur
    auto *new_values = new float[vd * hashTable.size()];
    float *old_values = hashTable.getValues();
    float *hashTableBase = old_values;

    auto *zero = new float[vd]{0};
    //for (int k = 0; k < vd; k++)
    //    zero[k] = 0;

    // For each of d+1 axes,
    for (int j = 0; j <= d; j++) {
        // For each vertex in the lattice,
        for (int i = 0; i < hashTable.size(); i++) { // blur point i in dimension j

            short *key = hashTable.getKeys() + i * (d); // keys to current vertex
            for (int k = 0; k < d; k++) {
                n1_key[k] = key[k] + 1;
                n2_key[k] = key[k] - 1;
            }
            n1_key[j] = key[j] - d;
            n2_key[j] = key[j] + d; // keys to the neighbors along the given axis.

            float *oldVal = old_values + i * vd;
            float *newVal = new_values + i * vd;

            float *n1_value, *n2_value;

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
                newVal[k] = (0.25f * n1_value[k] + 0.5f * oldVal[k] + 0.25f * n2_value[k]);
        }
        std::swap(old_values, new_values);
        // the freshest data is now in old_values, and new_values is ready to be written over
    }




    // depending where we ended up, we may have to copy data
    if (old_values != hashTableBase) {
        memcpy(hashTableBase, old_values, hashTable.size() * vd * sizeof(float));
        delete[] old_values;
    } else {
        delete[] new_values;
    }
    printf("\n");

    delete[] zero;
    delete[] n1_key;
    delete[] n2_key;
}


PermutohedralLattice::PermutohedralLattice(int d_, int im_channels, int N_): d(d_), vd(im_channels + 1), N(N_), hashTable(
        static_cast<size_t>((d + 1) * N_), d_, im_channels + 1) {

    // Allocate storage for various arrays
    replay = new ReplayEntry[N * (d + 1)];
    nReplay = 0;

    //lattice properties
    canonical = compute_canonical_simplex();
    scaleFactor = compute_scale_factor();

    //arrays that are used in splatting, they are overwritten for each point but we only allocate once for speed
    // position embedded in subspace Hd
    elevated = std::unique_ptr<float[]>(new float[d + 1]);
    // remainder-0 and rank describe the enclosing simplex of a point
    rem0 = std::unique_ptr<float[]>(new float[d + 1]);
    rank = std::unique_ptr<short[]>(new short[d + 1]);
    // barycentric coordinates of position
    barycentric = std::unique_ptr<float[]>(new float[d + 2]);


}