#include <cstdio>
#include <cstring>
#include <cstdlib>
#define cimg_display 0
#include "CImg.h"
#include <sys/time.h>
#include <ctime>

#include "PermutohedralLattice.h"


void filter(float * im, float* ref, float * out, int ref_channels, int im_channels, int num_points){

    timeval t[5];

    // Create lattice
    gettimeofday(t + 0, nullptr);
    PermutohedralLattice lattice(ref_channels, im_channels, num_points);

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
    lattice.slice(out);


    // Print time elapsed for each step
    gettimeofday(t + 4, nullptr);
    const char *names[4] = {"Init  ", "Splat ", "Blur  ", "Slice "};
    for (int i = 1; i < 5; i++)
        printf("%s: %3.3f ms\n", names[i - 1], (t[i].tv_sec - t[i - 1].tv_sec) +
                                               (t[i].tv_usec - t[i - 1].tv_usec) / 1000000.0);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        printf("Usage: ./bilateral <input file> <output file> <spatial standard deviation> <color standard deviation>\n");
        printf("        PNG, JPG are supported file formats.\n");
        printf("        For instance, try ./bilateral input.png output.png 4 0.5\n");
        return 1;
    }
    float pixel_depth = 255.0;
    cimg_library::CImg<unsigned char> image(argv[1]);

    auto flat = new float [image.width() * image.height() * 3]{0};
    int idx=0;
    for(int x=0; x < image.width(); ++x){
        for(int y=0; y < image.height(); ++y){
            for (int channel=0; channel < 3; ++channel){
                flat[idx] = image(x, y, 0, channel) / pixel_depth;
                idx++;
            }
        }
    }

    float invSpatialStdev = 1.0f / atof(argv[3]);
    float invColorStdev = 1.0f / atof(argv[4]);

    auto positions = new float [image.width() * image.height() * 5]{0};

    idx = 0;
    for(int x=0; x < image.width(); ++x){
        for(int y=0; y < image.height(); ++y){
            positions[idx] = invSpatialStdev * x;
            positions[idx+1] = invSpatialStdev * y;
            positions[idx+2] = invColorStdev * image(x, y, 0) / pixel_depth;
            positions[idx+3] = invColorStdev * image(x, y, 1) / pixel_depth;
            positions[idx+4] = invColorStdev * image(x, y, 2) / pixel_depth;
            idx+=5;
        }
    }


    // Filter the input with respect to the position vectors. (see permutohedral.h)
    int N = image.width() * image.height();
    auto out = new float [N * 3]{255};

    printf("Calling filter...\n");
    std:clock_t begin = std::clock();
    filter(flat, positions, out, 5, 3, N);
    std::clock_t end = std::clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    printf("%f seconds\n", elapsed_secs);


    idx=0;
    for(int x=0; x < image.width(); ++x){
        for(int y=0; y < image.height(); ++y){
            for (int channel=0; channel < 3; ++channel){
                int value{int(out[idx] * pixel_depth)};
                if(value > pixel_depth)
                    value = (int) pixel_depth;
                if(value < 0)
                    value = 0;
                image(x, y, channel) = value;
                idx++;
            }
        }
    }

    image.save(argv[2]);

    delete[] flat;
    delete[] positions;
    delete[] out;

    return 0;
}
