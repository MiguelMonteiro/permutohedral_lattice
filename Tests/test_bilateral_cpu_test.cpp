#include <cstdio>
#include <cstring>
#include <cstdlib>
#define cimg_display 0
#include "CImg.h"
#include <sys/time.h>
#include <ctime>
#include "bilateral_filter_cpu.h"
#include "utils.h"

using T = double;

int main(int argc, char **argv) {
    if (argc < 5) {
        printf("Usage: ./test_bilateral_cpu <input file> <output file> <spatial standard deviation> <color standard deviation>\n");
        return 1;
    }


    cimg_library::CImg<unsigned char> image(argv[1]);
    int N = image.width() * image.height();
    int sdims[2]{image.height(), image.width()};
    float pixel_depth=255.0;

    auto flat = get_flat_float_from_image<T>(image, pixel_depth);

    //float invSpatialStdev = 1.0f / atof(argv[3]);
    //float invColorStdev = 1.0f / atof(argv[4]);
    //auto positions = compute_kernel(image, invSpatialStdev, invColorStdev);
    T theta_alpha = atof(argv[3]);
    T theta_beta = atof(argv[4]);

    std::clock_t begin = std::clock();
    bilateral_filter_cpu<T>(flat, 3, 2, sdims, N, theta_alpha, theta_beta);
    std::clock_t end = std::clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    printf("Measured from function call: %f seconds\n", elapsed_secs);

    printf("Saving output...\n");
    save_output<T>(flat, image, argv[2], pixel_depth);

    delete[] flat;

    return 0;
}
