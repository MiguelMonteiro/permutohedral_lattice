#include <cstdio>
#include <cstring>
#include <cstdlib>
#define cimg_display 0
#include "CImg.h"
#include <ctime>
#include "utils.h"
#include "bilateral_filter_gpu.h"

int main(int argc, char **argv) {
    if (argc < 5) {
		printf("Usage: ./test_bilateral_gpu <image file> <output file> <spatial standard deviation> <color standard deviation>\n");
		return 1;
	}

	cimg_library::CImg<unsigned char> image(argv[1]);
    float pixel_depth = 255.0;
    int N = image.width() * image.height();
    int sdims[2]{image.height(), image.width()};

    auto flat = get_flat_float_from_image(image, pixel_depth);

    //float invSpatialStdev = 1.0f / atof(argv[3]);
    //float invColorStdev = 1.0f / atof(argv[4]);

    //printf("Constructing inputs...\n");
    // Construct the position vectors out of x, y, r, g, and b.
    //auto positions = compute_kernel(image, invSpatialStdev, invColorStdev);
    float theta_alpha = atof(argv[3]);
    float theta_beta = atof(argv[4]);

	std::clock_t begin = std::clock();
    bilateral_filter_gpu<float>(flat, 3, 2, sdims, N, theta_alpha, theta_beta);
	std::clock_t end = std::clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    printf("Measured from function call: %f seconds\n", elapsed_secs);


    //
    printf("Testing for double percision\n");
    double* flat_double = get_flat_float_from_image<double>(image, pixel_depth);
    begin = std::clock();
    bilateral_filter_gpu<double>(flat_double, 3, 2, sdims, N, (double)theta_alpha, (double) theta_beta);
    end = std::clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    printf("Measured from function call: %f seconds\n", elapsed_secs);
    delete[] flat_double;
     //

    save_output<float>(flat, image, argv[2], pixel_depth);

    delete[] flat;





    return 0;
}
