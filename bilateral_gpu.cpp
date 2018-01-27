#include <cstdio>
#include <cstring>
#include <cstdlib>
#define cimg_display 0
#include "CImg.h"
#include <ctime>
#include "utils.h"



//extern "C++" template<int pd, int vd> void permutohedral::filter(float *values, float *positions, int n);
extern "C++" void filter(float *input, float *positions, int n);

int main(int argc, char **argv) {

    if (argc < 5) {
		printf("Usage: ./bilateral <image file> <output file> <spatial standard deviation> <color standard deviation>\n");
		return 1;
	}

	float pixel_depth = 255.0;

	cimg_library::CImg<unsigned char> image(argv[1]);
    auto flat = get_flat_float_from_image(image, pixel_depth);

    float invSpatialStdev = 1.0f / atof(argv[3]);
    float invColorStdev = 1.0f / atof(argv[4]);

    printf("Constructing inputs...\n");
    // Construct the position vectors out of x, y, r, g, and b.
    auto positions = compute_kernel(image, invSpatialStdev, invColorStdev);

    int N = image.width() * image.height();


	// Filter the image with respect to the position vectors.
	printf("Calling filter...\n");
	std:clock_t begin = std::clock();
    filter(flat, positions, N);
	std::clock_t end = std::clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	printf("%f seconds\n", elapsed_secs);

    save_output(flat, image, argv[2], pixel_depth);

    delete[] flat;
    delete[] positions;
    return 0;

}
