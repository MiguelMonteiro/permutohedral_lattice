#include <stdio.h>
#define cimg_display 0
#include "CImg.h"
#include <sys/time.h>
#include <ctime>


//extern "C++" template<int pd, int vd> void permutohedral::filter(float *values, float *positions, int n);
extern "C++" void filter(float *values, float *positions, int n);

int main(int argc, char **argv) {
	if (argc < 5) {
		printf("Usage: ./bilateral <image file> <output file> <spatial standard deviation> <color standard deviation>\n");
		printf("        PNG, JPG are supported file formats.\n");
		printf("        For instance, try ./bilateral image.png output.png 4 0.5\n");
		return 1;
	}

	/* Start a timer that expires after 2.5 seconds */
	struct itimerval timer;
	timer.it_value.tv_sec = 5;
	timer.it_value.tv_usec = 500000;
	timer.it_interval.tv_sec = 0;
	timer.it_interval.tv_usec = 0;
	setitimer (ITIMER_VIRTUAL, &timer, 0);


	float pixel_depth = 255.0;
	cimg_library::CImg<unsigned char> image(argv[1]);
	float* flat = new float[image.width() * image.height() * 3]{0};
	int idx=0;

	for(int x=0; x < image.width(); ++x){
		for(int y=0; y < image.height(); ++y){
			for (int channel=0; channel < 3; ++channel){
				flat[idx] = image(x, y, 0, channel) / pixel_depth;
				idx++;
			}
		}
	}


	float invSpatialStdev = 1.0f/atof(argv[3]);
	float invColorStdev = 1.0f/atof(argv[4]);

	printf("Constructing inputs...\n");
	// Construct the position vectors out of x, y, r, g, and b.
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


	// Filter the image with respect to the position vectors.
	printf("Calling filter...\n");
	std:clock_t begin = std::clock();
    filter(flat, positions, image.width() * image.height());
	std::clock_t end = std::clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	printf("%f seconds\n", elapsed_secs);

	printf("Saving output...\n");

	cimg_library::CImg<unsigned char> original("cpu_result.bmp");
	int wrong_pixels = 0;
	int tol=0;

	idx=0;
	for(int x=0; x < image.width(); ++x){
		for(int y=0; y < image.height(); ++y){
			for (int channel=0; channel < 3; ++channel){
				int value{int(flat[idx] * pixel_depth)};
				if(value > pixel_depth)
					value = (int) pixel_depth;
				if(value < 0)
					value = 0;
				if(abs(original(x,y,channel) - value) > tol)
					wrong_pixels+=1;

				image(x, y, channel) = value;
				idx++;
			}
		}
	}
    image.save(argv[2]);

	if(wrong_pixels==0)
		printf("The algorithm produced the correct result\n");
	else
		printf("The result is not correct, it is %f percent different\n", (100.0*wrong_pixels/(3.0*image.width()*image.height())));

	delete[] flat;
	delete[] positions;

	return 0;
}
