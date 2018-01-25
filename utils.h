//
// Created by Miguel Monteiro on 25/01/2018.
//

#define cimg_display 0
#include "CImg.h"

float * get_flat_float_from_image(cimg_library::CImg<unsigned char> image, float pixel_depth=255.0){

    auto flat = new float[image.width() * image.height() * 3]{0};
    int idx{0};

    for(int x=0; x < image.width(); ++x){
        for(int y=0; y < image.height(); ++y){
            for (int channel=0; channel < 3; ++channel){
                flat[idx] = image(x, y, 0, channel) / pixel_depth;
                idx++;
            }
        }
    }
    return flat;
}

float * compute_kernel(cimg_library::CImg<unsigned char> image, float invSpatialStdev, float invColorStdev, float pixel_depth=255.0){
    auto positions = new float [image.width() * image.height() * 5]{0};
    int idx{0};
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
    return positions;
}

void save_output(float*out, cimg_library::CImg<unsigned char> image, char*filename, float pixel_depth=255.0){
    int idx{0};
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
    image.save(filename);
}