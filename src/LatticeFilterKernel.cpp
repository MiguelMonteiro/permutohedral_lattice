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

#include "LatticeFilterKernel.h"
#include "PermutohedralLatticeCPU.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

//#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <typename T, int pd, int vd>
struct LatticeFilter<CPUDevice, T, pd, vd> {

    LatticeFilter(OpKernelContext *context,
                  int num_hypervoxels,
                  int num_input_channels,
                  int num_spatial_dims,
                  int num_reference_channels,
                  const T *spatial_std,
                  const T *color_std,
                  bool reverse): context(context),
                                 num_hypervoxels(num_hypervoxels),
                                 num_spatial_dims(num_spatial_dims),
                                 num_input_channels(num_input_channels),
                                 num_reference_channels(num_reference_channels),
                                 spatial_std(spatial_std),
                                 color_std(color_std),
                                 reverse(reverse){};

    void operator()(T* output_image,
                    const T *input_image,
                    const T *reference_image,
                    int *spatial_dims){
        int pd_ = num_reference_channels + num_spatial_dims;
        int vd_ = num_input_channels + 1;
        assert(pd == pd_ && vd == vd_);

        auto position_vectors = new T[num_hypervoxels * pd];
        compute_position_vectors(reference_image,
                                 position_vectors,
                                 num_hypervoxels,
                                 num_reference_channels,
                                 num_spatial_dims,
                                 spatial_dims,
                                 spatial_std,
                                 color_std);

        auto lattice = PermutohedralLatticeCPU<T>(pd, vd, num_hypervoxels);
        lattice.filter(output_image, input_image, position_vectors, reverse);
    }

private:
    OpKernelContext* context;
    int num_hypervoxels;
    int num_input_channels;
    int num_spatial_dims;
    int num_reference_channels;
    const T *spatial_std;
    const T *color_std;
    bool reverse;

};

REGISTER_OP("BilateralFilter")
        .Attr("T: {float32, float64}")
        .Attr("reverse: bool = false")
        .Input("input_image: T")
        .Input("reference_image: T")
        .Input("theta_spatial: T")
        .Input("theta_color: T")
        .Output("output: T")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            c->set_output(0, c->input(0));
            return Status::OK();
        });

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T, int pd, int vd>
class BilateralFilterOp : public OpKernel {
public:
    explicit BilateralFilterOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("reverse", &reverse));
    }

    void Compute(OpKernelContext* context) override {
        // Grab the input tensors
        const Tensor& input_image_tensor = context->input(0);
        OP_REQUIRES(context, input_image_tensor.NumElements() <= tensorflow::kint32max,
                    errors::InvalidArgument("Too many elements in tensor"));
        const Tensor& reference_image_tensor = context->input(1);
        OP_REQUIRES(context, reference_image_tensor.NumElements() <= tensorflow::kint32max,
                    errors::InvalidArgument("Too many elements in tensor"));
        assert(reference_image_tensor.shape() == input_image_tensor.shape());
        const Tensor& theta_spatial = context->input(2);
        assert(theta_spatial.dims() == 0);
        const Tensor& theta_color = context->input(3);
        assert(theta_color.dims() == 0);

        // Create an output tensor
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_image_tensor.shape(), &output_tensor));

        // calculate dimensions; dimension 0 is batch; last dimension is channel
        auto num_hypervoxels = static_cast<int>(input_image_tensor.shape().num_elements());
        int rank = input_image_tensor.dims();
        int num_spatial_dims = rank - 2;
        auto batch_size = static_cast<int>(input_image_tensor.dim_size(0));
        auto num_input_channels = static_cast<int>(input_image_tensor.dim_size(rank - 1));
        auto num_reference_channels = static_cast<int>(reference_image_tensor.dim_size(rank - 1));
        auto spatial_dims = new int[num_spatial_dims];
        for (int i = 0; i < num_spatial_dims; i++)
            spatial_dims[i] = static_cast<int>(input_image_tensor.dim_size(i + 1));

        const T *spatial_std = &(theta_spatial.flat<T>().data()[0]);
        const T* color_std = &(theta_color.flat<T>().data()[0]);

        auto filter = LatticeFilter<Device, T, pd, vd>(context,
                                                       num_hypervoxels,
                                                       num_input_channels,
                                                       num_spatial_dims,
                                                       num_reference_channels,
                                                       spatial_std,
                                                       color_std,
                                                       reverse);
        for(int b=0; b < batch_size; b++){
            filter(&(output_tensor->flat<T>().data()[b * num_hypervoxels * num_input_channels]),
                   &(input_image_tensor.flat<T>().data()[b * num_hypervoxels * num_input_channels]),
                   &(reference_image_tensor.flat<T>().data()[b * num_hypervoxels * num_reference_channels]),
                   spatial_dims);
        }
        delete[](spatial_dims);
    }

private:
    bool reverse;
};

REGISTER_OP("GaussianFilter")
        .Attr("T: {float32, float64}")
        .Attr("reverse: bool = false")
        .Input("input_image: T")
        .Input("theta_spatial: T")
        .Output("output: T")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            c->set_output(0, c->input(0));
            return Status::OK();
        });

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T, int pd, int vd>
class GaussianFilterOp : public OpKernel {
public:
    explicit GaussianFilterOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("reverse", &reverse));
    }

    void Compute(OpKernelContext* context) override {
        // Grab the input tensors
        const Tensor& input_image_tensor = context->input(0);
        OP_REQUIRES(context, input_image_tensor.NumElements() <= tensorflow::kint32max,
                    errors::InvalidArgument("Too many elements in tensor"));

        const Tensor& theta_spatial = context->input(1);
        assert(theta_spatial.dims() == 0);

        // Create an output tensor
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_image_tensor.shape(), &output_tensor));

        // calculate dimensions; dimension 0 is batch; last dimension is channel
        auto num_hypervoxels = static_cast<int>(input_image_tensor.shape().num_elements());
        int rank = input_image_tensor.dims();
        int num_spatial_dims = rank - 2;
        auto batch_size = static_cast<int>(input_image_tensor.dim_size(0));
        auto num_input_channels = static_cast<int>(input_image_tensor.dim_size(rank - 1));
        auto spatial_dims = new int[num_spatial_dims];
        for (int i = 0; i < num_spatial_dims; i++)
            spatial_dims[i] = static_cast<int>(input_image_tensor.dim_size(i + 1));

        const T *spatial_std = &(theta_spatial.flat<T>().data()[0]);

        auto filter = LatticeFilter<Device, T, pd, vd>(context,
                                                       num_hypervoxels,
                                                       num_input_channels,
                                                       num_spatial_dims,
                                                       0,
                                                       spatial_std,
                                                       nullptr,
                                                       reverse);
        for(int b=0; b < batch_size; b++){
            filter(&(output_tensor->flat<T>().data()[b * num_hypervoxels * num_input_channels]),
                   &(input_image_tensor.flat<T>().data()[b * num_hypervoxels * num_input_channels]),
                   nullptr,
                   spatial_dims);
        }
        delete[](spatial_dims);
    }

private:
    bool reverse;
};

#ifndef SPATIAL_DIMS
#define SPATIAL_DIMS 2
#endif
#ifndef INPUT_CHANNELS
#define INPUT_CHANNELS 3
#endif
#ifndef REFERENCE_CHANNELS
#define REFERENCE_CHANNELS 3
#endif

//SPATIAL_DIMS + REFERENCE_CHANNELS INPUT_CHANNELS + 1
// Register the CPU kernels.
#define REGISTER_BILATERAL_CPU(T) REGISTER_KERNEL_BUILDER(Name("BilateralFilter").Device(DEVICE_CPU).TypeConstraint<T>("T"), BilateralFilterOp<CPUDevice, T, SPATIAL_DIMS + REFERENCE_CHANNELS, INPUT_CHANNELS + 1>);
#define REGISTER_GAUSSIAN_CPU(T) REGISTER_KERNEL_BUILDER(Name("GaussianFilter").Device(DEVICE_CPU).TypeConstraint<T>("T"), GaussianFilterOp<CPUDevice, T, SPATIAL_DIMS, INPUT_CHANNELS + 1>);

REGISTER_BILATERAL_CPU(float);
REGISTER_BILATERAL_CPU(double);
REGISTER_GAUSSIAN_CPU(float);
REGISTER_GAUSSIAN_CPU(double);


// Register the GPU kernels.
#ifdef GOOGLE_CUDA
/* Declare explicit instantiations in kernel_example.cu.cc. */
/*extern template struct LatticeFilter<GPUDevice, float>;
extern template struct LatticeFilter<GPUDevice, double>*/
/* Declare explicit instantiations in kernel_example.cu.cc. *//*
extern template struct LatticeFilter<GPUDevice, float, SPATIAL_DIMS + REFERENCE_CHANNELS, INPUT_CHANNELS + 1>;
extern template struct LatticeFilter<GPUDevice, double, SPATIAL_DIMS + REFERENCE_CHANNELS, INPUT_CHANNELS + 1>;
extern template struct LatticeFilter<GPUDevice, float, SPATIAL_DIMS, INPUT_CHANNELS + 1>;
extern template struct LatticeFilter<GPUDevice, double, SPATIAL_DIMS, INPUT_CHANNELS + 1>;*/

#define REGISTER_BILATERAL_GPU(T) REGISTER_KERNEL_BUILDER(Name("BilateralFilter").Device(DEVICE_GPU).TypeConstraint<T>("T"), BilateralFilterOp<GPUDevice, T, SPATIAL_DIMS + REFERENCE_CHANNELS, INPUT_CHANNELS + 1>);
#define REGISTER_GAUSSIAN_GPU(T) REGISTER_KERNEL_BUILDER(Name("GaussianFilter").Device(DEVICE_GPU).TypeConstraint<T>("T"), GaussianFilterOp<GPUDevice, T, SPATIAL_DIMS, INPUT_CHANNELS + 1>);

REGISTER_BILATERAL_GPU(float);
REGISTER_BILATERAL_GPU(double);
REGISTER_GAUSSIAN_GPU(float);
REGISTER_GAUSSIAN_GPU(double);


#endif  // GOOGLE_CUDASPATIAL_DIMS + REFERENCE_CHANNELS, INPUT_CHANNELS + 1 ; SPATIAL_DIMS , INPUT_CHANNELS + 1