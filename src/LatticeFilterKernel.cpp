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

// ignore pd and vd, see kernel registration for why
template <typename T, int pd, int vd>
struct LatticeFilter<CPUDevice, T, pd, vd> {

    LatticeFilter(OpKernelContext *context,
                  int num_hypervoxels,
                  int num_input_channels,
                  int num_spatial_dims,
                  int num_reference_channels,
                  int *host_spatial_dims,
                  const T *spatial_std,
                  const T *color_std,
                  bool reverse) : num_hypervoxels(num_hypervoxels),
                                  num_input_channels(num_input_channels),
                                  num_spatial_dims(num_spatial_dims),
                                  num_reference_channels(num_reference_channels),
                                  spatial_dims(host_spatial_dims),
                                  spatial_std(spatial_std),
                                  color_std(color_std),
                                  reverse(reverse) {
        int pd_ = num_reference_channels + num_spatial_dims;
        position_vectors = new T[num_hypervoxels * pd_];
    }
    ~LatticeFilter(){
        delete[](position_vectors);
    }

    void operator()(T* output_image, const T *input_image, const T *reference_image) {
        int pd_ = num_reference_channels + num_spatial_dims;
        int vd_ = num_input_channels + 1;

        compute_position_vectors(reference_image,
                                 position_vectors,
                                 num_hypervoxels,
                                 num_reference_channels,
                                 num_spatial_dims,
                                 spatial_dims,
                                 spatial_std,
                                 color_std);

        auto lattice = PermutohedralLatticeCPU<T>(pd_, vd_, num_hypervoxels);
        lattice.filter(output_image, input_image, position_vectors, reverse);
    }

private:
    int num_hypervoxels;
    int num_input_channels;
    int num_spatial_dims;
    int num_reference_channels;
    int *spatial_dims{};
    const T* spatial_std;
    const T* color_std;
    bool reverse;
    T * position_vectors;

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

        // calculate dimensions; dimension 0 is batch; last dimension is channel
        auto batch_size = static_cast<int>(input_image_tensor.dim_size(0));
        assert(batch_size == reference_image_tensor.shape().dim_size(0));
        int rank = input_image_tensor.dims();
        auto num_input_channels = static_cast<int>(input_image_tensor.dim_size(rank - 1));
        auto num_reference_channels = static_cast<int>(reference_image_tensor.dim_size(rank - 1));
        auto num_hypervoxels = static_cast<int>(input_image_tensor.shape().num_elements()) / (num_input_channels * batch_size);
        assert(num_hypervoxels == reference_image_tensor.shape().num_elements() / (num_reference_channels * batch_size))
        int num_spatial_dims = rank - 2;
        auto spatial_dims = new int[num_spatial_dims];
        for (int i = 0; i < num_spatial_dims; i++)
            spatial_dims[i] = static_cast<int>(input_image_tensor.dim_size(i + 1));

        const Tensor& theta_spatial = context->input(2);
        assert(theta_spatial.dims() == 0);
        const Tensor& theta_color = context->input(3);
        assert(theta_color.dims() == 0);
        const T* spatial_std = theta_spatial.flat<T>().data();
        const T* color_std = theta_color.flat<T>().data();

        // Create an output tensor
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_image_tensor.shape(), &output_tensor));

        auto lattice = LatticeFilter<Device, T, pd, vd>(context,
                                                        num_hypervoxels,
                                                        num_input_channels,
                                                        num_spatial_dims,
                                                        num_reference_channels,
                                                        spatial_dims,
                                                        spatial_std,
                                                        color_std,
                                                        reverse);
        for(int b=0; b < batch_size; b++){
            auto ref_ptr = reference_image_tensor.flat<T>().data() +(b * num_hypervoxels * num_reference_channels);
            auto in_ptr = input_image_tensor.flat<T>().data() + (b * num_hypervoxels * num_input_channels);
            auto out_ptr = output_tensor->flat<T>().data() + (b * num_hypervoxels * num_input_channels);
            lattice(out_ptr, in_ptr, ref_ptr);
        }
        delete[](spatial_dims);
    }
private:
    bool reverse;
};


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

        // calculate dimensions; dimension 0 is batch; last dimension is channel
        auto batch_size = static_cast<int>(input_image_tensor.dim_size(0));
        int rank = input_image_tensor.dims();
        auto num_input_channels = static_cast<int>(input_image_tensor.dim_size(rank - 1));
        auto num_hypervoxels = static_cast<int>(input_image_tensor.shape().num_elements());
        int num_spatial_dims = rank - 2;
        auto spatial_dims = new int[num_spatial_dims];
        for (int i = 0; i < num_spatial_dims; i++)
            spatial_dims[i] = static_cast<int>(input_image_tensor.dim_size(i + 1));

        const Tensor& theta_spatial = context->input(1);
        assert(theta_spatial.dims() == 0);
        const T *spatial_std = theta_spatial.flat<T>().data();

        // Create an output tensor
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_image_tensor.shape(), &output_tensor));

        auto lattice = LatticeFilter<Device, T, pd, vd>(context,
                                                        num_hypervoxels,
                                                        num_input_channels,
                                                        num_spatial_dims,
                                                        0,
                                                        spatial_dims,
                                                        spatial_std,
                                                        nullptr,
                                                        reverse);

        for(int b=0; b < batch_size; b++){
            auto in_ptr = input_image_tensor.flat<T>().data() + (b * num_hypervoxels * num_input_channels);
            auto out_ptr = output_tensor->flat<T>().data() + (b * num_hypervoxels * num_input_channels);
            lattice(out_ptr, in_ptr, nullptr);
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

/* Register the CPU kernels. pd and vd don't matter for CPU implementation because memory allocation is dynamic so
 * register only one kernel each with <CPUDevice, 0, 0> to avoid compiling the code multiple times */
#define REGISTER_BILATERAL_CPU(T) REGISTER_KERNEL_BUILDER(Name("BilateralFilter").Device(DEVICE_CPU).TypeConstraint<T>("T"), BilateralFilterOp<CPUDevice, T, 0, 0>);
#define REGISTER_GAUSSIAN_CPU(T) REGISTER_KERNEL_BUILDER(Name("GaussianFilter").Device(DEVICE_CPU).TypeConstraint<T>("T"), GaussianFilterOp<CPUDevice, T, 0, 0>);

REGISTER_BILATERAL_CPU(float);
REGISTER_BILATERAL_CPU(double);
REGISTER_GAUSSIAN_CPU(float);
REGISTER_GAUSSIAN_CPU(double);


// Register the GPU kernels.
#ifdef GOOGLE_CUDA

#define REGISTER_BILATERAL_GPU(T, PD, VD) REGISTER_KERNEL_BUILDER(Name("BilateralFilter").Device(DEVICE_GPU).TypeConstraint<T>("T"), BilateralFilterOp<GPUDevice, T, PD, VD>);
#define REGISTER_GAUSSIAN_GPU(T, PD, VD) REGISTER_KERNEL_BUILDER(Name("GaussianFilter").Device(DEVICE_GPU).TypeConstraint<T>("T"), GaussianFilterOp<GPUDevice, T, PD, VD>);

//Bilateral Filter
REGISTER_BILATERAL_GPU(float, SPATIAL_DIMS + REFERENCE_CHANNELS, INPUT_CHANNELS + 1);
REGISTER_BILATERAL_GPU(double, SPATIAL_DIMS + REFERENCE_CHANNELS, INPUT_CHANNELS + 1);
//Bilateral Filter theta grads
REGISTER_BILATERAL_GPU(float, SPATIAL_DIMS + REFERENCE_CHANNELS, REFERENCE_CHANNELS + 1);
REGISTER_BILATERAL_GPU(float, SPATIAL_DIMS + REFERENCE_CHANNELS, REFERENCE_CHANNELS + 1);
REGISTER_BILATERAL_GPU(double, SPATIAL_DIMS + REFERENCE_CHANNELS, SPATIAL_DIMS + 1);
REGISTER_BILATERAL_GPU(double, SPATIAL_DIMS + REFERENCE_CHANNELS, SPATIAL_DIMS + 1);
//Gaussian Filter
REGISTER_GAUSSIAN_GPU(float, SPATIAL_DIMS, INPUT_CHANNELS + 1);
REGISTER_GAUSSIAN_GPU(double, SPATIAL_DIMS, INPUT_CHANNELS + 1);
//Gaussian Filter theta grads
REGISTER_GAUSSIAN_GPU(float, SPATIAL_DIMS, SPATIAL_DIMS + 1);
REGISTER_GAUSSIAN_GPU(double, SPATIAL_DIMS, SPATIAL_DIMS + 1);


#endif  // GOOGLE_CUDA SPATIAL_DIMS + REFERENCE_CHANNELS, INPUT_CHANNELS + 1 ; SPATIAL_DIMS , INPUT_CHANNELS + 1