//
// Created by Miguel Monteiro on 29/01/2018.
//

#include "BilateralKernel.h"
#include "PermutohedralLatticeCPU.h"
#include "cstdio"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Bilateral")
        .Attr("T: {float32}")
        .Input("input_image: T")
        .Input("reference_image: T")
        .Output("output: T")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            c->set_output(0, c->input(0));
            return Status::OK();
        });


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU specialization of actual computation.
template <typename T>
struct ExampleFunctor<CPUDevice, T> {
    void operator()(const CPUDevice& d,
                    T * output,
                    const T *input,
                    const T *reference_image,
                    int num_super_pixels,
                    int n_spatial_dims,
                    int *spatial_dims,
                    int n_input_channels,
                    int n_reference_channels,
                    float theta_alpha,
                    float theta_beta) {

        int pd = n_reference_channels + n_spatial_dims;
        int vd = n_input_channels + 1;
        int n = num_super_pixels;

        auto positions= new float[num_super_pixels * pd];

        compute_bilateral_kernel_cpu(reference_image,
                                     positions,
                                     num_super_pixels,
                                     n_reference_channels,
                                     n_spatial_dims,
                                     spatial_dims,
                                     theta_alpha,
                                     theta_beta);

        lattice_filter_cpu(output, input, positions, pd, vd, n);

        delete[] positions;
    }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class BilateralOp : public OpKernel {
public:
    explicit BilateralOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& input_tensor = context->input(0);
        const Tensor& image_tensor = context->input(1);

        // calculate dimensions; assumes channel is last dimension
        int rank = input_tensor.dims();
        int n_spatial_dims = rank -1;
        auto input_channels = static_cast<int>(input_tensor.dim_size(n_spatial_dims));

        auto spatial_dims = new int[rank-1];

        int num_super_pixels{1};
        for (int i = 0; i < n_spatial_dims; i++){
            num_super_pixels *= input_tensor.dim_size(i);
            spatial_dims[i] = static_cast<int>(input_tensor.dim_size(i));
        }

        assert(image_tensor.dims() ==  rank);
        auto ref_channels = static_cast<int>(image_tensor.dim_size(n_spatial_dims));

        float theta_alpha{8};
        float theta_beta{0.125};
        //int spatial_dims[2]{1000,800};

        //int* spatial_dims = input_tensor.shape().dim_sizes().data();
        int n_input_channels = input_tensor.dim_size(rank-1);
        int n_reference_channels = image_tensor.dim_size(rank-1);

        // Create an output tensor
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
        //OP_REQUIRES_OK(context, context->set_output(0, &input_tensor));
        // Do the computation.
        OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max, errors::InvalidArgument("Too many elements in tensor"));


        ExampleFunctor<Device, T>()(context->eigen_device<Device>(),
                                    output_tensor->flat<T>().data(),
                                    input_tensor.flat<T>().data(),
                                    image_tensor.flat<T>().data(),
                                    num_super_pixels,
                                    n_spatial_dims,
                                    spatial_dims,
                                    n_input_channels,
                                    n_reference_channels,
                                    theta_alpha,
                                    theta_beta);
        delete[](spatial_dims);
    }
};

// Register the CPU kernels.
#define REGISTER_CPU(T) REGISTER_KERNEL_BUILDER(Name("Bilateral").Device(DEVICE_CPU).TypeConstraint<T>("T"), BilateralOp<CPUDevice, T>);

REGISTER_CPU(float);
//REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
/* Declare explicit instantiations in kernel_example.cu.cc. */
#define REGISTER_GPU(T) extern template ExampleFunctor<GPUDevice, float>; REGISTER_KERNEL_BUILDER(Name("Bilateral").Device(DEVICE_GPU).TypeConstraint<T>("T"), ExampleOp<GPUDevice, T>); REGISTER_GPU(float);
//REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA