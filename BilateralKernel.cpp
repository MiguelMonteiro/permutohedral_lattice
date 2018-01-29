//
// Created by Miguel Monteiro on 29/01/2018.
//

#include "BilateralKernel.h"
#include "PermutohedralLatticeCPU.h"

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// CPU specialization of actual computation.
template <typename T>
struct ExampleFunctor<CPUDevice, T> {
    void operator()(const CPUDevice& d, int ref_channels, int input_channels, int num_super_pixels, const T* input, T* image) {

        auto reference = compute_kernel(image);
        filter_cpu(input, reference, ref_channels, input_channels, num_super_pixels);
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
        auto input_channels = static_cast<int>(input_tensor.dim_size(rank - 1));

        int num_super_pixels{1};
        for (int i = 0; i < rank - 1; i++)
            num_super_pixels *= input_tensor.dim_size(i);

        assert(image_tensor.dims() ==  rank);
        auto ref_channels = static_cast<int>(image_tensor.dim_size(rank - 1));



        // Create an output tensor
        //Tensor* output_tensor = nullptr;
        //OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));

        // Do the computation.
        OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max, errors::InvalidArgument("Too many elements in tensor"));

        ExampleFunctor<Device, T>()(
                context->eigen_device<Device>(), ref_channels, input_channels, num_super_pixels, input_tensor.flat<T>().data(), image_tensor.flat<T>().data());
    }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Bilateral").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      BilateralOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
  extern template ExampleFunctor<GPUDevice, float>;              \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Example").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ExampleOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA