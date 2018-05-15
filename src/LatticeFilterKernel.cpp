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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

REGISTER_OP("LatticeFilter")
        .Attr("T: {float32, float64}")
        .Attr("reverse: bool = false")
        .Attr("bilateral: bool = true")
        .Attr("theta_alpha: float = 1.0")
        .Attr("theta_beta: float = 1.0")
        .Attr("theta_gamma: float = 1.0")
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
template<typename T>
struct ComputeKernel<CPUDevice, T>{
    void operator()(const CPUDevice& d,
                    OpKernelContext* context,
                    const T *reference_image,
                    T * positions,
                    int num_super_pixels,
                    int n_spatial_dims,
                    int *spatial_dims,
                    int n_reference_channels,
                    T spatial_std,
                    T features_std){
        compute_kernel_cpu<T>(reference_image, positions, num_super_pixels, n_reference_channels, n_spatial_dims,
                              spatial_dims, spatial_std, features_std);
    }
};

template <typename T>
struct LatticeFilter<CPUDevice, T>{
    void operator()(const CPUDevice& d,
                    OpKernelContext* context,
                    T* output,
                    const T *input,
                    const T *positions,
                    int num_super_pixels,
                    int pd,
                    int vd,
                    bool reverse){
        auto lattice = PermutohedralLatticeCPU<T>(pd, vd, num_super_pixels);
        lattice.filter(output, input, positions, reverse);
    }
};


// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class LatticeFilterOp : public OpKernel {
public:
    explicit LatticeFilterOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("reverse", &reverse));
        OP_REQUIRES_OK(context, context->GetAttr("bilateral", &bilateral));
        OP_REQUIRES_OK(context, context->GetAttr("theta_alpha", &theta_alpha));
        OP_REQUIRES_OK(context, context->GetAttr("theta_beta", &theta_beta));
        OP_REQUIRES_OK(context, context->GetAttr("theta_gamma", &theta_gamma));
    }

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& input_tensor = context->input(0);
        const Tensor& reference_image_tensor = context->input(1);

        // Create an output tensor
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));

        // Do the computation.
        OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                    errors::InvalidArgument("Too many elements in tensor"));

        // calculate dimensions; dimension 0 is batch; last dimension is channel
        int rank = input_tensor.dims();
        int n_spatial_dims = rank - 2;

        auto batch_size = static_cast<int>(input_tensor.dim_size(0));
        auto n_input_channels = static_cast<int>(input_tensor.dim_size(rank - 1));
        auto spatial_dims = new int[n_spatial_dims];

        int num_super_pixels{1};
        for (int i = 0; i < n_spatial_dims; i++){
            auto dim_size = static_cast<int>(input_tensor.dim_size(i + 1));
            num_super_pixels *= dim_size;
            spatial_dims[i] = dim_size;
        }

        vd = n_input_channels + 1;
        T spatial_std;
        T features_std;
        int n_reference_channels;

        if(bilateral){
            assert(reference_image_tensor.dims() == rank);
            n_reference_channels = static_cast<int>(reference_image_tensor.dim_size(rank - 1));
            pd = n_reference_channels + n_spatial_dims;
            spatial_std = theta_alpha;
            features_std = theta_beta;
        }else{
            pd = n_spatial_dims;
            n_reference_channels = 0; //set to zero so ComputeKernel does not use reference image channels
            spatial_std = theta_gamma;
            features_std = -1; //does not matter
        }

        // Allocate kernel positions and calculate them
        Tensor positions;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(),
                                                       TensorShape({batch_size * num_super_pixels * pd}),
                                                       &positions));

        auto allocator = DeviceMemoryAllocator(context);

        for(int b=0; b < batch_size; b++){

            auto ref_ptr = &(reference_image_tensor.flat<T>().data()[b * num_super_pixels * n_reference_channels]);
            auto pos_ptr = &(positions.flat<T>().data()[b * num_super_pixels * pd]);
            auto in_ptr = &(input_tensor.flat<T>().data()[b * num_super_pixels * n_input_channels]);
            auto out_ptr = &(output_tensor->flat<T>().data()[b * num_super_pixels * n_input_channels]);

            ComputeKernel<Device, T>()(context->eigen_device<Device>(),
                                       context,
                                       ref_ptr,
                                       pos_ptr,
                                       num_super_pixels,
                                       n_spatial_dims,
                                       spatial_dims,
                                       n_reference_channels,
                                       spatial_std,
                                       features_std);

            LatticeFilter<Device, T>()(context->eigen_device<Device>(),
                                       context,
                                       out_ptr,
                                       in_ptr,
                                       pos_ptr,
                                       num_super_pixels,
                                       pd,
                                       vd,
                                       reverse);
        }
        delete[](spatial_dims);
    }

private:
    bool reverse;
    bool bilateral;
    float theta_alpha;
    float theta_beta;
    float theta_gamma;
    int pd;
    int vd;
};

// Register the CPU kernels.
#define REGISTER_CPU(T) REGISTER_KERNEL_BUILDER(Name("LatticeFilter").Device(DEVICE_CPU).TypeConstraint<T>("T"), LatticeFilterOp<CPUDevice, T>);

REGISTER_CPU(float);
REGISTER_CPU(double);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
/* Declare explicit instantiations in kernel_example.cu.cc. */
extern template struct LatticeFilter<GPUDevice, float>;
extern template struct LatticeFilter<GPUDevice, double>;

#define REGISTER_GPU(T) REGISTER_KERNEL_BUILDER(Name("LatticeFilter").Device(DEVICE_GPU).TypeConstraint<T>("T"), LatticeFilterOp<GPUDevice, T>);

REGISTER_GPU(float);
REGISTER_GPU(double);
#endif  // GOOGLE_CUDA