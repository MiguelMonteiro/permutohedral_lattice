##Permutohedral Lattice Tensorflow OP

This code implements the Permutohedral Lattice  for high dimensional filtering.
The code contatains:
- A CPU implementation (C++);
- A GPU implementation (C++/CUDA);
- TensorFlow Op Kernels that wrap the CPU and GPU implementations to be used in Python/TensorFlow;

This code can be used to perform (approximate) bilateral filtering, gaussian filtering, non-local means etc...
It can be used as part of larger algorithms such as Conditional Random Fields.


<img src="Images/input.bmp" width="400"> | <img src="Images/output.bmp" width="400"> 
 
#### How to compile and use

1. Install CMake (version >= 3.9).

2. Open the file `build.sh` and change the variables `CXX_COMPILER` and `CUDA_COMPILER` to the path of the C++ and nvcc
 (CUDA) compilers on your machine.
 
3. To compile the code run:
````
sh build.sh
````
This will create a directory called `build_dir` which will contain the compiled code.

###### Caveats

This script will try to compile code for both CPU and GPU at the same time, so if you don't want the GPU part
 (and want the script to run) you must change `CMakeLists.txt`.
 
Because of the way the GPU (CUDA) code is implemented, the number of spatial dimensions and number of channels of
 the input and reference images must be known at compile time. This can be changed in the `build.sh` script as well by
  changing the variables `SPATIAL_DIMS`, `INPUT_CHANNELS` and `REFERENCE_CHANNELS`.
  If you only need the CPU version this variables do nothing to it and these values can be run-time values.
            


#### Example Usage

##### CPU C++
````
./build_dir/test_bilateral_cpu Images/input.bmp Images/output.bmp 8 0.125
````
##### GPU C++/CUDA

````
./build_dir/test_bilateral_gpu Images/input.bmp Images/output.bmp 8 0.125
````

##### TensorFlow Python



#### Known Issues

1. The GPU version must know `SPATIAL_DIMS`, `INPUT_CHANNELS` and `REFERENCE_CHANNELS` at run time.
2. The TensorFlow CPU and GPU Op kernels don't play well with each other, even though they both work individually.
I have not been able to figure out how to choose between them in python, and if both are ON the CPU one is always chosen
 as default. As a result, in the file `LatticeFilterKernel.cpp` the registering of the CPU kernel is commented.
 You can un-comment it to use it.