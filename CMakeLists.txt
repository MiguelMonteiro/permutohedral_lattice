cmake_minimum_required(VERSION 3.9)

project(permutohedral_lattice LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CUDA_SEPARABLE_COMPILATION ON)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

#just needed for Clion code indexing
include_directories(/usr/local/cuda/include/)

#tensorflow
#execute_process(COMMAND bash -c "python2 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'" OUTPUT_VARIABLE TF_INC)
#execute_process(COMMAND bash -c "python2 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())'" OUTPUT_VARIABLE TF_LIB)

execute_process(COMMAND python2 -c "import tensorflow as tf;  print(tf.sysconfig.get_include())" OUTPUT_VARIABLE TF_INC)
execute_process(COMMAND python2 -c "import tensorflow as tf; print(tf.sysconfig.get_lib())" OUTPUT_VARIABLE TF_LIB)
string(REGEX REPLACE "\n$" "" TF_INC "${TF_INC}")
string(REGEX REPLACE "\n$" "" TF_LIB "${TF_LIB}")

include_directories(${TF_INC})
include_directories(${TF_INC}/external/nsync/public)
link_directories(${TF_LIB})


# CPU Test executable
add_executable(bilateral_cpu_test
        Test/bilateral_cpu_test.cpp
        Test/bilateral_filter_cpu.h
        Test/CImg.h
        Test/utils.h
        PermutohedralLatticeCPU.cpp
        PermutohedralLatticeCPU.h)

if (LINUX)
    find_package(Threads REQUIRED)
    target_link_libraries(bilateral_cpu_test PRIVATE ${CMAKE_THREAD_LIBS_INIT})
endif ()



# CUDA Library
add_library(permutohedral STATIC PermutohedralLatticeGPU.cu)

# Request that permutohedral be built with -std=c++11
# As this is a public compile feature anything that links to
# particles will also build with -std=c++11
target_compile_features(permutohedral PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the
# permutohedral library to be built with -dc as the member functions
# could be called by other libraries and executables
set_target_properties(permutohedral PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# GPU test executable
add_executable(bilateral_gpu_test
        Test/bilateral_gpu_test.cpp
        Test/bilateral_filter_gpu.h
        Test/CImg.h
        Test/utils.h)


set_property(TARGET bilateral_gpu_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(bilateral_gpu_test PRIVATE permutohedral)

if(APPLE)
    # We need to add the path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET bilateral_gpu_test PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()


# CPU and GPU compare executable

add_executable(compare_and_gpu_test
        Test/compare_and_gpu_test.cpp
        Test/bilateral_filter_cpu.h
        Test/bilateral_filter_gpu.h
        Test/CImg.h
        Test/utils.h
        PermutohedralLatticeCPU.h
        PermutohedralLatticeCPU.cpp)

set_property(TARGET compare_and_gpu_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(compare_and_gpu_test PRIVATE permutohedral)

if(APPLE)
    # We need to add the path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET compare_and_gpu_test PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()



####

add_executable(cuda_kernel.o BilateralKernel.cu PermutohedralLatticeGPU.cu BilateralKernel.h)
target_compile_options(cuda_kernel.o PUBLIC "-D GOOGLE_CUDA=1")

####

# CPU executable

#if (LINUX)
#    find_package(Threads REQUIRED)
#    target_link_libraries(bilateral_cpu PRIVATE tensorflow_framework ${CMAKE_THREAD_LIBS_INIT}) #needed for CImg
#else()
#    target_link_libraries(bilateral_cpu PRIVATE tensorflow_framework)
#endif ()


