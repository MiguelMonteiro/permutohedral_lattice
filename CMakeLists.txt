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

execute_process(COMMAND python2 -c "import tensorflow as tf;  print(tf.sysconfig.get_include())" OUTPUT_VARIABLE TF_INC)
execute_process(COMMAND python2 -c "import tensorflow as tf; print(tf.sysconfig.get_lib())" OUTPUT_VARIABLE TF_LIB)
string(REGEX REPLACE "\n$" "" TF_INC "${TF_INC}")
string(REGEX REPLACE "\n$" "" TF_LIB "${TF_LIB}")
include_directories(${TF_INC} ${TF_INC}/external/nsync/public)
link_directories(${TF_LIB})

if (MAKE_TESTS)
    include(CMakeTests)
endif ()


add_library(lattice_filter SHARED src/LatticeFilterKernel.cpp src/LatticeFilterKernel.cu src/LatticeFilterKernel.h src/PermutohedralLatticeCPU.h src/PermutohedralLatticeGPU.cuh src/DeviceMemoryAllocator.h)
target_compile_features(lattice_filter PUBLIC cxx_std_11)
target_link_libraries(lattice_filter PUBLIC tensorflow_framework)
set_target_properties(lattice_filter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(lattice_filter PROPERTIES PREFIX "")
target_compile_options(lattice_filter PUBLIC "-D GOOGLE_CUDA=1" "-D GLIBCXX_USE_CXX11_ABI=0")
target_compile_options(lattice_filter PUBLIC "-D SPATIAL_DIMS=${SPATIAL_DIMS}")
target_compile_options(lattice_filter PUBLIC "-D INPUT_CHANNELS=${INPUT_CHANNELS}")
target_compile_options(lattice_filter PUBLIC "-D REFERENCE_CHANNELS=${REFERENCE_CHANNELS}")
if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0)
    target_compile_options(lattice_filter PUBLIC "-D_GLIBCXX_USE_CXX11_ABI=0")
endif()





