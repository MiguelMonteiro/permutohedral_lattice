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
include_directories(${TF_INC} ${TF_INC}/external/nsync/public)
link_directories(${TF_LIB})

set(TEST TRUE)

if (TEST)
    include(CMakeTests)
endif ()



####


#
# CUDA Library

#[[
add_library(cuda_bilateral SHARED BilateralKernel.cu BilateralKernel.h)
target_compile_features(cuda_bilateral PUBLIC cxx_std_11)
set_target_properties(cuda_bilateral  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#target_compile_options(cuda_bilateral PUBLIC "-D GOOGLE_CUDA=1" "--expt-relaxed-constexpr")
target_compile_options(cuda_bilateral PUBLIC "-D GOOGLE_CUDA=1")


add_library(bilateral SHARED BilateralKernel.cpp BilateralKernel.h PermutohedralLatticeCPU.cpp)
target_compile_options(bilateral PUBLIC "-D GOOGLE_CUDA=1")
target_link_libraries(bilateral PUBLIC tensorflow_framework cuda_bilateral)
set_target_properties(bilateral PROPERTIES PREFIX "")
#target_include_directories(cuda_kernel.o INTERFACE ${EIGEN3_INCLUDE_DIR} PRIVATE ${TF_INC} ${TF_INC}/external/nsync/public)
#link_directories(${TF_LIB})
]]



####


# CPU executable

#if (LINUX)
#    find_package(Threads REQUIRED)
#    target_link_libraries(bilateral_cpu PRIVATE tensorflow_framework ${CMAKE_THREAD_LIBS_INIT}) #needed for CImg
#else()
#    target_link_libraries(bilateral_cpu PRIVATE tensorflow_framework)
#endif ()


