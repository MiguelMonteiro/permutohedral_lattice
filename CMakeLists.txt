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

set(LatticeTests True)

if (LatticeTests)
    include(CMakeTests)
endif ()


#[[]]
add_library(cuda_lattice_filter SHARED LatticeFilterKernel.cu LatticeFilterKernel.h)
target_compile_features(cuda_lattice_filter PUBLIC cxx_std_11)
set_target_properties(cuda_lattice_filter  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#target_compile_options(cuda_bilateral PUBLIC "-D GOOGLE_CUDA=1" "--expt-relaxed-constexpr")
target_compile_options(cuda_lattice_filter PUBLIC "-D GOOGLE_CUDA=1")

add_library(lattice_filter SHARED LatticeFilterKernel.cpp LatticeFilterKernel.h PermutohedralLatticeCPU.h)
target_compile_options(lattice_filter PUBLIC "-D GOOGLE_CUDA=1")
target_link_libraries(lattice_filter PUBLIC tensorflow_framework cuda_lattice_filter)
set_target_properties(lattice_filter PROPERTIES PREFIX "")

#if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0)
#    target_compile_options(bilateral PUBLIC "-D GLIBCXX_USE_CXX11_ABI=0")
#endif()


#target_include_directories(cuda_kernel.o INTERFACE ${EIGEN3_INCLUDE_DIR} PRIVATE ${TF_INC} ${TF_INC}/external/nsync/public)
#link_directories(${TF_LIB})




####


# CPU executable

#if (LINUX)
#    find_package(Threads REQUIRED)
#    target_link_libraries(bilateral_cpu PRIVATE tensorflow_framework ${CMAKE_THREAD_LIBS_INIT}) #needed for CImg
#else()
#    target_link_libraries(bilateral_cpu PRIVATE tensorflow_framework)
#endif ()


