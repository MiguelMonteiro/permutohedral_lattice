cmake_minimum_required(VERSION 3.9)

project(permutohedral_lattice LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CUDA_SEPARABLE_COMPILATION ON)

add_executable(bilateral_cpu bilateral_cpu.cpp PermutohedralLattice.cpp PermutohedralLattice.h CImg.h utils.h)
if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()
if (LINUX)
    find_package(Threads REQUIRED)
    target_link_libraries(bilateral_cpu PRIVATE ${CMAKE_THREAD_LIBS_INIT})
endif ()

add_library(permutohedral STATIC permutohedral.cu hash_table.cu MirroredArray.h cuda_memory.h cuda_code_indexing.h)

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to
# particles will also build with -std=c++11
target_compile_features(permutohedral PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the
# particle library to be built with -dc as the member functions
# could be called by other libraries and executables
set_target_properties(permutohedral PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#include_directories(/Developer/NVIDIA/CUDA-9.1/include)
add_executable(bilateral_gpu bilateral_gpu.cpp CImg.h utils.h)

set_property(TARGET bilateral_gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(bilateral_gpu PRIVATE permutohedral)

if(APPLE)
    # We need to add the path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET bilateral_gpu PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

##

add_executable(compare_and_gpu compare_and_gpu.cpp CImg.h utils.h PermutohedralLattice.h PermutohedralLattice.cpp)

set_property(TARGET compare_and_gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(compare_and_gpu PRIVATE permutohedral)

if(APPLE)
    # We need to add the path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET compare_and_gpu PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
