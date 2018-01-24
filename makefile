
BUILDDIR=binaries

all: bilateral_cpu

bilateral_cpu: bilateral_cpu.cpp CImg.h PermutohedralLattice.cpp PermutohedralLattice.h
    $(CC) --std=c++11 -03 bilateral_cpu.cpp


#bilateral_gpu: bilateral_gpu.cpp CImg.h permutohedral.o
#	/usr/local/cuda/bin/nvcc --std=c++11 -O3 bilateral_gpu.cpp permutohedral.o -o bilateral

#permutohedral.o: permutohedral.cu hash_table.cu
#	/usr/local/cuda/bin/nvcc --std=c++11 -use_fast_math -O3 -DLIBRARY -c permutohedral.cu -o permutohedral.o

#clean:
#	rm bilateral_gpu permutohedral.o