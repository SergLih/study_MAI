#NVCC = nvcc --std=c++11
GCC  = g++ --std=c++11
GCCFLAGS = -pedantic -Wall -Wno-sign-compare -pthread #-Werror
GPUFLAGS = -Wno-long-long -lm
#GPUFLAGS = -Werror cross-execution-space-call -lm
MAIN_FILE_CPU = main.cpp vector.hpp model.h model.cpp tracer.h sphere.h sphere.cpp tracer.cpp Matrix.hpp TexturePlane.cpp TexturePlane.h ssaa.h ssaa.cpp
PROG = cpu

all:
	@$(GCC) $(GCCFLAGS) -o $(PROG) $(MAIN_FILE_CPU)

clean:
	@rm -rf $(PROG)