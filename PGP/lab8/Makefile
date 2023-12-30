NVCC = nvcc -ccbin=mpic++ --std=c++11
#GCCFLAGS = -fopenmp -pedantic -Wall -Werror -Wno-sign-compare
FLAGS = -Werror cross-execution-space-call -lm
MAIN_FILE = Lih_lab8.cu
FILES = 
PROG = lab8

all:
	@$(NVCC) -o $(PROG) $(MAIN_FILE) $(FLAGS) $(FILES)
#$(GCCFLAGS)
clean:
	@rm -rf $(PROG)