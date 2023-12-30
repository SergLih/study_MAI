GCC = mpic++ --std=c++11
GCCFLAGS = -fopenmp -pedantic -Wall -Werror -Wno-sign-compare
FLAGS = -Wno-long-long -lm
MAIN_FILE = Lih_lab9.cpp
FILES = 
PROG = lab9

all:
	@$(GCC) $(GCCFLAGS) -o $(PROG) $(MAIN_FILE) $(FLAGS) $(FILES)

clean:
	@rm -rf $(PROG)