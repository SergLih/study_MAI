# makefile
CC = g++
FLAGS = -pedantic -Wall -std=c++11 -Werror -Wno-sign-compare -lm

all: da9 benchmark

da9: da9.o  
	$(CC) $^ -o $@

benchmark: benchmark.o
	$(CC) $^ -o $@

.cpp.o:
	$(CC) $(FLAGS) -c $< -o $@
	
clean:
	rm -f da9 *.o 
