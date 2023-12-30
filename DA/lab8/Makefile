# makefile
CC = g++
FLAGS = -pedantic -Wall -std=c++11 -Werror -Wno-sign-compare -lm

all: da8 benchmark

da7: da8.o  
	$(CC) $^ -o $@

benchmark: benchmark.o
	$(CC) $^ -o $@

.cpp.o:
	$(CC) $(FLAGS) -c $< -o $@
	
clean:
	rm -f da8 *.o 
