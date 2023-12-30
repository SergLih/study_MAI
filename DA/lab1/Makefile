# makefile
CC = gcc
FLAGS = -pedantic -Wall -std=c99 -Werror -Wno-sign-compare -lm

all: da1 benchmark

da1: main.o vector.o  
	$(CC) $^ -o $@

benchmark: benchmark.o vector.o
	$(CC) $^ -o $@

.c.o:
	$(CC) $(FLAGS) -c $< -o $@
	
clean:
	rm -f main *.o 
