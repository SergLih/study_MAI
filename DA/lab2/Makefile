# makefile
CC = g++
FLAGS = -pedantic -Wall -std=c++11 -Werror -Wno-sign-compare -lm

all: da2

da2: da2.o rbtree.o mystring.o  
	$(CC) $^ -o $@

.cpp.o:
	$(CC) $(FLAGS) -c $< -o $@
	
clean:
	rm -f da2 *.o 
	
