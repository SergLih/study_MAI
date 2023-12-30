# makefile
CC = g++
FLAGS = -std=c++11 -Wall -Werror -Wno-sign-compare -Wno-unused-result

all: run

run: main.o tbinary_tree.o pentagon.o 
	$(CC) $^ -o $@

.cpp.o:
	$(CC) $(FLAGS) -c $< -o $@
clean:
	rm -f main *.o
