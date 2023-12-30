# makefile
CC = g++
FLAGS = -pedantic -Wall -std=c++11 -Werror -Wno-sign-compare -lm

all: da4

da4: main.o ttrie.o  
	$(CC) $^ -o $@

.cpp.o:
	$(CC) $(FLAGS) -c $< -o $@
	
clean:
	rm -f main *.o 
