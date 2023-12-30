CC = g++
FLAGS = -std=c++11 -pedantic -Wall -Werror -Wno-sign-compare -Wno-long-long -lm -O2 -g

all: da6

da6: longint.o main.o
	$(CC) $(FLAGS) $^ -o $@
.cpp.o:
	$(CC) $(FLAGS) -c $< -o $@

clean:
	rm -rf *.o da6
