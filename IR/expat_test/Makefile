# makefile
CC = g++
FLAGS = -I cparse -pedantic -Wall -std=c++14 -Werror -Wno-sign-compare -lm

all: clean expat_test

#g++ -I cparse -std=c++11 cparse/builtin-features.o cparse/core-shunting-yard.o main.cpp -o main

expat_test: main.cpp index.cpp tstring.cpp /home/sergey/bin/cparse/builtin-features.o /home/sergey/bin/cparse/core-shunting-yard.o
	$(CC) $^ -o $(CLION_EXE_DIR)/$@

.cpp.o:
	$(CC) $(FLAGS) -c $< -o $@
	
clean:
	rm -f expat_test *.o
