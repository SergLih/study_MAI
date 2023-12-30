# makefile
CC = gcc
FLAGS = -Wextra -Wall -Werror -pedantic -std=c99


all: run


run:  queue.o main.o
	$(CC) $(FLAGS) $^ -o $@


runlib: 
	$(CC) $(FLAGS) -c queue.c
	ar crs -D ./lib/libqueue_l.a queue.o
	$(CC) $(FLAGS) main.c -L./lib -lqueue_l -o queue
	

runlib2:
	$(CC) $(FLAGS) main.c -L./lib -lqueue_s -o queue


clean:
	rm -f queue *.o
