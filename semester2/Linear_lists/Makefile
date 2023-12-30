CC = gcc
CFLAGS = -Wall -Wextra -g -std=c99 -O3

all:  main

main:	main.o list.o iterator.o 
	$(CC) $^ -o $@

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f main *.o
