all: client server
client:
	gcc -std=c99 -Wall -D_POSIX_SOURCE=600 -D_XOPEN_SOURCE=600 client.c -lzmq -o client 
server:
	gcc -std=c99 -Wall -D_POSIX_SOURCE=600 server.c vector.c -lzmq -o server
clear:
	rm server client
