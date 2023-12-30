FLAGSP = -pedantic -Wall -Werror -Wno-sign-compare -lm
FLAGS = -Wall -Werror -Wno-sign-compare -lm -pthread
COMPILLER=gcc

all: start

start: libvector_st.so libvector_ext.so main.o main-dynam.o
	$(COMPILLER) $(FLAGSP) -o os-lab5 main.o -L. -lvector_st -Wl,-rpath,.
	$(COMPILLER) $(FLAGS) -o os-lab5-dynam main-dynam.o -ldl

main.o: main.c
	$(COMPILLER) -c $(FLAGSP) main.c

main-dynam.o: main-dynam.c
	$(COMPILLER) -c $(FLAGS) main-dynam.c

vector_st.o: vector_st.c
	$(COMPILLER) -c -fPIC $(FLAGSP) vector_st.c
	
vector_ext.o: vector_ext.c
	$(COMPILLER) -c -fPIC $(FLAGSP) vector_ext.c

libvector_st.so: vector_st.o
	$(COMPILLER) $(FLAGSP) -shared -o libvector_st.so vector_st.o
	
libvector_ext.so: vector_ext.o
	$(COMPILLER) $(FLAGSP) -shared -o libvector_ext.so vector_ext.o

clear:
	-rm -f *.o *.gch *.so os-lab5
