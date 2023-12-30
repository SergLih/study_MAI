#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>  // для этого компилируем с параметром -ldl
#include "vector_interface.h"

void instructions(void)
{
    printf("For assistance in using the program, write 'h':\n"
           "c <capacity>       create the vector with a given <capacity>\n"
           "a <item>           append <item> to the vector\n"
           "p                  print the vector contents\n"
           "q                  exit from the program\n");
}

int main()
{
    TVector *v = NULL;
    
    int code;
    TItem tmp;
    void *libHandle;
    printf("Welcome to the dynamic-linked program.\nPlease choose: 1 - fixed-size, 2 - flexible-size vectors: ");
    scanf("%d", &code);
    if(code == 1) {
        libHandle = dlopen("libvector_st.so", RTLD_LAZY);
        if (!libHandle) {
            fprintf(stderr, "%s\n", dlerror());
            exit(FAILURE);
        }
    } else {
        libHandle = dlopen("libvector_ext.so", RTLD_LAZY);
        if (!libHandle) {
            fprintf(stderr, "%s\n", dlerror());
            exit(FAILURE);
        }
    }
    vector_create_t  * VectorCreate  =  dlsym(libHandle, "VectorCreate");
    vector_append_t  * VectorAppend  =  dlsym(libHandle, "VectorAppend");
    vector_print_t   * VectorPrint   =  dlsym(libHandle, "VectorPrint");
    vector_destroy_t * VectorDestroy =  dlsym(libHandle, "VectorDestroy");
    
    char *err;
    if((err = dlerror())) {
        fprintf(stderr, "%s\n", err);
        exit(FAILURE);
    }
    
    instructions();
    while ((code =  getchar()) != EOF) {
        switch (code) {
        	case 'h':
        		instructions();
        		break;
            case 'q':
                printf("Now the program will be closed\n");
                VectorDestroy(&v);
                return 0;
            case 'p':
                VectorPrint(v);
                break;
            case 'a':
                scanf("%d", &tmp);
                VectorAppend(v, tmp);
                break;
            case 'c':
                scanf("%d", &tmp);
                if(v)
                    VectorDestroy(&v);
                v = VectorCreate(tmp);
                break;  
        }
    }
    VectorDestroy(&v);
    return 0;
}
