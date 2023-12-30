#include <stdio.h>
#include <stdlib.h>
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
    printf("Welcome to the static-linked program. You can use fixed-size vectors.\n");
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
