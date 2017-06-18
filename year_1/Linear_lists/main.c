#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "list.h"
#include "iterator.h"

#define STR_MAX_LEN 1000

void instructions(void)
{
    printf("For assistance in using the program, write 'h':\n"
           "p                  print the list.\n"
           "a <item>           append <item> to the list\n"
           "i <index> <item>   insert <item> into the list at position <index>.\n"
           "r <index>          remove element at <index> from the list.\n"
           "d <item>           delete element <item> from the list.\n"
           "l                  сounting the length of the list.\n"
           "e                  exchange 1st and 2nd halves\n");
}

int main()
{

    List *l = ListCreate();
    int code;
    int index;
    char buf[STR_MAX_LEN] = {0};
    instructions();
    while ((code =  getchar()) != EOF) {
        switch (code) {
        	case 'h':
        		instructions();
        		break;
            case 'q':
                ListDestroy(&l);
                return 0;
            case 'p':
                ListPrint(l);
                break;
            case 'a':
                //printf("add: ");
                scanf("%s", buf);
                ListAddNode(l, buf);
                break;
            case 'i':
                if (scanf("%d %s", &index, buf) == 2) {
                    ListInsert(l, index, buf);
                } else {
                    printf("Input error\n");
                }
                break;
            case 'r':
                if (scanf("%d", &index) == 1) {
                    Node *item_ptr = ListFindNumber(l, index);
                    if (item_ptr != NULL) {
                        ListDeleteNode(l, item_ptr);
                    } else {
                        printf("Incorrect index\n");
                    }
                } else {
                    printf("Input error\n");
                }
                break;
			case 'd':
                if (scanf("%s", buf) == 1) {
                    Node *item_ptr = ListFindValue(l, buf);
                    if (item_ptr != NULL) {
                        ListDeleteNode(l, item_ptr);
                    } else {
                        printf("Value has not been found in the list\n");
                    }
                } else {
                    printf("Input error\n");
                }
                break;
            case 'l':
            	printf("List length:%d\n", ListLength(l));
            	break;
            case 'e':
                ListExchange(l);
                break;               
        }
    }
    ListDestroy(&l);
    return 0;
}
