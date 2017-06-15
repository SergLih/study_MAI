#ifndef ___LIST_H_
#define ___LIST_H_


typedef struct NODE {
    struct NODE *next;
    char * value;
} Node;

typedef struct LIST {
    Node *head;
    Node *barrier;
    int size;
} List;

void ListPrint(List *a_list);

Node *NodeCreate(char * string_value);
void NodeDestroy(Node **node);
List *ListCreate();
void ListDestroy(List **list);
void ListAddNode(List *a_list, char * a_value);
void ListInsert(List *a_list, int index, char * a_value);
void ListExchange(List *a_list);

#endif
