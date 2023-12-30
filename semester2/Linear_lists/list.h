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

void ListPrint(List *list);

Node *NodeCreate(char * string_value);
void NodeDestroy(Node **node);
List *ListCreate();
void ListDestroy(List **list);
void ListAddNode(List *list, char * value);
void ListInsert(List *list, int index, char * value);
void ListExchange(List *list);
int  ListLength(List *list);
bool ListDeleteNode(List *list, Node *node_to_delete);

#endif
