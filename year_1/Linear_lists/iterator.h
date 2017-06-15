#ifndef ___IO_H_
#define ___IO_H_
#include <stdbool.h>
#include "list.h"


typedef struct {
    Node *node;
} Iterator;

bool IteratorEqual(const Iterator *lhs, const Iterator *rhs);
void IteratorStart(Iterator *i, List l);
bool IteratorNext(Iterator *i);
char * IteratorFetch(const Iterator *i);
void IteratorStore(const Iterator *i, const char * value);

Node *ListFindValue(List *list, char * value_to_find);
Node *ListFindNumber(List *list, int number);

#endif
