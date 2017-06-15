#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "iterator.h"

bool IteratorEqual(const Iterator *lhs, const Iterator *rhs)
{
    return lhs->node == rhs->node;
}

void IteratorStart(Iterator *i, List l)
{
	i->node = l.head;
}

bool IteratorNext(Iterator *i)
{
	if (i == NULL || i->node == NULL || *(i->node->value)=='\0') {
		return false;
    } else {
    	i->node = i->node->next;
    	return true;
    }
}

char * IteratorFetch(const Iterator *i)
{
    return i->node->value;
}

void IteratorStore(const Iterator *i, const char* value)
{
    strcpy(i->node->value, value);
}
