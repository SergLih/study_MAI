#ifndef VECTOR_ST_H
#define VECTOR_ST_H

#include <stdio.h>
#include <stdlib.h>
#include "vector_interface.h"

typedef struct _vector {
    TItem* arr;
    size_t size;
    size_t capacity;
} TVector;


#endif //VECTOR_H
