#ifndef VECTOR_EXT_H
#define VECTOR_EXT_H

#include <stdio.h>
#include <stdlib.h>
#include "vector_interface.h"

const static int VECTOR_EXTENSION_FACTOR = 2;

typedef struct _vector{
    TItem* arr;
    size_t size;
    size_t capacity;
} TVector;


bool VectorResize(TVector* vector);

#endif //VECTOR_H
