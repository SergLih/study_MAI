#ifndef VECTOR_INTERFACE_H
#define VECTOR_INTERFACE_H

#include <stdbool.h>

#define SUCCESS 0
#define FAILURE 1

typedef int TItem;
typedef struct _vector TVector;

TVector* VectorCreate();
bool VectorAppend(TVector* vector, TItem new_elem);
void VectorPrint(TVector* vector);
void VectorDestroy(TVector** vector);

typedef TVector* vector_create_t(size_t start_capacity);
typedef bool vector_append_t(TVector* vector, TItem new_elem);
typedef void vector_print_t(TVector* vector);
typedef void vector_destroy_t(TVector** vector);

#endif
