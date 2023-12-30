#include "vector_ext.h"

//функция которая создает один вектор динамической памяти и настраивает в него поля
TVector* VectorCreate(size_t start_capacity) {
    TVector* vector = (TVector*) malloc(sizeof(TVector));
    vector->arr = (TItem*) malloc(sizeof(TItem) * start_capacity);
    vector->size = 0;
    vector->capacity = start_capacity;
    return vector;
}

bool VectorResize(TVector* vector) {
    if (vector == NULL) {
        return false;
    }

    vector->capacity *= VECTOR_EXTENSION_FACTOR;
    TItem * temp_arr = (TItem*) realloc(vector->arr, sizeof(TItem) * vector->capacity);
    if (temp_arr != NULL) {
        vector->arr = temp_arr;
        return true;
    } else {
        fprintf(stderr, "ERROR: insufficient memory\n");
        return false;
    }
}

bool VectorAppend(TVector* vector, TItem new_elem) {
    if (vector->size >= vector->capacity) {
        if (VectorResize(vector) == false) {
            return false; 
        }
    }
    vector->arr[vector->size] = new_elem;
    vector->size++;
    return true;
}

void VectorPrint(TVector* vector) {
    if (vector) {
        if(vector->size == 0)
            printf("Empty vector");
        for (int i = 0; i < vector->size; i++) {
            printf("%d ", vector->arr[i]);
        }
        printf("\n");
    }
}

void VectorDestroy(TVector** vector) {
    if ((*vector) == NULL) {
        return;
    }

    if (*vector) {
        free((*vector)->arr);
        free(*vector);
        (*vector) = NULL;
    }
}
