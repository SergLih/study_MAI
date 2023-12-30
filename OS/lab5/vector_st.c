#include "vector_st.h"

//функция которая создает один вектор динамической памяти и настраивает в него поля
TVector* VectorCreate(size_t start_capacity) {
    TVector* vector = (TVector*) malloc(sizeof(TVector));
    vector->arr = (TItem*) malloc(sizeof(TItem) * start_capacity);
    vector->size = 0;
    vector->capacity = start_capacity;
    return vector;
}

bool VectorAppend(TVector* vector, TItem new_elem) {
    if (vector->size >= vector->capacity) {
        fprintf(stderr, "ERROR: insufficient memory\n");
        return false;
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
