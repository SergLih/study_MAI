#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <time.h>

#include "vector.h"

bool CheckOrder(TVector *vec)        //функция проверяющая упорядоченность ключей отсортированного вектора
{
    for (size_t i = 1; i < vec->size; ++i) {
        if(vec->arr[i - 1].key > vec->arr[i].key) {
            return false;
        }
    }
    return true;
}

int Compare(const void *arg1, const void *arg2) //функция-компоратор для 
{
    if (((TItem*)arg1)->key < ((TItem*)arg2)->key) {
        return -1;
    } else if (((TItem*)arg1)->key > ((TItem*)arg2)->key) {
        return 1;
    } else {
        /*if (((TItem*)arg1)->stable_index < ((TItem*)arg2)->stable_index) {
            return -1;
        } else if (((TItem*)arg1)->stable_index > ((TItem*)arg2)->stable_index) {
            return 1;
        } else { 
            return 0;
        }*/
        return 0;
    }
}

void RunSort(TVector *vec, char *s)
{
    time_t start = clock();
    if (!strcmp(s, "radix")) {
        RadixSort(vec);
    } else if (!strcmp(s, "qsort")) {
        qsort(vec->arr, vec->size, sizeof(*(vec->arr)), Compare);
    } else {
        fprintf(stderr, "Unknown type of sorting\n");
        return;
    }
    time_t end = clock();
    assert(CheckOrder(vec) == true);
    fprintf(stderr, "Type of the sort: %s\n", s);
    fprintf(stderr, "Working time:     %f sec.\n", (double)(end - start) / (double)CLOCKS_PER_SEC);
}


TVector* VectorCopy(TVector *vec)
{
    TVector * newVec = VectorCreate();
    newVec->arr = (TItem*) realloc(newVec->arr, sizeof(TItem) * (vec->size+1)); //создаем vectorcopy сразу нужного размера. 
    newVec->capacity = (vec->size+1);                                               //+1 на случай пустого вектора
    for (size_t i = 0; i < vec->size; ++i) {
        VectorAppend(newVec, vec->arr[i]);
    }
    return newVec;
}


int main(int argc, const char *argv[])
{
    TVector* vec = VectorCreate();
    char phone[LEN_TEXT];
    TValue val;
    while(scanf("%s", phone) == 1) {
        scanf("%llu", &val);
        TItem new_item;
        InputKey(phone, &new_item);
        new_item.val = val;
        VectorAppend(vec, new_item);
    }
    TVector * vecCopy = VectorCopy(vec);
    fprintf(stderr, "\nVector size: %zu\n", vec->size);
    //RunSort(vec, "radix");
    RunSort(vecCopy, "qsort");
    //VectorPrint(vec);
    printf("==============================\n");
    VectorPrint(vecCopy);
    
    VectorDestroy(&vec);
    VectorDestroy(&vecCopy);
    return 0;
}
