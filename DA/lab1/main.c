#include <stdio.h>
#include <stdlib.h>
#include "vector.h"

int main(void) {
    TVector *vector = VectorCreate();
    char phone[LEN_TEXT];
    TValue val;

    while (scanf("%s", phone) == 1) {
        scanf("%llu", &val);
        TItem new_item;
        InputKey(phone, &new_item);
        new_item.val = val;
        VectorAppend(vector, new_item);
    }
    RadixSort(vector);
    VectorPrint(vector);
    VectorDestroy(&vector);
    return 0;
}
