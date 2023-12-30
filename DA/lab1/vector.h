#ifndef VECTOR_H
#define VECTOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

const static int VECTOR_DEFAULT_CAPACITY = 2;
const static int VECTOR_EXTENSION_FACTOR = 2;

const static int COUNTRY_LEN = 3;
const static int REGION_LEN = 3;
const static int PHONE_LEN  = 7;

typedef enum {
    STATE_COUNTRY, STATE_REGION, STATE_PHONE
} TState;

enum {	LEN_TEXT = 64,
        RADIX = 256
        }; 

const static int DIGITS = 8;

typedef unsigned long long TKey;
typedef unsigned long long TValue;

typedef unsigned char TByte;

typedef struct item {
    TKey key;
    TValue val;
    size_t stable_index;       //дополненный ключ для устойчивой быстрой сортировки
    TByte real_country_len;    //для правильного форматирования телефона при выводе (ведущие нули)
    TByte real_region_len;
    TByte real_phone_len;
} TItem;

typedef struct vector {
    TItem* arr;
    size_t size;
    size_t capacity;
} TVector;

TVector* VectorCreate();
void InputKey(char[LEN_TEXT], TItem * item);
bool VectorAppend(TVector* vector, TItem new_elem);
bool VectorResize(TVector* vector);
void VectorPrint(TVector* vector);
void PrintItem(TItem * item);
TByte Digit(TKey key, TByte digit_idx);
void VectorDestroy(TVector** vector);
void RadixSort(TVector* vector);

#endif //VECTOR_H
