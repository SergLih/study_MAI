#include "vector.h"

TByte Digit(TKey key, TByte n_byte) {
    unsigned long long mask = 0xFF;
    return (key >> n_byte * 8) & mask;
}

void RadixSort(TVector* vector) {
    if (vector == NULL || vector->size < 2) {
        return;
    }

    TItem * aux = (TItem *) malloc(sizeof(TItem) * vector->size);
    for (int d = 0; d < DIGITS; d++) {
        size_t counts[RADIX];
        for (int i = 0; i < RADIX; i++) {
            counts[i] = 0;
        }
        for (int i = 0; i < vector->size; i++) {
            counts[ Digit(vector->arr[i].key, d) ]++; 
        }
        for (int i = 1; i < RADIX; i++) {
		    counts[i] += counts[i - 1];
        }
        for (int i = vector->size - 1; i >= 0; i--) {
            aux[--counts[Digit(vector->arr[i].key, d)]] = vector->arr[i];
        }
        for (int i = 0; i < vector->size; i++) {
            vector->arr[i] = aux[i];
        } 
    }

    free(aux);
    aux = NULL;
}

//функция которая создает один вектор динамической памяти и настраивает в него поля
TVector* VectorCreate(void) {
    TVector* vector = (TVector*) malloc(sizeof(TVector));
    vector->arr = (TItem*) malloc(sizeof(TItem) * VECTOR_DEFAULT_CAPACITY);
    vector->size = 0;
    vector->capacity = VECTOR_DEFAULT_CAPACITY;
    return vector;
}

void InputKey(char s[LEN_TEXT], TItem * item) {
    TState state = STATE_COUNTRY;
    TKey country_code = 0, region_code = 0, person_phone = 0, n = 1;
    item->real_country_len = 0;
    item->real_region_len = 0;
    item->real_phone_len = 0;
    for (int i = 0; s[i] != '\0'; i++) {
        if (s[i] == '-') {
            if (state == STATE_COUNTRY) {
                state = STATE_REGION;
            } else if (state == STATE_REGION) {
                state = STATE_PHONE;
            }
        } else if (s[i] >= '0' && s[i] <= '9') {
            if (state == STATE_COUNTRY) {
                country_code *= 10;
                country_code += (s[i] - '0');
                item->real_country_len++;
            } else if (state == STATE_REGION) {
                region_code *= 10;
                region_code += (s[i] - '0');
                item->real_region_len++;
            } else {
                person_phone *= 10;
                person_phone += (s[i] - '0');
                item->real_phone_len++;
            }
        }
    }
    n = 1;
    for (int i = 0; i < PHONE_LEN; i++)  {
        n *= 10;
    }
    region_code *= n;
    for (int i = 0; i < REGION_LEN; i++) {
        n *= 10;
    }
    country_code *= n;
    item->key = country_code + region_code + person_phone;
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
    new_elem.stable_index = vector->size;
    vector->arr[vector->size] = new_elem;
    vector->size++;
    return true;
}

void VectorPrint(TVector* vector) {
    if (vector) {
        for (int i = 0; i < vector->size; i++) {
            PrintItem(&(vector->arr[i]));
        }
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

void PrintItem(TItem *item) {
    TKey country_code, region_code, person_phone, n;
    TKey key = item->key;
    n = 1;
    for (int i = 0; i < PHONE_LEN; i++) {
        n *= 10;
    }
    person_phone = key % n;
    key /= n;
    n = 1;
    for (int i = 0; i < REGION_LEN; i++) {
        n *= 10;
    }
    region_code = key % n;
    key /= n;
    country_code = key;
    char format[30] = {0};
    sprintf(format, "+%%0%dllu-%%0%dllu-%%0%dllu\t%%llu\n", (int)item->real_country_len, (int)item->real_region_len, (int)item->real_phone_len);
    printf(format, country_code, region_code, person_phone, item->val);
}
