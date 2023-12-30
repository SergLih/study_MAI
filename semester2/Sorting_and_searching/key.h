#ifndef _KEY_H_
#define _KEY_H_

#include <stdbool.h>
#include <string.h>

typedef float Key;
typedef char * Value;

Key key_create();
void key_destroy(Key *k);
void value_destroy(Value * v);
int key_compare(Key k1, Key k2);
void key_copy(Key * dest, Key * src);
void value_copy(Value * dest, Value * src);
Key key_input(bool prompt);
Value value_input();

#endif
