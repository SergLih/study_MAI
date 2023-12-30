#ifndef _TABLE_H_
#define _TABLE_H_

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "key.h"

#define VALLEN 1000

//typedef struct _value * Value;
typedef struct _table * Table;

Table table_create (void);
void  table_destroy(Table *table);
void  table_push(Table table, Key k, Value v);
void  row_print(Table table, size_t pos);
void  table_print(Table table);
void  table_input_and_push(Table table);
void  table_swap_rows(Table table, size_t i, size_t j);
void  table_recursive_sort(Table table, size_t first, size_t last);
void  table_sort(Table table);
Value table_search(Table table, Key key, size_t *pos);
void  table_delete (Table table, Key key);
void  table_reverse(Table table);
void  table_shuffle(Table table);
void  generate_row(size_t k, size_t len, Table table);
void  clean_stdin();

#endif
