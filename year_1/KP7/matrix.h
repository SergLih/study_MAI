#ifndef _MATRIX_H_
#define _MATRIX_H_
#include <complex.h>

typedef complex double elem_type;

typedef struct {
    int n;
    int m;
    int entries;
    int *rowp;
    int *column;
    elem_type *elem;
} Matrix;

typedef struct {
    int n;
    elem_type *elem;
} Row;

void row_create(Row *row, int n);
void row_destroy(Row *row);
void matrix_create(Matrix *mat, int n, int m);
void matrix_destroy(Matrix *mat);
void matrix_set(Matrix *mat, int i, int j, elem_type val);
elem_type matrix_get(Matrix *mat, int i, int j);
void multrow(Row *row, Matrix *mat, Matrix *result);
void multsparse(Matrix *mat1, Matrix *mat2, Matrix *result);
void print_full_matrix(Matrix *mat);
void print_in_computer_view(Matrix *mat);
int nonzero(Matrix *mat);
#endif 
