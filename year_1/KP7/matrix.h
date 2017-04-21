#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <complex.h>

typedef double complex elem_type;

typedef struct matrix {
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
    
void create(Matrix *mat, int n, int m);
void destruction(Matrix *mat);
elem_type get(Matrix *mat, int i, int j);
void set(Matrix *mat, int i, int j, elem_type value);
void multrow(Row *row, Matrix *mat, Matrix *result);
void multsparse(Matrix *mat1, Matrix *mat2, Matrix *result);
void print(Matrix *mat);
void printsparse(Matrix *mat);
int nonzero(Matrix *mat);

#endif // __MATRIX_H__
