#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <string.h>
#include "matrix.h"

void row_create(Row *row, int n)
{
    row->n = n;
    row->elem = (elem_type*)malloc(n * sizeof(elem_type)); 
}

void row_destroy(Row *row)
{
    free(row->elem);
}

void matrix_create(Matrix* mat, int n, int m) {
    mat->n = n;
    mat->m = m;
    mat->entries = 0;
    mat->rowp = (int*)malloc((n + 1) * sizeof(int));
    mat->column = (int*)malloc(1 * sizeof(int));
    mat->elem = NULL;
    memset(mat->rowp, 0, (n + 1) * sizeof(int));
    mat->column[0] = 0;
}

void matrix_destroy(Matrix* mat) {
    free(mat->rowp);
    free(mat->column);
    free(mat->elem);
}

void matrix_set(Matrix* mat, int i, int j, elem_type val) {
    if (val == 0)
        return;
    for (int k = i + 1; k <= mat->n; ++k)
        mat->rowp[k]++;
    mat->entries++;
    mat->column = (int*)realloc(mat->column, (mat->entries + 1) * sizeof(int));
    mat->elem = (elem_type*)realloc(mat->elem, (mat->entries + 1) * sizeof(elem_type));
    mat->column[mat->entries - 1] = j;
    mat->elem[mat->entries - 1] = val;
}

elem_type get(Matrix* mat, int i, int j)
{
    if (mat->rowp[i] == -1)
        return 0;
    for (int k = i + 1; k <= mat->n; ++k) {
        if (mat->rowp[k] != -1) {
            for (int l = mat->rowp[i]; l < mat->rowp[k]; ++l) {
                if (mat->column[l] == j)
                    return mat->elem[l];
            }
            break;
        }
    }
    return 0;
}

void print_full_matrix(Matrix* mat)
{
    for (int i = 0; i < mat->n; ++i) {
        for (int j = 0; j < mat->m; ++j) {
            elem_type num = get(mat, i, j);
            printf("(%.2lf + %.2lfi) ", creal(num), cimag(num));
        }
        printf("\n");
    }
    printf("\n");
}

void multrow(Row *row, Matrix *mat, Matrix *res) {
    for (int k = 0; k < mat->m; ++k) {
        elem_type s = 0;
        for (int j = 0; j < row->n; ++j)
            s += row->elem[j]*get(mat, j, k);
        matrix_set(res, 0, k, s);
    }
}

void multsparse(Matrix *mat1, Matrix *mat2, Matrix *res) {    
    for (int i = 0; i < mat1->n; ++i)
        for (int k = 0; k < mat2->m; ++k) {
            elem_type s = 0;
            for (int t = mat1->rowp[i]; t < mat1->rowp[i+1]; ++t) {
                int j = mat1->column[t];
                s += get(mat1, i, j) * get(mat2, j, k);
            }
            matrix_set(res, i, k, s);
        }
}

void print_in_computer_view(Matrix* mat)
{
    printf("CIP: ");
    for (int i = 0; i <= mat->n; ++i) 
        printf("%d ", mat->rowp[i]);
    printf("\n");
    printf("PI: ");
    for (int i = 0; i < mat->entries; ++i)
        printf(" %d ", mat->column[i]);
    printf("\n");
    printf("YE: ");
    for (int i = 0; i < mat->entries; ++i)
        printf("(%.2lf + %.2lf) ", creal(mat->elem[i]), cimag(mat->elem[i]));
    printf("\n");
}

int nonzero(Matrix *mat) {
    return mat->entries;
}
