#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include "matrix.h"

int main(void)
{
    int n, m;
    double x = 0.0, y = 0.0;
    
    printf("Введите размеры матрицы: ");
    scanf("%d %d", &n, &m);
    
    Matrix mat, matrow;
    Row row;
    row_create(&row, n);
    matrix_create(&mat, n, m);
    matrix_create(&matrow, 1, n);
    
    printf("Введите матрицу размером %d на %d:\n", n, m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            scanf("%lf %lf", &x, &y);
            elem_type val = x + y * I;
            matrix_set(&mat, i, j, val);
        }
    }
    printf("Введите вектор-строку из %d элементов:\n", n);
    for (int i = 0; i < n; ++i) {
        scanf("%lf %lf", &x, &y);
        elem_type val = x + y * I;
        matrix_set(&matrow, 0, i, val);
        row.elem[i] = val;
    }
    Matrix mult, mult2;
    matrix_create(&mult,  1, m);
    matrix_create(&mult2, 1, m);
    
    printf("\nМатрица %d на %d в обычном представлении:\n", n, m);
    print_full_matrix(&mat);
    printf("\nМатрица %d на %d в заданной схеме хранения:\n", n, m);
    print_in_computer_view(&mat);
    
    printf("\nВектор-строка %d на %d в обычном представлении:\n", 1, n);
    print_full_matrix(&matrow);
    printf("\nВектор-строка %d на %d в заданной схеме хранения:\n", 1, n);
    print_in_computer_view(&matrow);
    
     printf("\n\n=======================\nРезультат умножения: две разреженные матрицы\n");
    multsparse(&matrow, &mat, &mult);

    printf("\nМатрица %d на %d в обычном представлении:\n", 1, m);
    print_full_matrix(&mult);
    printf("\nМатрица %d на %d в заданной схеме хранения:\n", 1, m);
    print_in_computer_view(&mult);
    printf("Количество ненулевых элементов: %d\n", nonzero(&mult));
    
    printf("\n\n=======================\nРезультат умножения: вектор-строка и разреженная матрица\n");
    multrow(&row, &mat, &mult2);
    
    printf("\n\nМатрица %d на %d в обычном представлении:\n", 1, m);
    print_full_matrix(&mult2);
    printf("\n\nМатрица %d на %d в заданной схеме хранения:\n", 1, m);
    print_in_computer_view(&mult2);
    printf("Количество ненулевых элементов: %d\n", nonzero(&mult2));
    
    row_destroy(&row);
    matrix_destroy(&mat);
    matrix_destroy(&matrow);
    matrix_destroy(&mult);
    matrix_destroy(&mult2);
    
    return 0;
}
