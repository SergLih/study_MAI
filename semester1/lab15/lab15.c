#include <stdio.h>

#define FALSE 0
#define TRUE 1

int main(void)
{
    int test_count = 0, n = 0, label = 0;
    int the_current_size = 0;
    scanf("%d %d", &test_count, &n);
    int matrix[n][n];
    int columns_for_delete[n];
    for (int test_meter = 0; test_meter < test_count; ++test_meter) {
        scanf("%d", &the_current_size);

        for (int i = 0; i < the_current_size; i++) {
            for (int j = 0; j < the_current_size; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }

        for (int i = 0; i < the_current_size; ++i) {
            columns_for_delete[i] = FALSE;
            for (int checked_index = 0; checked_index < the_current_size - 1; ++checked_index) {
                for (int j = checked_index + 1; j < the_current_size; ++j) {
                    if (columns_for_delete[j]) {
                        continue;
                    }
                    for (int row_iter = 0; row_iter < the_current_size; ++row_iter) {
                        if (matrix[row_iter][checked_index] == matrix[row_iter][j]) {
                            columns_for_delete[j] = TRUE;
                        } else {
                            columns_for_delete[j] = FALSE;
                            break;
                        }
                    }
                }
            }
        }

        //Сдвигаем столбцы матрицы, column -- итоговое количество столбцов
        int column = 0;
        for (int i = 0; i < the_current_size; i++) {
            column = 0;
            for (int j = 0; j < the_current_size; j++) {
                if (!columns_for_delete[j]) {
                    matrix[i][column++] = matrix[i][j];
                }
            }
        }
        
        //Печатаем итоговую матрицу
        for (int i = 0; i < the_current_size; i++) {
            for (int j = 0; j < column; j++) {
                printf("%d ", matrix[i][j]);
            }
            printf("\n");
        }
    }
    return 0;
}
