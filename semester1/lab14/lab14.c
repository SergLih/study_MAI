#include <stdio.h>

int main(void)
{
    int test_count = 0, n = 0, i, j;
    scanf("%d %d", &test_count, &n);
    long long matrix[n][n];
    for (int test_meter = 0; test_meter < test_count; ++test_meter) {
        int the_current_size = 0;
        scanf("%d", &the_current_size);
 
        for (int i = 0; i < the_current_size; i++) {
            for (int j = 0; j < the_current_size; j++) {
                scanf("%lld", &matrix[i][j]);
            }
        }
 
        for (int diag = 0; diag < 2 * the_current_size - 1; ++diag) {
            if (diag <= the_current_size - 1) {
                j = 0;
                i = diag;
            } else {
                j = diag - the_current_size + 1;
                i = the_current_size - 1;
            }
 
            if (!(diag % 2)) {
                for (; j < the_current_size && i >= 0; i--, j++) {
                    printf("%lld ", matrix[the_current_size - i - 1][j]);
                }
            } else {
                int buffer = i;
                i = j;
                j = buffer;
                for (; i < the_current_size && j >= 0; i++, j--) {
                    printf("%lld ", matrix[the_current_size - i - 1][j]);
                }
            }
        }
        printf("\n");
    }
    return 0;
}
