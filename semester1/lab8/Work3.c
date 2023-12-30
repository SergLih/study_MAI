#include <stdio.h>

int main(void)
{
    long int size = 0;
    long int result = 0;
    long int tmp = 0;
    scanf("%ld", &size);
    for (int i = 0; i < size; ++i) {
        scanf("%ld", &tmp);
        if (tmp >= 10) {
            result += tmp;
        } else {
            tmp = 0;
        }
    }
    printf("%ld\n", result);
    return 0;
}
