#include <stdio.h>

int main(void)
{
    long int size = 0;
    long int result = 0;
    long int tmp = 0;
    scanf("%ld", &size);
    for (int i = 0; i < size; ++i) {
        scanf("%ld", &tmp);
        result += tmp;
    }
    printf("%ld\n", result);
    return 0;
}
