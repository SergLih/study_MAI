#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define X1 -10
#define Y1 -10
#define X2 -20
#define Y2 -20
#define R1 10
#define R2 10

#define MAX_ITER 50

int sign(int a)
{
    return (a > 0) - (a < 0);
}

int max(int a, int b)
{
    return (a > b) ? a : b;
}

int mod(int a, int b)
{
    return (((a % b) + b) % b);
}

int dotInRound(double xcenter, double ycenter, double radius, double i, double j)
{
    return pow(i - xcenter, 2) + pow(j - ycenter, 2) <= pow(radius, 2);
}

int check(int i, int j)
{
    return dotInRound(X1, Y1, R1, i, j) && dotInRound(X2, Y2, R2, i, j);
}

int main(void)
{
    int i = 0, j = 0, l = 0, k = 0;
    float inext = 0, jnext = 0, lnext = 0;
    scanf("%d %d %d", &i, &j, &l);

    for (k = 1; k <= MAX_ITER; ++k) {
        inext = i / 3 - abs(i - k) * sign(l - j);
        jnext = mod(j, 10) - mod(max(i, l), (k + 1));
        lnext = i + mod(j * k, 5) + l / 5 + 3;
       
        i = inext;
        j = jnext;
        l = lnext;
        if (check(i, j)) {
            printf("YES\n");
            break;
        }
    }


    if (k > MAX_ITER) {
        printf("NO\n");
        --k;
    }

    printf("%d %d %d %d\n", i, j, l, k);
}
