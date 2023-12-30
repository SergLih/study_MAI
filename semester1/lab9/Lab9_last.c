#include <stdlib.h>

#include <stdio.h>
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
    return abs((a % b + b) % b);
}

int _div(double a, double b)
{
    return floor(a / b);
}


int dot_in_circle(int i, int j, int x, int y, int r)
{
    return pow(x - i, 2) + pow(y - j, 2) <= pow(r, 2);
}

int check(int i, int j)
{
   
    return dot_in_circle(i, j, X1, Y1, R1) && dot_in_circle(i, j, X2, Y2, R2);
}

int compute_i(int i, int j, int l, int k)
{
    return _div(i, 3) - abs(i - k) * sign(l - j);
}
 
int compute_j(int i, int j, int l, int k)
{
    return mod(j, 10) - mod(max(i, l), k + 1);
}
 
int compute_l(int i, int j, int l, int k)
{
    return i + mod(j * k, 5) + _div(l, 5) + 3;
}

int main(void)
{
    int k;
    int i = 0, j = 0, l = 0;
    int i_old = 0, j_old = 0, l_old = 0;
    scanf("%d %d %d", &i_old, &j_old, &l_old);

    for (k = 1; !check(i_old, j_old) && k <= 50; ++k) {
        i = compute_i(i_old, j_old, l_old, k - 1);
        j = compute_j(i_old, j_old, l_old, k - 1);
        l = compute_l(i_old, j_old, l_old, k - 1);
        i_old = i;
        j_old = j;
        l_old = l;
    }
    printf("%s\n", check(i_old , j_old) ? "Yes" : "No");
    printf("%d %d %d %d\n", i_old, j_old, l_old, k - 1);
    return 0;
}
