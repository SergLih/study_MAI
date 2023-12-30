#include <stdio.h>
#include <math.h>

double my_function(double x)
{
    return cosh(x);
}

double my_taylor(double x, int *count_terms, int k)
{
    double epsilon = 7.0 / 3.0 - 4.0 / 3.0 - 1.0;
    double eps = epsilon * k;
    double present_taylor = 1.0;//последнее слагаемое
    //double previous_taylor = 1.0;//предпоследнее слагаемое
    int number_iter = 1;
    double sum = 1.0;//сумма слагаемых
    do {
        //previous_taylor = present_taylor;
        //previous_sum = sum;
        present_taylor *= x;
        present_taylor /= number_iter;
        present_taylor *= x;
        present_taylor /= number_iter + 1.0;
        sum += present_taylor;
        number_iter += 2;
    } while (fabs(present_taylor /*- previous_taylor*/   /*previous_sum - sum*/) > eps && number_iter <= 99);
//пока последнее слагаемое больше eps
    *count_terms = number_iter / 2;
    return sum;
}

void print_functions(int n, int k, double a, double b)
{
    int count_terms;
    double h = (b - a) / n; 
    for (double x = a; x <= b; x += h) {
        printf("| %lf   | %.15lf    | %.15lf  | ", x, my_taylor(x, &count_terms, k), my_function(x));
        printf("       %d        |\n", count_terms);
    }
}

int main(void)
{
    int n; //количество точек
    int k;
    
    scanf("%d%d", &n, &k);
    printf(" __________________________________________________________________________ \n");
    printf("|    Таблица значений ряда Тейлора и стандартной функции для f(x)          |\n");
    printf("|__________________________________________________________________________|\n");
    printf("| Значение x | Зн-е по ф-ле Тейлора | Значение функции   | Кол-во итераций |\n");
    printf("|__________________________________________________________________________|\n");
    print_functions(n, k, 0.1, 0.6);
    printf("|____________|______________________|____________________|_________________|\n");
    return 0;
}
