#include <stdio.h>
#include <math.h>
#include <float.h>

#include "root_finding.h"

double f1(double x) 
{
    return 3 * x - 14 + exp(x) - exp(-x); 
}

double df1(double x)
{
    return 3 + exp(x) + exp(-x);
}

double f1_contraction(double x) 
{
    return log(exp(-x) - 3 * x + 14); 
}

double f2(double x) 
{
    return (sqrt(1 - x) - tan(x)); 
}

double df2(double x)
{
    return (-1) * (1 / 2) * (1 / (sqrt(1 - x))) - 1 / (pow(cos(x), 2));
}

double f2_contraction(double x) 
{
     return 1 - pow(tan(x), 2);
}

int main(void)
{
    double root;
    
    root = root_dihotom(f1, 1.0, 3.0);
    printf("Root function f1 obtained by method of dichotomy:\n");
    printf("f1 = %lf\n", root);
    
    root = root_iterations(f1_contraction, 1.0, 3.0);
    printf("Root function f1_contraction obtained by method of iterations:\n");
    printf("f1 = %lf\n", root);
    
    root = root_newton(f1, df1, 1.0, 3.0);
    printf("Root function f1 obtained by method of Newton:\n");
    printf("f1 = %lf\n", root);
    
    root = root_dihotom(f2, 0.0, 1.0);
    printf("Root function f2 obtained by method of dichotomy:\n");
    printf("f2 = %lf\n", root);
    
    root = root_iterations(f2_contraction, 0.0, 1.0);
    printf("Root function f2_contraction obtained by method of iterations:\n");
    printf("f2 = %lf\n", root);
    
    root = root_newton(f2, df2, 0.0, 0.9);
    printf("Root function f2 obtained by method of Newton:\n");
    printf("f2 = %lf\n", root);
    
    return 0;
}