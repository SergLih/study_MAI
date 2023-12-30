#include <stdio.h>
#include <math.h>
#include <float.h>

#include "root_finding.h"

double root_dihotom(double(*f)(double x), double a, double b)
{
    double middle = 2.0; // the middle of the segment
    while (fabs(b - a) > DBL_EPSILON) {
        if ((f)(middle) == 0) {
            return middle;
        }
        if ((f)(b) * (f)(middle) < 0) {
            a = middle;
        } else {
            b = middle;
        }
        middle = (a + b) / 2.0;
    }
    return middle;
}

double root_iterations(double(*f_contraction)(double x), double a, double b)
{
    double prev, curr = (a + b) / 2.0;
    do {
        prev = curr;
        curr = (f_contraction)(prev);
    } while (fabs(curr - prev) > DBL_EPSILON);
    return curr;
}

double root_newton(double(*f)(double x), double(*df)(double x), double a, double b)
{
    double x = (a + b) / 2.0;
    for (double t = 1.0; fabs(t) > DBL_EPSILON; x -= t) {
        t = f(x) / df(x);
    }
    return x;
}