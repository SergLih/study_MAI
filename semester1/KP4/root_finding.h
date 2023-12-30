#ifndef _ROOT_FINDING_H_
#define _ROOT_FINDING_H_

// uncomment to change DBL_PRECISION value from float.h to get results faster

// #define DBL_EPSILON 1e-6

double root_dihotom(double(*f)(double x), double a, double b);
double root_iterations(double(*f_contraction)(double x), double a, double b);
double root_newton(double(*f)(double x), double(*df)(double x), double a, double b);

#endif // _ROOT_FINDING_H_