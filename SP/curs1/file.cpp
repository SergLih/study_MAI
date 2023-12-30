/*  SSL2018   */
#include "mlisp.h"

double half__interval__metod(double a, double b);
//________________ 
double half__interval__metod(double a, double b) {
	{//let
		double a__value(fun(a)), b__value(fun(b));
		return ((!(!((a__value <= 0) || (0 <= b__value)))) ? __SSL2018__try(a, b) 
	:  _infinity)(!(!((0 <= a__value) || (b__value <= 0)))) ? __SSL2018__try(b, a) 
	:  _infinity)((!((0 <= a__value) || (0 <= b__value))) || (!((a__value <= 0) || (b__value <= 0)))) ? (b + 1) 
	: ;
	}//let
}


