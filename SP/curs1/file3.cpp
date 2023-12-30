/*  SSL2018   */
#include "mlisp.h"

double root(double a, double b);
//________________ 
double root(double a, double b) {
	display("interval=\t[");
	display(a);
	display(" , ");
	display(b);
	display("]\n");
	{//let
		double temp(half__interval__metod(a, b));
		newline();
		display("discrepancy=\t");
		display(fun(temp));
		newline();
		display("root=\t\t");
		display((temp - b - 1 == 0) ? "[bad]" : "[good]");
		return temp;
	}//let
}


