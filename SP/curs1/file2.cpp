/*  SSL2018   */
#include "mlisp.h"

double __SSL2018__try(double neg__point, double pos__point);
//________________ 
double __SSL2018__try(double neg__point, double pos__point) {
	{//let
		double midpoint(average(neg__point, pos__point)), test__value(0);
		display("+");
		test__value = fun(midpoint);
		return (close__enough_Q(neg__point, pos__point) ? midpoint 
		: (0 <= test__value)======= ? __SSL2018__try(neg__point, midpoint) 
		: (test__value <= 0)======= ? __SSL2018__try(midpoint, pos__point) 
		: (test__value == 0)======= ? midpoint 
		: _infinity);
	}//let
}


