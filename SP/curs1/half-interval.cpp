/*  SSL2018   */
#include "mlisp.h"

double half__interval__metod(double a, double b);
double __SSL2018__try(double neg__point, double pos__point);
bool close__enough_Q(double x, double y);
double average(double x, double y);
double root(double a, double b);
double tolerance = 0.00001;
double fun(double z);
//________________ 
double half__interval__metod(double a, double b) {
	{//let
		double a__value(fun(a)), b__value(fun(b));
		return ((!(!((!((!(a__value <= 0)) || (a__value == 0))) || (!(b__value <= 0))))) ? __SSL2018__try(a, b) 
		: (!(!((!(a__value <= 0)) || (!((!(b__value <= 0)) || (b__value == 0)))))) ? __SSL2018__try(b, a) 
		: ((!((!(a__value <= 0)) || (!(b__value <= 0)))) || (!((!((!(a__value <= 0)) || (a__value == 0))) || (!((!(b__value <= 0)) || (b__value == 0)))))) ? (b + 1) 
		: _infinity);
	}//let
}

double __SSL2018__try(double neg__point, double pos__point) {
	{//let
		double midpoint(average(neg__point, pos__point)), test__value(0);
		display("+");
		test__value = fun(midpoint);
		return (close__enough_Q(neg__point, pos__point) ? midpoint 
		: (!(test__value <= 0)) ? __SSL2018__try(neg__point, midpoint) 
		: (!((!(test__value <= 0)) || (test__value == 0))) ? __SSL2018__try(midpoint, pos__point) 
		: (test__value == 0) ? midpoint 
		: _infinity);
	}//let
}

bool close__enough_Q(double x, double y) {
	return (!((!(abs(x - y) <= tolerance)) || (abs(x - y) == tolerance)));
}
double average(double x, double y) {
	return (1. / 2.0e+0) * (x + y);
}

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

double fun(double z) {
	z = z - (1. / 109) * 108 - (1. / e);
	return z * 3 - log(z) * 4 - 5;
}

int main() {
  display(" SSL variant 8"); newline(); 
  display(root(4, 5)); newline();

  std::cin.get();
  return 0;
}

