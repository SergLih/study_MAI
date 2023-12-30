//  half-interval.cpp 2018
/*
(define(fun z)
  (set! z (- z (/ 17 18)(/ e)))
  (+(* 2 (exp (- z)))
    (sin (+ z pi))
    (tan z)
    (expt(- z 1)2)
    0.17)
)
(define(fun z)
  (set! z (- z (/ 108 109)(/ e)))  
  (-(-(* z 3) (*(log z) 4)) 5)
)
*/
#include "mlisp.h"

double half_interval_metod(double a, double b);
double __SSL__try(double neg_point, double pos_point);
bool close_enough(double x, double y);
double average(double x, double y);
double root(double a, double b);
double fun(double z);

double tolerance = 0.00001;

double half_interval_metod(double a, double b) {
	{//let
    double a__value(fun(a)), b__value(fun(b));
    return a__value < 0 && b__value > 0 ? __SSL__try(a, b) :
    a__value > 0 && b__value < 0 ? __SSL__try(b, a) :
    b + 1.;
	}//let
}

double __SSL__try(double neg_point, double pos_point) {
	{
    double midpoint(average(neg_point, pos_point)), test_value = 0;
    display("+");
    return close_enough(neg_point, pos_point) ? midpoint 
	    : true ?
	        test_value = fun(midpoint),
            test_value > 0 ? __SSL__try(neg_point, midpoint) :
            test_value < 0 ? __SSL__try(midpoint, pos_point) : 
            midpoint : _infinity;
	}
}

bool close_enough(double x, double y) {
	return abs(x - y) < tolerance;
}

double average(double x, double y) {
	return (x + y) / 2.;
}

double root(double a, double b) {
    newline();
    display("interval=\t[");
    display(a);
    display(" , ");
    display(b);
    display("]\n");
    {//let
    	double temp = half_interval_metod(a, b);
		newline();
	    display("discrepancy=\t");
	    display(fun(temp));
		newline();
	    display("root=\t\t");
	    display(temp - b - 1. == 0 ? "[bad]" : "[good]");
	    return temp;
    }//let
}

double fun(double z) {
	z = z - (108./109.) - (1./e);
    return (3. * z) - (4. * log(z)) - 5;
}

int main() {
	display("SSL variant 8");
	display(root(4, 5));
	newline();
	std::cin.get();
	return 0;
}
