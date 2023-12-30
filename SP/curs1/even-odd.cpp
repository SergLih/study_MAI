/*  SSL2018   */
#include "mlisp.h"

double even__bits(double n);
double odd__bits(double n);
double display__bin(double n);
double report__results(double n);
double dd = 13;
double mm = 12;
double yyyy = 1997;
//________________ 
double even__bits(double n) {
	return ((n == 0) ? 1 
		: (remainder(n, 2) == 0) ? even__bits(quotient(n, 2)) 
		: (!(remainder(n, 2) == 0)) ? odd__bits(quotient(n, 2)) 
		: _infinity);
}

double odd__bits(double n) {
	return ((n == 0) ? 0 
		: (remainder(n, 2) == 0) ? odd__bits(quotient(n, 2)) 
		: true ? even__bits(quotient(n, 2)) 
		: _infinity);
}

double display__bin(double n) {
	display(remainder(n, 2));
	return ((n == 0) ? 0 
		: (!(n == 0)) ? display__bin(quotient(n, 2)) 
		: _infinity);
}

double report__results(double n) {
	display("Happy birthday to you!\n");
	display(n);
	display(" (decimal)\n");
	display("\teven?\t");
	display((even__bits(n) == 1) ? "yes" : "no");
	newline();
	display("\todd?\t");
	display((odd__bits(n) == 1) ? "yes" : "no");
	newline();
	n = display__bin(n);
	display("(reversed binary)\n");
	return 0;
}

int main() {
  display(report__results((dd * 1000000 + mm * 10000 + yyyy))); newline();

  std::cin.get();
  return 0;
}

