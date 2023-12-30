/*  SSL2018   */
#include "mlisp.h"

double cc(double amount, double largest__coin);
double count__change(double amount);
double next__coin(double coin);
double GR__AMOUNT();
double dd = 13;
double mm = 12;
double LARGEST__COIN = 15;
//________________ 
double cc(double amount, double largest__coin) {
	return (((amount == 0) || (largest__coin == 1)) ? 1 
		: (!(!((!(0 <= amount)) || (!(0 <= largest__coin))))) ? 0 
		: (!(!((!((!(0 <= amount)) || (0 == amount))) || (!((!(0 <= largest__coin)) || (0 == largest__coin)))))) ? (cc(amount, next__coin(largest__coin)) + cc(amount - largest__coin, largest__coin)) 
		: _infinity);
}

double count__change(double amount) {
	return cc(amount, LARGEST__COIN);
}

double next__coin(double coin) {
	return ((coin == 15) ? 10 
		: (coin == 10) ? 5 
		: (coin == 5) ? 3 
		: (coin == 3) ? 2 
		: (coin == 2) ? 1 
		: ((!(coin == 15)) || (!(coin == 10)) || (!(coin == 5)) || (!(coin == 3)) || (!(coin == 2))) ? 0 
		: _infinity);
}

double GR__AMOUNT() {
	return remainder((100 * mm + dd), 137);
}

int main() {
  display(" SSL variant 8"); 
  newline(); 
  display(" 1-2-3-5-10-15"); 
  newline(); 
  display("count-change for 100 \t= "); 
  display(count__change(100)); 
  newline(); 
  display("count-change for "); 
  display(GR__AMOUNT()); 
  display(" \t= "); 
  display(count__change(GR__AMOUNT())); 
  newline(); 

  std::cin.get();
  return 0;
}

