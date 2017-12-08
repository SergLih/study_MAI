#include "pentagon.h"
#include <iostream>
#include <climits>
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif


Pentagon::Pentagon() {
    side = 0;
#ifdef DEBUG
	std::cout << "Pentagon: created by default ctr" << std::endl;
#endif
}

Pentagon::Pentagon(size_t side) {
	this->side = side;
#ifdef DEBUG
    std::cout << "Pentagon: created with side" << std::endl;
#endif
}

Pentagon::Pentagon(std::istream & is)
{
	long long tmp_side;
	std::cout << "Pentagon: enter side length: ";
	while (true) {
		is >> tmp_side;
		is.clear();
		is.sync();
		if (tmp_side <= 0) {
			std::cerr << "Error: The side of any figure must be > 0. Try again: ";
		}
		else {
			side = tmp_side;
			break;
		}
	}
#ifdef DEBUG
	std::cout << "Pentagon: created\n";
#endif // DEBUG
}

Pentagon::Pentagon(const Pentagon& orig) {   

#ifdef DEBUG
    std::cout << "Pentagon: created via copy ctr" << std::endl;
#endif
    side = orig.side;
}

double Pentagon::Square() const {
	return double((5.0 * side * side) / (4.0 * tan(M_PI / 5.0)));
}

void Pentagon::print(std::ostream & os) const
{
	os << *this;
}

Pentagon& Pentagon::operator=(const Pentagon& right) {
	if (this == &right) {
/*#ifdef DEBUG
		std::cout << "Pentagon: self-copy prevented" << std::endl;
#endif*/ //DEBUG
		return *this;
	}
/*#ifdef DEBUG
	std::cout << "Pentagon copied via =" << std::endl;
#endif*/ // DEBUG
    side = right.side;
    return *this;
}

Pentagon::~Pentagon() {
#ifdef DEBUG
	std::cout << "Pentagon deleted" << std::endl;
#endif // DEBUG
}


std::ostream& operator<<(std::ostream& os, const Pentagon& obj) {
	os.precision(2);
	os.setf(std::ios::fixed);
    os << "Pentagon \tside = " << obj.side << "\tsquare = " << obj.Square() << std::endl;
    return os;
}

std::istream& operator>>(std::istream& is, Pentagon& obj) {
    int tmp_side;
    while(!(is >> tmp_side) || tmp_side <= 0){
        is.clear();
        is.ignore(111111111, '\n');
        std::cerr << "Invalid input. Side must be a positive number. Try again: ";
    }
    obj.side = tmp_side;
    return is;
}
