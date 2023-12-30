#include "octagon.h"
#include <iostream>
#include <climits>
#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif


Octagon::Octagon() {
    side = 0;
#ifdef DEBUG
	std::cout << "Octagon: created by default ctr" << std::endl;
#endif
}

Octagon::Octagon(size_t side) {
	this->side = side;
#ifdef DEBUG
    std::cout << "Octagon: created with side" << std::endl;
#endif
}

Octagon::Octagon(std::istream & is)
{
	long long tmp_side;
	std::cout << "Octagon: enter side length: ";
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
	std::cout << "Octagon: created\n";
#endif // DEBUG
}

Octagon::Octagon(const Octagon& orig) {   

#ifdef DEBUG
    std::cout << "Octagon: created via copy ctr" << std::endl;
#endif
    side = orig.side;
}

double Octagon::Square() {
	return double((8.0 * side * side) / (4.0 * tan(M_PI / 8.0)));
}

void Octagon::print(std::ostream & os) const
{
	os << *this;
}

Octagon& Octagon::operator=(const Octagon& right) {
	if (this == &right) {
/*#ifdef DEBUG
		std::cout << "Octagon: self-copy prevented" << std::endl;
#endif*/ //DEBUG
		return *this;
	}
/*#ifdef DEBUG
	std::cout << "Octagon copied via =" << std::endl;
#endif*/ // DEBUG
    side = right.side;
    return *this;
}

Octagon::~Octagon() {
#ifdef DEBUG
	std::cout << "Octagon deleted" << std::endl;
#endif // DEBUG
}

std::ostream& operator<<(std::ostream& os, const Octagon& obj) {
    os << "Octagon, side = " << obj.side << std::endl;
    return os;
}

std::istream& operator>>(std::istream& is, Octagon& obj) {
    int tmp_side;
    while(!(is >> tmp_side) || tmp_side <= 0){
        is.clear();
        is.ignore(111111111, '\n');
        std::cerr << "Invalid input. Side must be a positive number. Try again: ";
    }
    obj.side = tmp_side;
    return is;
}
