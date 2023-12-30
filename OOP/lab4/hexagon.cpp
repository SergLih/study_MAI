#include "hexagon.h"
#include <iostream>
#include <climits>
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif


Hexagon::Hexagon() {
    side = 0;
#ifdef DEBUG
	std::cout << "Hexagon: created by default ctr" << std::endl;
#endif
}

Hexagon::Hexagon(size_t side) {
	this->side = side;
#ifdef DEBUG
    std::cout << "Hexagon: created with side" << std::endl;
#endif
}

Hexagon::Hexagon(std::istream & is)
{
	long long tmp_side;
	std::cout << "Hexagon: enter side length: ";
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
	std::cout << "Hexagon: created\n";
#endif // DEBUG
}

Hexagon::Hexagon(const Hexagon& orig) {   

#ifdef DEBUG
    std::cout << "Hexagon: created via copy ctr" << std::endl;
#endif
    side = orig.side;
}

double Hexagon::Square() {
	return double((5.0 * side * side) / (4.0 * tan(M_PI / 5.0)));
}

void Hexagon::print(std::ostream & os) const
{
	os << *this;
}

Hexagon& Hexagon::operator=(const Hexagon& right) {
	if (this == &right) {
/*#ifdef DEBUG
		std::cout << "Hexagon: self-copy prevented" << std::endl;
#endif*/ //DEBUG
		return *this;
	}
/*#ifdef DEBUG
	std::cout << "Hexagon copied via =" << std::endl;
#endif*/ // DEBUG
    side = right.side;
    return *this;
}

Hexagon::~Hexagon() {
#ifdef DEBUG
	std::cout << "Hexagon deleted" << std::endl;
#endif // DEBUG
}

std::ostream& operator<<(std::ostream& os, const Hexagon& obj) {
    os << "Hexagon, side = " << obj.side << std::endl;
    return os;
}

std::istream& operator>>(std::istream& is, Hexagon& obj) {
    int tmp_side;
    while(!(is >> tmp_side) || tmp_side <= 0){
        is.clear();
        is.ignore(111111111, '\n');
        std::cerr << "Invalid input. Side must be a positive number. Try again: ";
    }
    obj.side = tmp_side;
    return is;
}
