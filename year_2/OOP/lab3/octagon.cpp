#include "octagon.h"
#include <iostream>
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

Octagon::Octagon(const Octagon& orig) {   

#ifdef DEBUG
    std::cout << "Octagon: created via copy ctr" << std::endl;
#endif
    side = orig.side;
}

Octagon::Octagon(std::istream &is) {
	long long tmp_side;
	std::cout << "Octagon: enter side length: ";
	while(true){
	    is >> tmp_side;
        is.clear();
        is.sync();
	    if(tmp_side <= 0) {
	        std::cerr << "Error: The side of any figure must be > 0. Try again: ";
	    } else {
	        side = tmp_side;
	        break;
	    }
	}
#ifdef DEBUG
	std::cout << "Octagon: created\n";
#endif // DEBUG
}

double Octagon::Square() {
	return double((5.0 * side * side) / (4.0 * tan(M_PI / 5.0)));
}

Octagon& Octagon::operator=(const Octagon& right) {
	if (this == &right) {
#ifdef DEBUG
		std::cout << "Octagon: self-copy prevented" << std::endl;
#endif //DEBUG
		return *this;
	}
#ifdef DEBUG
	std::cout << "Octagon copied via =" << std::endl;
#endif // DEBUG
    side = right.side;
    return *this;
}

bool operator==(const Octagon& left,const Octagon& right) {
    return left.side == right.side;
}

bool operator>(const Octagon& left,const Octagon& right) {
    return left.side > right.side;
}

bool operator<(const Octagon& left,const Octagon& right) {
    return !(left > right || left == right);
}

bool operator!=(const Octagon& left,const Octagon& right) {
    return !(left == right);
}

bool operator<=(const Octagon& left,const Octagon& right) {
    return (left < right || left == right);
}

bool operator>=(const Octagon& left,const Octagon& right) {
    return (left > right || left == right);
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

void Octagon::print(std::ostream& os) const {
    os << *this;
}

std::istream& operator>>(std::istream& is, Octagon& obj) {
    is >> obj.side;
    return is;
}
