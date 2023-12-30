#include "pentagon.h"
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#define DEBUG 1
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

Pentagon::Pentagon(const Pentagon& orig) {   

#ifdef DEBUG
    std::cout << "Pentagon: created via copy ctr" << std::endl;
#endif
    side = orig.side;
}

double Pentagon::Square() {
	return double((5.0 * side * side) / (4.0 * tan(M_PI / 5.0)));
}

Pentagon& Pentagon::operator=(const Pentagon& right) {
	if (this == &right) {
#ifdef DEBUG
		std::cout << "Pentagon: self-copy prevented" << std::endl;
#endif //DEBUG
		return *this;
	}
#ifdef DEBUG
	std::cout << "Pentagon copied via =" << std::endl;
#endif // DEBUG
    side = right.side;
    return *this;
}

bool operator==(const Pentagon& left,const Pentagon& right) {
    return left.side == right.side;
}

bool operator>(const Pentagon& left,const Pentagon& right) {
    return left.side > right.side;
}

bool operator<(const Pentagon& left,const Pentagon& right) {
    return !(left > right || left == right);
}

bool operator!=(const Pentagon& left,const Pentagon& right) {
    return !(left == right);
}

bool operator<=(const Pentagon& left,const Pentagon& right) {
    return (left < right || left == right);
}

bool operator>=(const Pentagon& left,const Pentagon& right) {
    return (left > right || left == right);
}


Pentagon::~Pentagon() {
#ifdef DEBUG
	std::cout << "Pentagon deleted" << std::endl;
#endif // DEBUG
}

std::ostream& operator<<(std::ostream& os, const Pentagon& obj) {
    os << "Pentagon, side = " << obj.side << std::endl;
    return os;
}

std::istream& operator>>(std::istream& is, Pentagon& obj) {
    int tmp_side;
    while(!(is >> tmp_side) || tmp_side <= 0){
        is.clear();
        is.ignore(11111111111, '\n');
        std::cerr << "Invalid input. Side must be a positive number. Try again: ";
    }
    obj.side = tmp_side;
    return is;
}
