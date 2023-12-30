#include "pentagon.h"
#include <iostream>
#include <cmath>

Pentagon::Pentagon() : Pentagon(0) {
}

Pentagon::Pentagon(size_t side) {
    std::cout << "Pentagon: created\n";
}

Pentagon::Pentagon(std::istream &is) {
    long long tmp_side;
    std::cout << "Pentagon: enter side length: ";
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
    std::cout << "Pentagon: created\n";
}

Pentagon::Pentagon(const Pentagon& orig) {   
    std::cout << "Pentagon: created via copy ctr" << std::endl;
    side = orig.side;
}

double Pentagon::Square() {
    std::cout << "Pentagon: square: ";
    return double((5.0 * side * side) / (4.0 * tan(M_PI / 5.0)));
}

void Pentagon::Print() {
    std::cout << "Pentagon: side length = " << side << std::endl;
}

Pentagon::~Pentagon() {
    std::cout << "Pentagon: deleted " << std::endl;
}
