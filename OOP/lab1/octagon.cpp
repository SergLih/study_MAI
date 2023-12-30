#include "octagon.h"
#include <iostream>
#include <cmath>

Octagon::Octagon() : Octagon(0) {
}

Octagon::Octagon(size_t side) {
    std::cout << "Octagon: created\n";
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
    std::cout << "Octagon: created\n";
}

Octagon::Octagon(const Octagon& orig) {   
    std::cout << "Octagon: created via copy ctr" << std::endl;
    side = orig.side;
}

double Octagon::Square() {
    std::cout << "Octagon: square: ";
    return double(2.0 * side * side * (1.0 + sqrt(2.0)));
}

void Octagon::Print() {
    std::cout << "Octagon: side length = " << side << std::endl;
}

Octagon::~Octagon() {
    std::cout << "Octagon: deleted " << std::endl;
}
