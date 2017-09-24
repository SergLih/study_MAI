#include "hexagon.h"
#include <iostream>
#include <cmath>

Hexagon::Hexagon() : Hexagon(0) {
}

Hexagon::Hexagon(size_t side) {
    std::cout << "Hexagon: created\n";
}

Hexagon::Hexagon(std::istream &is) {
    long long tmp_side;
    std::cout << "Hexagon: enter side length: ";
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
    std::cout << "Hexagon: created\n";
}

Hexagon::Hexagon(const Hexagon& orig) {
    std::cout << "Hexagon: created via copy ctr" << std::endl;
    side = orig.side;
}

double Hexagon::Square() {
    std::cout << "Hexagon: square: ";
    return double((6.0 * side * side) / (4.0 * tan(M_PI / 6.0)));
}

void Hexagon::Print() {
    std::cout << "Hexagon: side length = " << side << std::endl;
}

Hexagon::~Hexagon() {
    std::cout << "Hexagon: deleted " << std::endl;
}
