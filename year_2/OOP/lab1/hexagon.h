#ifndef HEXAGON_H
#define HEXAGON_H
#include <cstdlib>
#include <iostream>
#include "figure.h"

class Hexagon : public Figure {
public:
    Hexagon();
    Hexagon(std::istream& is);
    Hexagon(size_t side);
    Hexagon(const Hexagon& orig);
    double Square() override;
    void   Print()  override;
    
    virtual ~Hexagon();
private:
    size_t side;
};

#endif   /*  Hexagon_H  */
