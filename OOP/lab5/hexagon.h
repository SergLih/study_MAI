#ifndef HEXAGON_H
#define HEXAGON_H
#include <cstdlib>
#include <iostream>
#include "figure.h"

class Hexagon : public Figure {
public:
	Hexagon();
	Hexagon(size_t side);
	Hexagon(std::istream &is);
	Hexagon(const Hexagon& orig);
	double Square() override;

	friend std::ostream& operator<<(std::ostream& os, const Hexagon& obj);
	friend std::istream& operator>>(std::istream& is, Hexagon& obj);
	void print(std::ostream& os) const;
	Hexagon& operator=(const Hexagon& right);

	virtual ~Hexagon();
private:
	int side;
};

#endif   /*  Hexagon_H  */
