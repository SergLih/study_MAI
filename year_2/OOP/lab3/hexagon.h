#ifndef HEXAGON_H
#define HEXAGON_H
#include <cstdlib>
#include <iostream>
#include "figure.h"

class Hexagon : public Figure {
	public:
	    Hexagon();
	    Hexagon(std::istream &is);
		Hexagon(size_t side);
		Hexagon(const Hexagon& orig);
		double Square() override;
        
        friend std::ostream& operator<<(std::ostream& os, const Hexagon& obj);
        friend std::istream& operator>>(std::istream& is,  Hexagon& obj);
        void print(std::ostream& os) const;
        
        Hexagon& operator=(const Hexagon& right);
        
        friend bool operator==(const Hexagon& left,const Hexagon& right);
        friend bool operator>(const Hexagon& left,const Hexagon& right);
        friend bool operator<(const Hexagon& left,const Hexagon& right);
        friend bool operator!=(const Hexagon& left,const Hexagon& right);
        friend bool operator<=(const Hexagon& left,const Hexagon& right);
        friend bool operator>=(const Hexagon& left,const Hexagon& right);
		
		virtual ~Hexagon();
	private:
		size_t side; //у правильного пятиугольника все стороны равны
};

#endif   /*  Hexagon_H  */
