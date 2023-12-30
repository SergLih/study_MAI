#ifndef OCTAGON_H
#define OCTAGON_H
#include <cstdlib>
#include <iostream>
#include "figure.h"

class Octagon : public Figure {
	public:
	    Octagon();
	    Octagon(std::istream &is);
		Octagon(size_t side);
		Octagon(const Octagon& orig);
		double Square() override;
        
        friend std::ostream& operator<<(std::ostream& os, const Octagon& obj);
        friend std::istream& operator>>(std::istream& is,  Octagon& obj);
        void print(std::ostream& os) const;
        
        Octagon& operator=(const Octagon& right);
        
        friend bool operator==(const Octagon& left,const Octagon& right);
        friend bool operator>(const Octagon& left,const Octagon& right);
        friend bool operator<(const Octagon& left,const Octagon& right);
        friend bool operator!=(const Octagon& left,const Octagon& right);
        friend bool operator<=(const Octagon& left,const Octagon& right);
        friend bool operator>=(const Octagon& left,const Octagon& right);
		
		virtual ~Octagon();
	private:
		size_t side; //у правильного пятиугольника все стороны равны
};

#endif   /*  Octagon_H  */
