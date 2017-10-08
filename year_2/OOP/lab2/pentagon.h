#ifndef PENTAGON_H
#define PENTAGON_H
#include <cstdlib>
#include <iostream>
#include "figure.h"

class Pentagon : public Figure {
	public:
	    Pentagon();
		Pentagon(size_t side);
		Pentagon(const Pentagon& orig);
		double Square() override;

        friend bool operator==(const Pentagon& left,const Pentagon& right);
        friend bool operator>(const Pentagon& left,const Pentagon& right);
        friend bool operator<(const Pentagon& left,const Pentagon& right);
        friend bool operator!=(const Pentagon& left,const Pentagon& right);
        friend bool operator<=(const Pentagon& left,const Pentagon& right);
        friend bool operator>=(const Pentagon& left,const Pentagon& right);
        
        friend std::ostream& operator<<(std::ostream& os, const Pentagon& obj);
        friend std::istream& operator>>(std::istream& is,  Pentagon& obj);
        Pentagon& operator=(const Pentagon& right);
		
		virtual ~Pentagon();
	private:
		int side; //у правильного пятиугольника все стороны равны
};

#endif   /*  Pentagon_H  */
