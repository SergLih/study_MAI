#ifndef PENTAGON_H
#define PENTAGON_H
#include <cstdlib>
#include <iostream>
#include "figure.h"

class Pentagon : public Figure {
	public:
	    Pentagon();
	    Pentagon(std::istream &is);
		Pentagon(size_t side);
		Pentagon(const Pentagon& orig);
		double Square() override;
        
        friend std::ostream& operator<<(std::ostream& os, const Pentagon& obj);
        friend std::istream& operator>>(std::istream& is,  Pentagon& obj);
        void print(std::ostream& os) const;
        
        Pentagon& operator=(const Pentagon& right);
        
        friend bool operator==(const Pentagon& left,const Pentagon& right);
        friend bool operator>(const Pentagon& left,const Pentagon& right);
        friend bool operator<(const Pentagon& left,const Pentagon& right);
        friend bool operator!=(const Pentagon& left,const Pentagon& right);
        friend bool operator<=(const Pentagon& left,const Pentagon& right);
        friend bool operator>=(const Pentagon& left,const Pentagon& right);
		
		virtual ~Pentagon();
	private:
		size_t side; //у правильного пятиугольника все стороны равны
};

#endif   /*  Pentagon_H  */
