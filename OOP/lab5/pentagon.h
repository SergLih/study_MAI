#ifndef PENTAGON_H
#define PENTAGON_H
#include <cstdlib>
#include <iostream>
#include "figure.h"

class Pentagon : public Figure {
	public:
	    Pentagon();
		Pentagon(size_t side);
		Pentagon(std::istream &is);
		Pentagon(const Pentagon& orig);
		double Square() override;

    friend std::ostream& operator<<(std::ostream& os, const Pentagon& obj);
    friend std::istream& operator>>(std::istream& is,  Pentagon& obj);
		void print(std::ostream& os) const;
    Pentagon& operator=(const Pentagon& right);
		
		virtual ~Pentagon();
	private:
		int side;
};

#endif   /*  Pentagon_H  */
