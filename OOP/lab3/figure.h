#ifndef FIGURE_H
#define FIGURE_H

#include <iostream>
#include <memory>
//#define DEBUG 0

class Figure {
	public:
        virtual void print(std::ostream& os) const;
		virtual double Square() = 0;
		bool TypedEquals(std::shared_ptr<Figure> other);
		bool SquareLess(std::shared_ptr<Figure> other);
		bool SquareGreater(std::shared_ptr<Figure> other);
		virtual ~Figure() {};
};
	

#endif  /*  FIGURE_H  */
