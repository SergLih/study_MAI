#ifndef FIGURE_H
#define FIGURE_H
#include <iostream>
#include <memory>

//#define DEBUG 1

class Figure {
	public:
		virtual void print(std::ostream& os) const = 0;
		virtual double Square() const = 0;
		bool TypedEquals(std::shared_ptr<Figure> other) const;
		bool SquareLess(std::shared_ptr<Figure> other) const;
		bool SquareGreater(std::shared_ptr<Figure> other) const;
		virtual ~Figure() {};

		friend std::ostream& operator<<(std::ostream& os, const Figure& obj);
};

#endif  /*  FIGURE_H  */
