#include "figure.h"


void Figure::print(std::ostream & os) const
{
}

bool Figure::TypedEquals(std::shared_ptr<Figure> other) const
{
	return (this->Square() == other->Square() && typeid(*this) == typeid(*other.get()));
}

bool Figure::SquareLess(std::shared_ptr<Figure> other) const
{
	return (this->Square() < other->Square());
}

bool Figure::SquareGreater(std::shared_ptr<Figure> other) const
{
	return this->Square() > other->Square();
}

std::ostream & operator<<(std::ostream & os, const Figure & obj)
{
	obj.print(os);
	return os;
}
