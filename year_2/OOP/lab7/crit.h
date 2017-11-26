#ifndef IREMOVECRITERIA_H 
#define IREMOVECRITERIA_H 

#include "figure.h"
#include <memory>
#include <typeindex>
#pragma warning(disable : 4996)  

template <class T> 
class IRemoveCriteria 
{
public:     
	virtual bool operator()(std::shared_ptr<T> value) = 0; 
};


class RemoveCriteriaByMaxSquare : public IRemoveCriteria<Figure>
{
public:     
	RemoveCriteriaByMaxSquare(double value)
	{
		_MaxSquareValue = value;
	}
	bool operator()(std::shared_ptr<Figure> value) override
	{
		return value->Square() < _MaxSquareValue;
	}
private:  
	double _MaxSquareValue;
};


class RemoveCriteriaByFigureType : public IRemoveCriteria<Figure>
{
public:
	RemoveCriteriaByFigureType(const char * value)
	{
		_TypeName = new char[strlen(value) + 1];
		strcpy(_TypeName, value);
	}

	bool operator()(std::shared_ptr<Figure> value) override
	{
		return strcmp(typeid(*value).name()+6, _TypeName)==0;
	}

	~RemoveCriteriaByFigureType()
	{
		delete _TypeName;
	}
private:
	char * _TypeName;
};


#endif
