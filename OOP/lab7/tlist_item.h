#ifndef TLISTITEM_H
#define TLISTITEM_H

#include "tlist.h"
#include "tlist_iterator.h"

template <class T>
class TList;

template <class T>
class TListIterator;

template <class T>
class TListItem
{
public:
	TListItem(const T &val, TListItem<T> *item)
	{
		value = new T(val);
		next = item;
	}

	virtual ~TListItem()
	{
		delete value;
	}

	void Push(const T &val)
	{
		*value = val;
	}

	T &Pop() const
	{
		return *value;
	}

	void SetNext(TListItem<T> *item)
	{
		next = item;
	}
	
	TListItem<T> &GetNext() const
	{
		return *next;
	}

	friend class TList<T>;
	friend class TListIterator<T>;

private:
	T *value;
	TListItem<T> *next;
};

#endif
