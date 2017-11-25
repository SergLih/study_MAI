#ifndef TLIST_ITERATOR_H
#define TLIST_ITERATOR_H

#include "tlist.h"
#include "tlist_item.h"

template<class T>
class TListItem;

template <class T>
class TListIterator 
{
public:
	TListIterator(TListItem<T> *item) 
	{
		ptr = item;
	}

	T* operator * () 
	{
		return ptr->value;
	}

	T* operator -> () {
		return ptr->value;
	}

	void operator ++ () {
		ptr = ptr->next;
	}

	TListIterator operator ++ (int) {
		TListIterator iter(*this);
		++(*this);
		return iter;
	}
	bool operator == (TListIterator const& i) {
		return ptr == i.ptr;
	}

	bool operator != (TListIterator const& i) {
		return !(*this == i);
	}

private:
	TListItem<T> *ptr;
};
#endif
