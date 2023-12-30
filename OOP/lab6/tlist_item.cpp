#include "tlist_item.h"

template <class T>
TListItem<T>::TListItem(const T &val, TListItem<T> *item)
{
	value = new T(val);
	next = item;
}

template <class T>
void TListItem<T>::Push(const T &val)
{
	*value = val;
}

template <class T>
T &TListItem<T>::Pop() const
{
	return *value;
}

template <class T>
void TListItem<T>::SetNext(TListItem<T> *item)
{
	next = item;
}

template <class T>
TListItem<T> &TListItem<T>::GetNext() const
{
	return *next;
}

template <class T>
TListItem<T>::~TListItem()
{
	delete value;
}


typedef unsigned char TByte;

template class
TListItem<void *>;
