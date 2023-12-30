#include <iostream>
#include <memory>
#include "tlist.h"

template <class T>
TList<T>::TList()
{
	head = nullptr;
	count = 0;
}

template <class T>
void TList<T>::Push(const T &item)
{
	TListItem<T> *tmp = new TListItem<T>(item, head);
	head = tmp;
	++count;
}

template <class T>
bool TList<T>::IsEmpty() const
{
	return !count;
}

template <class T>
size_t TList<T>::GetLength() const
{
	return count;
}

template <class T>
void TList<T>::Pop()
{
	if (head) {
		TListItem<T> *tmp = &head->GetNext();
		delete head;
		head = tmp;
		--count;
	}
}

template <class T>
T &TList<T>::Top()
{
	return head->Pop();
}

template <class T>
TList<T>::~TList()
{
	for (TListItem<T> *tmp = head, *tmp2; tmp; tmp = tmp2) {
		tmp2 = &tmp->GetNext();
		delete tmp;
	}
}

typedef unsigned char TByte;

template class
TList<void *>;
