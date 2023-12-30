#ifndef TLIST_ALLOC_H
#define TLIST_ALLOC_H

#include <cstdint>
#include "tlist_alloc_item.h"

template <class T>
class TListAllocItem;

template <class T>
class TListAlloc
{
public:
	TListAlloc();
	virtual ~TListAlloc();

	void Push(const T &item);
	void Pop();
	T &Top();
	bool IsEmpty() const;
	size_t GetLength() const;

private:
	TListAllocItem<T> *head;
	size_t count;
};

template <class T>
TListAlloc<T>::TListAlloc()
{
	head = nullptr;
	count = 0;
}

template <class T>
void TListAlloc<T>::Push(const T &item)
{
	TListAllocItem<T> *tmp = new TListAllocItem<T>(item, head);
	head = tmp;
	++count;
}

template <class T>
bool TListAlloc<T>::IsEmpty() const
{
	return !count;
}

template <class T>
size_t TListAlloc<T>::GetLength() const
{
	return count;
}

template <class T>
void TListAlloc<T>::Pop()
{
	if (head) {
		TListAllocItem<T> *tmp = &head->GetNext();
		delete head;
		head = tmp;
		--count;
	}
}

template <class T>
T &TListAlloc<T>::Top()
{
	return head->Pop();
}

template <class T>
TListAlloc<T>::~TListAlloc()
{
	for (TListAllocItem<T> *tmp = head, *tmp2; tmp; tmp = tmp2) {
		tmp2 = &tmp->GetNext();
		delete tmp;
	}
}

#endif