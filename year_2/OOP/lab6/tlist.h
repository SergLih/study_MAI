#ifndef TLIST_H
#define TLIST_H

#include <cstdint>
#include "tlist_item.h"

template <class T>
class TListItem;

template <class T>
class TList
{
public:
	TList();
	virtual ~TList();

	void Push(const T &item);
	void Pop();
	T &Top();
	bool IsEmpty() const;
	size_t GetLength() const;

	template <class A> friend std::ostream& operator<<(std::ostream &os, const TList<A> &stack);

private:
	TListItem<T> *head;
	size_t count;
};

#endif
