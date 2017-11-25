#ifndef TLIST_H
#define TLIST_H

#include <cstdint>
#include "tlist_item.h"

template <class T>
class TList
{
public:
	TList()
	{
		head = nullptr;
		count = 0;
	}
	virtual ~TList()
	{
		for (TListItem<T> *tmp = head, *tmp2; tmp; tmp = tmp2) {
			tmp2 = &tmp->GetNext();
			delete tmp;
		}
	}

	void Push(const T &item)
	{
		TListItem<T> *tmp = new TListItem<T>(item, head);
		head = tmp;
		++count;
	}

	void Pop()
	{
		if (head) {
			TListItem<T> *tmp = &head->GetNext();
			delete head;
			head = tmp;
			--count;
		}
	}

	T &Top()
	{
		return head->Pop();	//см. ListItem
	}

	bool IsEmpty() const
	{
		return !count;
	}
	size_t GetLength() const
	{
		return count;
	}

	TListIterator<T> begin()
	{
		return TListIterator<T>(head);
	}

	TListIterator<T> end()
	{
		return TListIterator<T>(nullptr);
	}

private:
	TListItem<T> *head;
	size_t count;
};

#endif

//template<class A>
//std::ostream & operator<<(std::ostream & os, const TList<A>& list)
//{
//	return os;
//}
