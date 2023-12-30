#ifndef TLIST_H
#define TLIST_H

#include <cstdint>
#include "tlist_item.h"
#include "tlist_iterator.h"

//template <class T>
//class TListItem;

template <class T>
class TListIterator;

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

	void Delete(TListIterator<T> it)
	{
		TListItem<T> * node = it.ptr;
		if (node == nullptr)
			return;

		if (node == head)
		{
			Pop();
		}
		else
		{
			TListItem<T> * tmp = head, *tmp2;
			while (&tmp->GetNext() != node)
				tmp = &tmp->GetNext();
			tmp2 = &tmp->GetNext();
			tmp->SetNext(&tmp2->GetNext());
			delete tmp2;
			count--;
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

	//template <class A> friend std::ostream& operator<<(std::ostream &os, const TList<A> &stack);

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
