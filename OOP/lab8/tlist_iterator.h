#ifndef TLISTITERATOR_H
#define TLISTITERATOR_H
#include <memory>
#include <iostream>

template <class T>
class TListItem;

template <class T>
using TListItemPtr = std::shared_ptr<TListItem<T> >;

template <class T> 
class TListIterator {
public:
	TListIterator(TListItemPtr<T> n) 
	{ 
		list_item_ptr = n; 
	}
	
	std::shared_ptr<T> operator * () 
	{ 
		return list_item_ptr->GetValue(); 
	}

	std::shared_ptr<T> operator -> () 
	{
		return list_item_ptr->GetValue(); 
	}
	void operator ++ () 
	{ 
		list_item_ptr = list_item_ptr->GetNext(); 
	}

	TListIterator operator ++ (int) 
	{ 
		TListIterator iter(*this);
		++(*this);
		return iter; 
	}

	bool operator == (TListIterator const& i) 
	{ 
		return list_item_ptr == i.list_item_ptr; 
	}

	bool operator != (TListIterator const& i)
	{
		return !(*this == i); 
	}
private:
	TListItemPtr<T> list_item_ptr;
};

#endif /* TITERATOR_H */ 
