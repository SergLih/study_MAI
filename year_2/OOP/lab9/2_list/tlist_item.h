#ifndef TSTACKITEM_H
#define TSTACKITEM_H
#include <memory> 

template<class T>
class TListItem;

template <class T>
using TListItemPtr = std::shared_ptr<TListItem<T> >;

template<class T>
class TListItem {
	public:
		TListItem(T *item)
		{
			this->item = std::shared_ptr<T>(item);
			this->next = nullptr;
		}

		TListItem(std::shared_ptr<T> item)
		{
			this->item = item;
			this->next = nullptr;
		}

		friend std::ostream& operator<<(std::ostream& os, const TListItem<T>& obj)
		{
			os << *(obj.GetValue());
			return os;
		}

		TListItemPtr<T> SetNext(TListItemPtr<T> next)
		{
			TListItemPtr<T> old = this->next;
			this->next = next;
			return old;
		}

		TListItemPtr<T> GetNext()
		{
			return this->next;
		}
		std::shared_ptr<T> GetValue() const
		{
			return this->item;
		}
		virtual ~TListItem() {/*nothing need to be done, shared_ptrs make all the work!*/}
	private:
		std::shared_ptr<T> item;
		TListItemPtr<T> next;
};
#endif