#ifndef TLIST_H
#define TLIST_H 
#include "tlist_iterator.h"
#include "tlist_item.h"
#include <memory>
#include <future>
#include <mutex>

template <class T>
class TListItem;

template <class T>
using TListItemPtr = std::shared_ptr<TListItem<T> >;

template <class T> class TList {
	public:
		TList()
		{
			head = nullptr;
		}


		void Push(T* item)
		{
			TListItemPtr<T> other(new TListItem<T>(item));     
			other->SetNext(head);     
			head = other;
		}

		void Push(std::shared_ptr<T> item)
		{
			TListItemPtr<T> other(new TListItem<T>(item));
			other->SetNext(head);     
			head = other;
		}

		bool IsEmpty() const
		{
			return head == nullptr;
		}

		size_t Size()
		{
			size_t result = 0;     
			for (auto i : *this) 
				result++;     
			return result;
		}

		TListIterator<T> begin()
		{
			return TListIterator<T>(head);
		}

		TListIterator<T> end()
		{
			return TListIterator<T>(nullptr);
		}

		void Sort()
		{
			if (Size() > 1) {
				std::shared_ptr<T> middle = Pop();
				TList<T> left, right;
				while (!IsEmpty()) {
					std::shared_ptr<T> item = Pop();
					if (!item->SquareLess(middle)) {
						left.Push(item);
					} else {
						right.Push(item);
					}
				}
				left.Sort();
				right.Sort();
				while (!left.IsEmpty()) {
					Push(left.PopLast());
				}
				Push(middle);
				while (!right.IsEmpty()) {
					Push(right.PopLast());
				}
			}
		}

		void SortParallel()
		{
			if (Size() > 1) {
				std::shared_ptr<T> middle = PopLast();
				TList<T> left, right;
				while (!IsEmpty()) {
					std::shared_ptr<T> item = PopLast();
					if (!item->SquareLess(middle)) {
						left.Push(item);
					} else {
						right.Push(item);
					}
				}
				std::future<void> left_res = left.BackgroundSort();
				std::future<void> right_res = right.BackgroundSort();


				left_res.get();


				while (!left.IsEmpty()) {
					Push(left.PopLast());
				}
				Push(middle);
				right_res.get();
				while (!right.IsEmpty()) {
					Push(right.PopLast());
				}
			}
		}

		std::shared_ptr<T> Pop()
		{
			std::shared_ptr<T> result;     
			if (head != nullptr) { 
				result = head->GetValue();         
				head = head->GetNext(); 
			}
			return result;
		}

		std::shared_ptr<T> PopLast()
		{
			std::shared_ptr<T> result;
			if (head != nullptr) {
				TListItemPtr <T> element = head;
				TListItemPtr <T> prev = nullptr;
				while (element->GetNext() != nullptr) {
					prev = element;
					element = element->GetNext();
				}
				if (prev != nullptr) {
					prev->SetNext(nullptr);
					result = element->GetValue();
				} else {
					result = element->GetValue();
					head = nullptr;
				}
			}
			return result;
		}

		void Delete(std::shared_ptr<T> key)
		{
			bool found = false;
			if (head != nullptr) {
				TListItemPtr <T> element = head;
				TListItemPtr <T> prev = nullptr;
				while (element != nullptr) { 
					if (element->GetValue()->TypedEquals(key)) {	//found :)
						found = true;
						break;
					}
					prev = element;
					element = element->GetNext();
				}
				if (found) {
					if (prev != nullptr) {
						prev->SetNext(element->GetNext());
					}
					else {
						head = element->GetNext();
					}
				}
			}
		}
		
		template <class A> 
		friend std::ostream& operator<<(std::ostream& os, const TList<A>& list)
		{
			TListItemPtr<A> item = list.head;
			if (list.IsEmpty())
				os << "List is empty\n";
			while (item != nullptr) {
				os << *item;
				item = item->GetNext();
			}
			return os;
		}
		virtual ~TList() {/*nothing need to be done, shared_ptrs make all the work!*/}
	private:
		std::future<void> BackgroundSort()
		{
			std::packaged_task<void(void) > task(std::bind(std::mem_fn(&TList<T>::SortParallel), this));
			std::future<void> res(task.get_future());
			std::thread thr(std::move(task));
			thr.detach();
			return res;
		}
		TListItemPtr<T> head;
};
#endif  