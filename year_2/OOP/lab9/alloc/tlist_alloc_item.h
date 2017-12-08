#ifndef TLISTALLOCITEM_H
#define TLISTALLOCITEM_H

template <class T>
class TListAllocItem
{
public:
	TListAllocItem(const T &val, TListAllocItem<T> *item);
	virtual ~TListAllocItem();

	void Push(const T &val);
	T &Pop() const;
	void SetNext(TListAllocItem<T> *item);
	TListAllocItem<T> &GetNext() const;

private:
	T *value;
	TListAllocItem<T> *next;
};

template <class T>
TListAllocItem<T>::TListAllocItem(const T &val, TListAllocItem<T> *item)
{
	value = new T(val);
	next = item;
}

template <class T>
void TListAllocItem<T>::Push(const T &val)
{
	*value = val;
}

template <class T>
T &TListAllocItem<T>::Pop() const
{
	return *value;
}

template <class T>
void TListAllocItem<T>::SetNext(TListAllocItem<T> *item)
{
	next = item;
}

template <class T>
TListAllocItem<T> &TListAllocItem<T>::GetNext() const
{
	return *next;
}

template <class T>
TListAllocItem<T>::~TListAllocItem()
{
	delete value;
}

#endif