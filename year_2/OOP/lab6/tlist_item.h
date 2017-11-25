#ifndef TLISTITEM_H
#define TLISTITEM_H

template <class T>
class TListItem
{
public:
	TListItem(const T &val, TListItem<T> *item);
	virtual ~TListItem();

	void Push(const T &val);
	T &Pop() const;
	void SetNext(TListItem<T> *item);
	TListItem<T> &GetNext() const;

private:
	T *value;
	TListItem<T> *next;
};

#endif
