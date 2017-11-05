#ifndef TITERATOR_H
#define TITERATOR_H

#include "common.h"
#include "tbinary_tree.h"

template <class T>
class TBinaryTreeIterator {
public:
	TBinaryTreeIterator(TreeNodePtr<T> n) {
		treeNodePtr = n;
	}
	std::shared_ptr<T> operator * () {
		return treeNodePtr->GetPtr();
	}
	std::shared_ptr<T> operator -> () {
		return treeNodePtr->GetPtr();
	}
	void operator ++ () {
		treeNodePtr = treeNodePtr->GetNext();
	}
	TBinaryTreeIterator operator ++ (int) {
		TBinaryTreeIterator iter(*this);
		++(*this);
		return iter;
	}
	bool operator == (TBinaryTreeIterator const& i) {
		return treeNodePtr == i.treeNodePtr;
	}
	bool operator != (TBinaryTreeIterator const& i) {
		return !(*this == i);
	}
private:
	TreeNodePtr<T> treeNodePtr;
};
#endif 
