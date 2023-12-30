#ifndef TITERATOR_H
#define TITERATOR_H

#include "tbinary_tree.h"

template <class T>
class TreeNode;

template <class T>
class TBinaryTree;

template <class T>
using TreeNodePtr = std::shared_ptr<TreeNode<T> >;

template <class T>
class TBinaryTreeIterator {
public:
	TBinaryTreeIterator(TreeNodePtr<T> n) {
		treeNodePtr = n;
		if (!treeNodePtr)
			return;

		while (treeNodePtr->left)
			treeNodePtr = treeNodePtr->left;
	}

	std::shared_ptr<T> operator * () {
		return treeNodePtr->GetPtr();
	}
	std::shared_ptr<T> operator -> () {
		return treeNodePtr->GetPtr();
	}
	void operator ++ () {
		TreeNodePtr<T> res = treeNodePtr;
		if (res->right) {
			res = res->right;
			while (res->left)
				res = res->left;
		}
		else
		{
			while (true) {
				if (!res->parent) {
					res = nullptr;
					break;
				}
				if (res->parent->left == res) {
					res = res->parent;
					break;
				}
				res = res->parent;
			}
		}
		treeNodePtr = res;
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
