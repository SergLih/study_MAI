#ifndef TBINARY_TREE_ITEM_H
#define TBINARY_TREE_ITEM_H

#include "figure.h"
#include "tbinarytree_iterator.h"
#include "TAllocationBlock.h"
#include <memory>

#define MAX_TREE_CAPACITY 100

template<class T>
class TreeNode {
public:
	TreeNode();
	TreeNode(std::shared_ptr<T> data, TreeNodePtr<T> parent = nullptr);
	std::shared_ptr<T> GetPtr();

	void *operator new(size_t size);
	void operator delete(void *ptr);

	friend class TBinaryTree<T>;
	friend class TBinaryTreeIterator<T>;
private:
	TreeNodePtr<T> left;
	TreeNodePtr<T> right;
	TreeNodePtr<T> parent;
	std::shared_ptr<T> data;

	static TAllocationBlock allocator;
};

#endif /* TBINARY_TREE_ITEM_H */ 
