#ifndef TBINARY_TREE_ITEM_H
#define TBINARY_TREE_ITEM_H

#include "../alloc/TAllocationBlock.h"
#include "common.h"

#define MAX_TREE_CAPACITY 100

template<class T>
class TreeNode {
public:
	TreeNode();
	TreeNode(std::shared_ptr<T> item, std::recursive_mutex *parent, TreeNodePtr<T> parentNode = nullptr );
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
	std::recursive_mutex *tree_mutex;
};

template <class T> TAllocationBlock	//происходит только один раз, так как это единое статическое поле для всех объектов класса
TreeNode<T>::allocator(sizeof(TreeNode<T>), MAX_TREE_CAPACITY);

template <class T>
TreeNode<T>::TreeNode() {
#ifdef DEBUG
	std::cout << "TreeNode: default ctr ";
#endif 
	left = nullptr;
	right = nullptr;
	parent = nullptr;
}

template<class T>
TreeNode<T>::TreeNode(std::shared_ptr<T> item, std::recursive_mutex *parent, TreeNodePtr<T> parentNode = nullptr) 
{
	left = nullptr;
	right = nullptr;
	this->tree_mutex = parent;
	this->parent = parentNode;
	this->data = item;
}

template<class T>
std::shared_ptr<T> TreeNode<T>::GetPtr()
{
	std::unique_lock<std::recursive_mutex> lock(*tree_mutex);
	return data;
}

template <class T>
void *TreeNode<T>::operator new(size_t size)
{
	return allocator.Allocate();
}

template <class T>
void TreeNode<T>::operator delete(void *ptr)
{
	allocator.Deallocate(ptr);
}

#endif /* TBINARY_TREE_ITEM_H */ 