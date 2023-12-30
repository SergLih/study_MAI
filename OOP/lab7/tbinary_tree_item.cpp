#include <iostream>
#include <cstdlib>
#include <memory>

#include "tbinary_tree.h"

//это инициализация, то есть присваивание начального значения статическому полю класса TreeNode.
template <class T> TAllocationBlock	// Она происходит только один раз, так как статическое поле - единое для всех объектов этого класса.
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
TreeNode<T>::TreeNode(std::shared_ptr<T> data, TreeNodePtr<T> parent)
{
	left = nullptr;
	right = nullptr;
	this->parent = parent;
	this->data = data;
}

template<class T>
std::shared_ptr<T> TreeNode<T>::GetPtr()
{
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

template class TreeNode<Figure>;
