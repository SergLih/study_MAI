#include <iostream>
#include <cstdlib>
#include <memory>

#include "figure.h"
#include "tbinary_tree.h"
#include "tbinary_tree_item.h"

template <class T>
TBinaryTree<T>::TBinaryTree() {
	root = nullptr;
	count = 0;
}

template <class T>
void TBinaryTree<T>::Insert(std::shared_ptr<T> val) {
	if (root == nullptr) {
		root = TreeNodePtr<T>(new TreeNode<T>(val));
	}
	else {
		TreeNodePtr<T> curNode = root;
		TreeNodePtr<T> parNode = nullptr;
		while (curNode != nullptr) {
			parNode = curNode;
			if (val->SquareLess(curNode->data))
				curNode = curNode->left;
			else
				curNode = curNode->right;
		}
		if (val->SquareLess(parNode->data))
			parNode->left = TreeNodePtr<T>(new TreeNode<T>(val, parNode));
		else
			parNode->right = TreeNodePtr<T>(new TreeNode<T>(val, parNode));
	}
	count++;
}


template <class T>
std::shared_ptr<T> TBinaryTree<T>::Find(std::shared_ptr<T> key)
{
	if (root == nullptr)
		return nullptr;

	TreeNodePtr<T> curNode = root;
	TreeNodePtr<T> parNode = nullptr;
	while (curNode != nullptr) {
		parNode = curNode;
		if (key->TypedEquals(curNode->data))
			return curNode->data;
		if (key->SquareLess(curNode->data))
			curNode = curNode->left;
		else
			curNode = curNode->right;
	}
	return nullptr;
}

template<class T>
size_t TBinaryTree<T>::GetCount()
{
	return count;
}

template<class T>
TBinaryTreeIterator<T> TBinaryTree<T>::begin()
{
	return TBinaryTreeIterator<T>(root);
}

template<class T>
TBinaryTreeIterator<T> TBinaryTree<T>::end()
{
	return TBinaryTreeIterator<T>(nullptr);//nullptr;
}


template<class T>
TreeNodePtr<T> TBinaryTree<T>::MinValueTreeNode(TreeNodePtr<T> node)
{
	TreeNodePtr<T> current = node;

	while (current->left != nullptr)
		current = current->left;

	return current;
}

template<class T>
TreeNodePtr<T> TBinaryTree<T>::deleteTreeNode(TreeNodePtr<T> _root, std::shared_ptr<T> key)
{
	if (_root == nullptr) return _root;
	if (key->TypedEquals(_root->data)) {
		if (_root->left == nullptr) {
			TreeNodePtr<T> temp = _root->right;
			return temp;
		}
		else if (_root->right == nullptr) {
			TreeNodePtr<T> temp = _root->left;
			return temp;
		}

		TreeNodePtr<T> temp = MinValueTreeNode(_root->right);
		_root->data = temp->data;
		_root->right = deleteTreeNode(_root->right, temp->data);
	}
	else if (key->SquareLess(_root->data))
		_root->left = deleteTreeNode(_root->left, key);
	else
		_root->right = deleteTreeNode(_root->right, key);

	return _root;
}

template<class T>
std::ostream & TBinaryTree<T>::InOrderPrint(std::ostream & os, TreeNodePtr<T> node, size_t level) const
{
	if (node != nullptr) {
		InOrderPrint(os, node->left, level + 1);
		for (size_t i = 0; i < level; i++)
			os << '\t';
		node->data->print(os);
		InOrderPrint(os, node->right, level + 1);
	}
	return os;
}

template <class T>
void TBinaryTree<T>::Delete(std::shared_ptr<T> key)
{
	if (key == nullptr)
		return;
	root = deleteTreeNode(root, key);
	count--;
}

template <class A>
std::ostream& operator<<(std::ostream& os, TBinaryTree<A> & bintree) {
	if (bintree.IsEmpty())
		os << "Empty tree\n";
	else
		os << "==============================================\n";
	bintree.InOrderPrint(os, bintree.root, 0);
	os << "==============================================\n\n";
	return os;
}

template <class T>
bool TBinaryTree<T>::IsEmpty() const {
	return (root == nullptr);
}

template <class T>
TBinaryTree<T>::~TBinaryTree() {
	root = nullptr;
#ifdef DEBUG
	std::cout << "bintree dtr\n";
#endif // DEBUG 
}

template class TBinaryTree<Figure>; 
template std::ostream& operator<<(std::ostream& os, TBinaryTree<Figure>& tree);
