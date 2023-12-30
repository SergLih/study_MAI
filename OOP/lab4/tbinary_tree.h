#ifndef TBINARY_TREE_H
#define TBINARY_TREE_H

#include "pentagon.h"
#include <memory>

template <class T>
class TreeNode;

template <class T>
using TreeNodePtr = std::shared_ptr<TreeNode<T> >;

template<class T>
class TreeNode {
public:
	TreeNode();
	TreeNode(std::shared_ptr<T> data);
	TreeNodePtr<T> left;
	TreeNodePtr<T> right;
	std::shared_ptr<T> data;
};

template <class T> 
class TBinaryTree {
    private:
        TreeNodePtr<T> root;
		TreeNodePtr<T> MinValueTreeNode(TreeNodePtr<T> node);
		TreeNodePtr<T> deleteTreeNode(TreeNodePtr<T> _root, std::shared_ptr<T> key);
		std::ostream& InOrderPrint(std::ostream& os, TreeNodePtr<T> node, size_t level) const;
    public:
        TBinaryTree();
        TBinaryTree(const TBinaryTree<T>& orig);
		void Insert(std::shared_ptr<T> figure);
        bool IsEmpty() const;
        void Delete(std::shared_ptr<T> key);
		std::shared_ptr<T> Find(std::shared_ptr<T> key);

        template <class A> friend std::ostream& operator<<(std::ostream& os, TBinaryTree<A> & bintree);
        virtual ~TBinaryTree();
};

template <class T>
TBinaryTree<T>::TBinaryTree() {
	root = nullptr;
}

template <class T>
void TBinaryTree<T>::Insert(std::shared_ptr<T> val) {
	if (root == nullptr)
		root = TreeNodePtr<T>(new TreeNode<T>(val));
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
			parNode->left = TreeNodePtr<T>(new TreeNode<T>(val));
		else
			parNode->right = TreeNodePtr<T>(new TreeNode<T>(val));
	}
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
}

template <class A>
std::ostream& operator<<(std::ostream& os, TBinaryTree<A> & bintree) {
	if (bintree.IsEmpty())
		os << "Empty tree\n";
	else
		os << "===============================================\n";
	bintree.InOrderPrint(os, bintree.root, 0);
	os << "===============================================\n\n";
	return os;
}

template <class T>
bool TBinaryTree<T>::IsEmpty() const {
	return (root == nullptr);
}

template <class T>
TBinaryTree<T>::TBinaryTree(const TBinaryTree<T> & orig) {
	std::cout << "not impl\n";
}

template <class T>
TBinaryTree<T>::~TBinaryTree() {
	root = nullptr;
#ifdef DEBUG
	std::cout << "bintree dtr\n";
#endif // DEBUG 
}


template <class T>
TreeNode<T>::TreeNode() {
#ifdef DEBUG
	std::cout << "TreeNode: default ctr ";
#endif 
	left = nullptr;
	right = nullptr;
}

template<class T>
TreeNode<T>::TreeNode(std::shared_ptr<T> data)
{
	left = nullptr;
	right = nullptr;
	this->data = data;
}

#endif /* TBINARY_TREE_H */
