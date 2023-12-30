#ifndef TBINARY_TREE_H
#define TBINARY_TREE_H

#include "tbinarytree_iterator.h"
#include "tbinary_tree_item.h"
#include "common.h"

template <class T> 
class TBinaryTree {
    private:
        TreeNodePtr<T> root;
		mutable std::recursive_mutex tree_mutex;	//модификатор mutable применяется для того, чтобы можно было изменять эту переменную в const-функциях
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

		TBinaryTreeIterator<T> begin();
		TBinaryTreeIterator<T> end();

        template <class A> friend std::ostream& operator<<(std::ostream& os, TBinaryTree<A> & bintree);
        virtual ~TBinaryTree();
};

template <class T>
TBinaryTree<T>::TBinaryTree() {
	root = nullptr;
}

template <class T>
void TBinaryTree<T>::Insert(std::shared_ptr<T> val) {
	std::lock_guard<std::recursive_mutex> lock(tree_mutex);     
	if (root == nullptr) {
		root = TreeNodePtr<T>(new TreeNode<T>(val, &tree_mutex));
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
			parNode->left = TreeNodePtr<T>(new TreeNode<T>(val, &tree_mutex, parNode));
		else
			parNode->right = TreeNodePtr<T>(new TreeNode<T>(val, &tree_mutex, parNode));
	}
}


template <class T>
std::shared_ptr<T> TBinaryTree<T>::Find(std::shared_ptr<T> key)
{
	std::lock_guard<std::recursive_mutex> lock(tree_mutex);
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
TBinaryTreeIterator<T> TBinaryTree<T>::begin()
{
	return TBinaryTreeIterator<T>(root);
}

template<class T>
TBinaryTreeIterator<T> TBinaryTree<T>::end()
{
	return TBinaryTreeIterator<T>(nullptr);
}


template<class T>
TreeNodePtr<T> TBinaryTree<T>::MinValueTreeNode(TreeNodePtr<T> node)
{
	std::lock_guard<std::recursive_mutex> lock(tree_mutex);
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
	std::lock_guard<std::recursive_mutex> lock(tree_mutex);
	if (key == nullptr)
		return;
	root = deleteTreeNode(root, key);
}

template <class A>
std::ostream& operator<<(std::ostream& os, TBinaryTree<A> & bintree) {
	std::lock_guard<std::recursive_mutex> lock(bintree.tree_mutex);
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
	std::lock_guard<std::recursive_mutex> lock(tree_mutex);
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

#endif /* TBINARY_TREE_H */
