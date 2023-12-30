#include <iostream>
#include <cstdlib>
//#define DEBUG 1
#include "tbinary_tree.h"

TBinaryTree::TBinaryTree() {
    root = nullptr;
}

void TBinaryTree::Insert(const TItem& val){
	if (root == nullptr)
		root = new TreeNode(val);
	else {
		TreeNode * curNode = root;
		TreeNode * parNode = nullptr;
		while (curNode != nullptr) {
		 	parNode = curNode;
			if (val < curNode->data)
				curNode = curNode->left;
			else
				curNode = curNode->right;
		}
		if (val < parNode->data)
			parNode->left = new TreeNode(val);
		else
			parNode->right = new TreeNode(val);
	}
}

TItem * TBinaryTree::Find(size_t side)
{
	if (root == nullptr)
		return nullptr;
	else
	{
		Pentagon penta_search(side);
		TreeNode * curNode = root;
		while (curNode != nullptr) {
			if (penta_search < curNode->data)
				curNode = curNode->left;
			else if (penta_search > curNode->data)
				curNode = curNode->right;
			else
				return &(curNode->data);
		}
	}
	return nullptr;
}

TBinaryTree::TreeNode * TBinaryTree::MinValueTreeNode(TreeNode* node)
{
	TreeNode* current = node;

	while (current->left != nullptr)
		current = current->left;

	return current;
}

void TBinaryTree::DeleteNode(TreeNode * node)
{
	if (node == nullptr)
		return;

	DeleteNode(node->left);
	DeleteNode(node->right);
	delete node;
}

TBinaryTree::TreeNode*  TBinaryTree::deleteTreeNode(TreeNode* _root, TItem & key)
{
	if (_root == nullptr) return _root;

	if (key < _root->data)
		_root->left = deleteTreeNode(_root->left, key);
	else if (key > _root->data)
		_root->right = deleteTreeNode(_root->right, key);
	else {
		if (_root->left == nullptr) {
			TreeNode *temp = _root->right;
			delete _root;
			return temp;
		}
		else if (_root->right == nullptr) {
			TreeNode *temp = _root->left;
			delete _root;
			return temp;
		}

		TreeNode* temp = MinValueTreeNode(_root->right);
		_root->data = temp->data;
		_root->right = deleteTreeNode(_root->right, temp->data);
	}
	return _root;
}

bool TBinaryTree::Delete(size_t side)
{
	Pentagon penta_search(side);
	root = deleteTreeNode(root, penta_search);
	return false;
}

std::ostream& TBinaryTree::InOrderPrint(std::ostream& os, TreeNode * node, size_t level) const{
    if(node != nullptr){
        InOrderPrint(os, node->left, level + 1);
        for(size_t i = 0; i < level; i++)
            os << '\t';
        os << node->data;
        InOrderPrint(os, node->right, level + 1);
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const TBinaryTree& bintree){
	if (bintree.IsEmpty())
		os << "Empty tree\n";
	else
		os << "===============================================\n";
        bintree.InOrderPrint(os, bintree.root, 0);
		os << "===============================================\n\n";
    return os;
}

bool TBinaryTree::IsEmpty() const{
    return (root == nullptr);
}

TBinaryTree::TBinaryTree(const TBinaryTree& orig){
    std::cout << "not impl\n";
}

TBinaryTree::~TBinaryTree() {
	DeleteNode(root);
	root = nullptr;
#ifdef DEBUG
	std::cout << "bintree dtr\n";
#endif // DEBUG 
}



TBinaryTree::TreeNode::TreeNode() {
#ifdef DEBUG
	std::cout << "TreeNode: default ctr ";
#endif // DEBUG

    left = nullptr;
    right = nullptr;
}

TBinaryTree::TreeNode::TreeNode(const TItem& data){
#ifdef DEBUG
	std::cout << "TreeNode: ctr with data\n";
#endif // DEBUG

    left = nullptr;
    right = nullptr;
    this->data = data;	//перегруженный оператор = по сути конструктор копирования
}
