#include <iostream>
#include <cstdlib>
#include <memory>
//#define DEBUG 1
#include "tbinary_tree.h"

TBinaryTree::TBinaryTree() {
    root = nullptr;
}

void TBinaryTree::Insert(std::shared_ptr<Figure> val){
	if (root == nullptr) {
		root = std::make_shared<TBinaryTree::TreeNode>(val);
		return;
	} else {
		std::shared_ptr<TBinaryTree::TreeNode> curNode = root;
		std::shared_ptr<TBinaryTree::TreeNode> parNode = nullptr;
		while (curNode != nullptr) {
		 	parNode = curNode;
			if (val->SquareLess(curNode->data))
				curNode = curNode->left;
			else 
				curNode = curNode->right;
		}
		if (val->SquareLess(parNode->data))
			parNode->left = std::make_shared<TBinaryTree::TreeNode>(val);
		else
			parNode->right = std::make_shared<TBinaryTree::TreeNode>(val);
	}
}

std::shared_ptr<TBinaryTree::TreeNode> TBinaryTree::MinValueTreeNode(std::shared_ptr<TBinaryTree::TreeNode> node)
{
	std::shared_ptr<TBinaryTree::TreeNode> current = node;

	while (current->left != nullptr)
		current = current->left;

	return current;
}

void TBinaryTree::DeleteNode(std::shared_ptr<Figure> key)
{
	if (key == nullptr)
		return;

	root = deleteTreeNode(root, key);
	
}

std::shared_ptr<TBinaryTree::TreeNode>  TBinaryTree::deleteTreeNode(std::shared_ptr<TBinaryTree::TreeNode> _root, std::shared_ptr<Figure> key)
{
	if (_root == nullptr) return _root;

	if (key->TypedEquals(_root->data)) {
		if (_root->left == nullptr) {
			std::shared_ptr<TBinaryTree::TreeNode> temp = _root->right;
			return temp;
		}
		else if (_root->right == nullptr) {
			std::shared_ptr<TBinaryTree::TreeNode> temp = _root->left;
			return temp;
		}

		std::shared_ptr<TBinaryTree::TreeNode> temp = MinValueTreeNode(_root->right);
		_root->data = temp->data;
		_root->right = deleteTreeNode(_root->right, temp->data);
	}
	else if (key->SquareLess(_root->data))
		_root->left = deleteTreeNode(_root->left, key);
	else		//(key->SquareGreater(_root->data))
		_root->right = deleteTreeNode(_root->right, key);

	return _root;
}


std::ostream& TBinaryTree::InOrderPrint(std::ostream& os, std::shared_ptr<TBinaryTree::TreeNode> node, size_t level) const{
    if(node != nullptr){
        InOrderPrint(os, node->left, level + 1);
        for(size_t i = 0; i < level; i++)
            os << '\t';
        (node->data)->print(os);
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

TBinaryTree::TreeNode::TreeNode(std::shared_ptr<Figure> data){
#ifdef DEBUG
	std::cout << "TreeNode: ctr with data\n";
#endif // DEBUG

    left = nullptr;
    right = nullptr;
    this->data = data;	//перегруженный оператор = по сути конструктор копирования
}


std::shared_ptr<Figure> TBinaryTree::Find(std::shared_ptr<Figure> key) {
	if (root == nullptr) return nullptr;

	std::shared_ptr<TBinaryTree::TreeNode> curNode = root;
	std::shared_ptr<TBinaryTree::TreeNode> parNode = nullptr;
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
