#ifndef TBINARY_TREE_H
#define TBINARY_TREE_H

#include "figure.h"
#include "tbinarytree_iterator.h"
#include "tbinary_tree_item.h"
#include <memory>

template <class T>
class TreeNode;

template <class T>
using TreeNodePtr = std::shared_ptr<TreeNode<T> >;

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

		TBinaryTreeIterator<T> begin();
		TBinaryTreeIterator<T> end();

        template <class A> friend std::ostream& operator<<(std::ostream& os, TBinaryTree<A> & bintree);
        virtual ~TBinaryTree();
};

#endif /* TBINARY_TREE_H */
