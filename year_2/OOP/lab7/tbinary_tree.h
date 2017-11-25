#ifndef TBINARY_TREE_H
#define TBINARY_TREE_H

#include "tbinary_tree_item.h"
#include "TAllocationBlock.h"
#include "tbinarytree_iterator.h"

template <class T> 
class TBinaryTree 
{
	private:
		size_t count;
		TreeNodePtr<T> root;
		TreeNodePtr<T> MinValueTreeNode(TreeNodePtr<T> node);
		TreeNodePtr<T> deleteTreeNode(TreeNodePtr<T> _root, std::shared_ptr<T> key);
		std::ostream& InOrderPrint(std::ostream& os, TreeNodePtr<T> node, size_t level) const;
	public:
		TBinaryTree();
		void Insert(std::shared_ptr<T> figure);
		bool IsEmpty() const;
		void Delete(std::shared_ptr<T> key);
		std::shared_ptr<T> Find(std::shared_ptr<T> key);
		size_t GetCount();

		TBinaryTreeIterator<T> begin();
		TBinaryTreeIterator<T> end();

		template <class A> friend std::ostream& operator<<(std::ostream& os, TBinaryTree<A> & bintree);
		virtual ~TBinaryTree();
};

#endif /* TBINARY_TREE_H */
