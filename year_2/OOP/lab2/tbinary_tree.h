#ifndef TBINARY_TREE_H
#define TBINARY_TREE_H

#include "pentagon.h"

typedef Pentagon TItem;

class TBinaryTree {
    private:
        class TreeNode {
            public:
                TreeNode();
				TreeNode(const TItem & data);
                TreeNode *left;
                TreeNode *right;
                TItem data;
        };      
        
        TreeNode *root;
		void DeleteNode(TreeNode * node);
    public:
        TBinaryTree();
        TBinaryTree(const TBinaryTree& orig);
		void Insert(const TItem & val);
        bool IsEmpty() const;
        TItem* Find(size_t side);   //поиск в бинарном дереве пятиугольника с размером side
        bool Delete(size_t side);
		TreeNode * MinValueTreeNode(TreeNode * node);
		TreeNode * deleteTreeNode(TreeNode * _root, TItem & key);
		//удаление в бинарном дереве пятиугольника с размером side
        std::ostream& InOrderPrint(std::ostream& os, TreeNode * node, size_t level) const;
        friend std::ostream& operator<<(std::ostream& os, const TBinaryTree& bintree);
        virtual ~TBinaryTree();
    
};

#endif /* TBINARY_TREE_H */
