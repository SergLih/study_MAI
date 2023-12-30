#ifndef TBINARY_TREE_H
#define TBINARY_TREE_H

#include "pentagon.h"
#include "hexagon.h"
#include "octagon.h"

typedef Pentagon TItem;

class TBinaryTree {
    private:
        class TreeNode {
            public:
                TreeNode();
				TreeNode(std::shared_ptr<Figure> data);
                std::shared_ptr<TreeNode> left;
                std::shared_ptr<TreeNode> right;
                std::shared_ptr<Figure> data;
        };      
        
        std::shared_ptr<TreeNode> root;
		void DeleteNode(std::shared_ptr<TreeNode> node);
    public:
        TBinaryTree();
        TBinaryTree(const TBinaryTree& orig);
		void Insert(std::shared_ptr<Figure> val);
        bool IsEmpty() const;
		void DeleteNode(std::shared_ptr<Figure> key);
		std::shared_ptr<TreeNode> MinValueTreeNode(std::shared_ptr<TreeNode> node);
		std::shared_ptr<TreeNode> deleteTreeNode(std::shared_ptr<TreeNode> _root, std::shared_ptr<Figure>  key);
		//удаление в бинарном дереве пятиугольника с размером side
        std::ostream& InOrderPrint(std::ostream& os, std::shared_ptr<TreeNode> node, size_t level) const;
        friend std::ostream& operator<<(std::ostream& os, const TBinaryTree &bintree);
        virtual ~TBinaryTree();

		std::shared_ptr<Figure> Find(std::shared_ptr<Figure> key);
    
};

#endif /* TBINARY_TREE_H */
