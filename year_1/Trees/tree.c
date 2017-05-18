#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "tree.h"

#define DEEP 0
#define max(x,y) ((x) > (y) ? (x) : (y))


struct _tree {
    TreeItem value;
    Tree child;
    Tree sibling;
};

Tree tree_create(TreeItem value)
{
    Tree tree = (Tree) malloc(sizeof(*tree));
    tree->value = value;
    tree->child = NULL;
    tree->sibling = NULL;
    
    return tree;
}

void tree_add_node(Tree tree, int parent, int value)
{
    if (tree) {
        Tree parent_node = tree_find(tree, parent);
        if (parent_node) {
            if (parent_node->child == NULL) {
                parent_node->child = tree_create(value);
            } else {
                parent_node = parent_node->child;
                while (parent_node->sibling) {
                    parent_node = parent_node->sibling;
                }
                parent_node->sibling = tree_create(value);
            }
        } else {
            printf("This parent vertex is not found.\n");
        }
    }
}


Tree tree_find(Tree tree, TreeItem c)
{
    if(!tree) {
        return NULL;
    }

    if(tree->value == c) {
        return tree;
    }

    Tree result = NULL;
    if(tree->child) {
        result = tree_find(tree->child, c);
        if(result) return result;
    }

    if(tree->sibling) {
        result = tree_find(tree->sibling, c);
        if(result) return result;
    }

}


void tree_print_node(Tree tree, int indent)
{
    for(int i = 0; i < indent; ++i) {
        printf("\t");
    }
    printf("%d\n", tree->value);
    if(tree->child) {
        tree_print_node(tree->child, indent + 1);
    }
    if(tree->sibling) {
        tree_print_node(tree->sibling, indent);
    }
}

void tree_print(Tree tree)
{
    tree_print_node(tree, 0);
}

void tree_destroy(Tree tree)
{

    if(tree->child) {
        tree_destroy(tree->child);
    }
    if(tree->sibling) {
        tree_destroy(tree->sibling);
    }
    free(tree);
    tree = NULL;
}


void tree_del_node(Tree tree, TreeItem value)
{
    if(tree->child) {
        if(tree->child->value == value) {
            Tree tmp = tree->child;
            tree->child = tree->child->sibling;
            if (tmp->child) {
                tree_destroy(tmp->child);
            }
            free(tmp);
            tmp = NULL;
            return;
        } else {
            tree_del_node(tree->child, value);
        }
    }


    if(tree->sibling) {
        if(tree->sibling->value == value) {
            Tree tmp = tree->sibling;
            tree->sibling = tree->sibling->sibling;
            if(tmp->child) {
                tree_destroy(tmp->child);
            }
            free(tmp);
            tmp = NULL;
            return;
        } else {
            tree_del_node(tree->sibling, value);
        }
    }
}

int depth_of_tree(Tree tree)
{
    if(tree == NULL) return 0;
    return max(1 + depth_of_tree(tree->child), depth_of_tree(tree->sibling));
}
