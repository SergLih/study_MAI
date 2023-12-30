#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "tree.h"


int main(void)
{
    char s[8];

    Tree tree = NULL;
    int deep;
    int root = 0, ver = 0, parent = 0;

    printf("\nTo get help in use, write 'help' or 'h'.\n\n");
    while (1) {
        scanf("%7s", s);
        if (!strcmp(s, "insert") || !strcmp(s, "ins")) {
            if(!tree) {
                printf("Enter the value of the tree root:\n");
                scanf("%d", &root);
                tree = tree_create(root);
            }
            while (scanf("%d%d", &parent, &ver)) {
                tree_add_node(tree, parent, ver);
            }
        } else if (!strcmp(s, "delete") || !strcmp(s, "del")) {
            if(!tree) printf("The tree does not exist, use the commands 'help' or 'h'.\n");
            else {
                scanf("%d", &ver);
                tree_del_node(tree, ver);
            }
        } else if (!strcmp(s, "quit") || !strcmp(s, "exit") || !strcmp(s, "q")) {
            if (tree) tree_destroy(tree);
            break;
        } else if (!strcmp(s, "run") || !strcmp(s, "r")) {
            if(!tree) printf("The tree does not exist, use the commands 'help' or 'h'.\n");
            else {
                printf("Depth of tree: %d\n", depth_of_tree(tree));
            }
        } else if (!strcmp(s, "print") || !strcmp(s, "p")) {
            if (!tree) printf("The tree does not exist, use the commands 'help' or 'h'.\n");
            else {
                printf("\n\n");
                tree_print(tree);
                printf("\n\n");
            }
        } else if (!strcmp(s, "destroy") || !strcmp(s, "des")) {
            if (!tree) printf("The tree does not exist, use the commands 'help' or 'h'.\n");
            else {
                tree_destroy(tree);
                tree = NULL;
            }
        } else if (!strcmp(s, "help") || !strcmp(s, "h")) {
            printf("\n\nThe 'insert' and 'ins' commands, if the tree is not created - create a tree, if created - add vertices to the tree.\n\n");
            printf("The 'delete num' and 'del num' commands delete the vertex and all its children.\n\n");
            printf("The commands 'print' and 'p' print the tops of the tree.\n\n");
            printf("The 'run' and 'r' commands determine the depth of the tree.\n\n");
            printf("The commands 'quit', 'q' and 'exit' exit the program.\n\n");
            printf("The 'destroy' and 'des' commands remove the entire tree.\n\n");
        } else {
            printf("\n\nSuch a command does not exist, use the commands 'help' or 'h'. \n\n");
        }
    }
    return 0;
}
