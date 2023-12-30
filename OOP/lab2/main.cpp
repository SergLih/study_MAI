#include <iostream>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "pentagon.h"
#include "tbinary_tree.h"


int main(int argc, char** argv) {
    std::cout << "Use 'help' or 'h' to get help." << std::endl;
    const int size = 16;
    char s[size];
  
    Pentagon pentagon;
    TBinaryTree tree;
    int side;
    while (1) {
        std::cin.getline(s, size);
        std::cin.clear();
        std::cin.sync();
        if (strcmp(s, "quit") == 0 || strcmp(s, "exit") == 0 || strcmp(s, "q") == 0) {
            break;
        } else if (strcmp(s, "print") == 0 || strcmp(s, "p") == 0) {
            std::cout << tree << std::endl;
        } else if (strcmp(s, "insert") == 0 || strcmp(s, "ins") == 0) {	        
                std::cin >> pentagon;
                tree.Insert(pentagon);
        } else if (strcmp(s, "delete") == 0 || strcmp(s, "del") == 0) {
            std::cout << "Input side of pentagon to delete: ";
            std::cin >> side;
            tree.Delete(side);
        } else if (strcmp(s, "find") == 0 || strcmp(s, "f") == 0) {
            std::cout << "Input side of pentagon to search: ";
            std::cin >> side;
            TItem * data = tree.Find(side);
            if (data == nullptr) {
                std::cout << "Not found...\n";
            } else {
                std::cout << "Found: " << *data << "\n";
            }
        } else if (strcmp(s, "help") == 0 || strcmp(s, "h") == 0) {
            std::cout << "\n\nins[ert]             insert <pentagon> in binary tree";
            std::cout << "\np[rint]                print binary tree";
            std::cout << "\nf[ind]                 search in the binary tree of a pentagon with the <side>";
            std::cout << "\ndel[ete]               delete in a binary tree a pentagon with the size <side>";
            std::cout << "\nq[uit]                 exit the program\n\n";
        }
    }
    return 0;
}
