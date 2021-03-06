#include <iostream>
#include <cstdlib>
#include <cstring>

#include "pentagon.h"
#include "octagon.h"
#include "hexagon.h"
#include "tbinary_tree.h"

int main(int argc, char** argv) {

    std::cout << "Use 'help' or 'h' to get help." << std::endl;
    const int size = 16;
    char s[size];


	TBinaryTree<Figure> tree;
	std::shared_ptr<Figure> ptr_fig = nullptr, ptr2 = nullptr;
	while (1) {
		std::cin.getline(s, size);
		std::cin.clear();
		std::cin.sync();
		if (strcmp(s, "quit") == 0 || strcmp(s, "exit") == 0 || strcmp(s, "q") == 0) {
			break;
		}
		else if (strcmp(s, "print") == 0 || strcmp(s, "p") == 0) {
			std::cout << tree << std::endl;
		}
		else if (strcmp(s, "iterate") == 0 || strcmp(s, "it") == 0) {
			for (TBinaryTreeIterator<Figure> it = tree.begin(); it != tree.end(); it++) {
				it->print(std::cout);
				std::cout << std::endl;
			}
				
		}
		else if (strcmp(s, "append") == 0 || strcmp(s, "add") == 0) {
			std::cout << "Which figure do you want to append? (pent/hex/oct[agon]): ";
			std::cin.getline(s, size);
			std::cin.clear();
			std::cin.sync();

			if (strcmp(s, "pentagon") == 0 || strcmp(s, "pent") == 0) {
				tree.Insert(std::shared_ptr<Pentagon>(new Pentagon(std::cin)));
			}
			else if (strcmp(s, "hexagon") == 0 || strcmp(s, "hex") == 0) {
				tree.Insert(std::make_shared<Hexagon>(std::cin));
			}
			else if (strcmp(s, "octagon") == 0 || strcmp(s, "oct") == 0) {
				tree.Insert(std::make_shared<Octagon>(std::cin));
			}
			else {
				std::cout << "Invalid choice. The figure has not been created! " << std::endl;
				continue;
			}
		}
		else if (strcmp(s, "delete") == 0 || strcmp(s, "del") == 0) {
			std::cout << "Which figure do you want to delete? (pent/hex/oct[agon]): ";
			std::cin.getline(s, size);
			std::cin.clear();
			std::cin.sync();

			if (strcmp(s, "pentagon") == 0 || strcmp(s, "pent") == 0) {
				ptr_fig = std::shared_ptr<Pentagon>(new Pentagon(std::cin));
			}
			else if (strcmp(s, "hexagon") == 0 || strcmp(s, "hex") == 0) {
				ptr_fig = std::shared_ptr<Hexagon>(new Hexagon(std::cin));
			}
			else if (strcmp(s, "octagon") == 0 || strcmp(s, "oct") == 0) {
				ptr_fig = std::shared_ptr<Octagon>(new Octagon(std::cin));
			}
			else {
				std::cout << "Invalid choice. The figure has not been created! " << std::endl;
				continue;
			}
			tree.Delete(ptr_fig);
		}
		else if (strcmp(s, "find") == 0 || strcmp(s, "f") == 0) {
			std::cout << "Which figure do you want to find? (pent/hex/oct[agon]): ";
			std::cin.getline(s, size);
			std::cin.clear();
			std::cin.sync();

			if (strcmp(s, "pentagon") == 0 || strcmp(s, "pent") == 0) {
				ptr_fig = std::shared_ptr<Pentagon>(new Pentagon(std::cin));
			}
			else if (strcmp(s, "hexagon") == 0 || strcmp(s, "hex") == 0) {
				ptr_fig = std::shared_ptr<Hexagon>(new Hexagon(std::cin));
			}
			else if (strcmp(s, "octagon") == 0 || strcmp(s, "oct") == 0) {
				ptr_fig = std::shared_ptr<Octagon>(new Octagon(std::cin));
			}
			else {
				std::cout << "Invalid choice. The figure has not been created! " << std::endl;
				continue;
			}
			if (tree.Find(ptr_fig) != nullptr)
				std::cout << "The figure was FOUND in the binary tree\n";
			else
				std::cout << "The figure was NOT FOUND in the binary tree\n";
		}
		else if (strcmp(s, "help") == 0 || strcmp(s, "h") == 0) {
			std::cout << "\n\nappend or add        insert figure in binary tree";
			std::cout << "\nf[ind]				         find figure in binary tree";
			std::cout << "\np[rint]                print binary tree";
			std::cout << "\nit[erate]              print iterator of binary tree";
			std::cout << "\ndel[ete]               delete in a binary tree a figure with the size <side>";
			std::cout << "\nq[uit]                 exit the program\n\n";
		}
	}


    return 0;
}
