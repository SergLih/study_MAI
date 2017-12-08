#include <iostream>
#include <cstdlib>
#include <cstring>

#include "pentagon.h"
#include "octagon.h"
#include "hexagon.h"
#include "tlist.h"


int main(int argc, char** argv) {
    std::cout << "Use 'help' or 'h' to get help." << std::endl;
    const int size = 16;
    char s[size];


	TList<Figure> list;
	std::shared_ptr<Figure> ptr_fig = nullptr, ptr2 = nullptr;
	while (1) {
		std::cin.getline(s, size);
		std::cin.clear();
		std::cin.sync();
		if (strcmp(s, "quit") == 0 || strcmp(s, "exit") == 0 || strcmp(s, "q") == 0) {
			break;
		}
		else if (strcmp(s, "print") == 0 || strcmp(s, "p") == 0) {
			std::cout << list << std::endl;
		}
		else if (strcmp(s, "append") == 0 || strcmp(s, "add") == 0) {
			std::cout << "Which figure do you want to append? (pent/hex/oct[agon]): ";
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
			list.Push(ptr_fig);
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
			list.Delete(ptr_fig);
		}
		else if (strcmp(s, "sort") == 0 || strcmp(s, "s") == 0) {
			list.Sort();
		}
		else if (strcmp(s, "psort") == 0 || strcmp(s, "ps") == 0) {
			list.SortParallel();
		}
		else if (strcmp(s, "help") == 0 || strcmp(s, "h") == 0) {
			std::cout << "\n\nappend or add        insert figure in list";
			std::cout << "\np[rint]                print list";
			std::cout << "\ndel[ete]               delete figure with the size <side> from the list";
			std::cout << "\ns[ort]                 sort list";
			std::cout << "\nps[ort]                sort list in parallel";
			std::cout << "\nq[uit]                 exit the program\n\n";
		}
	}


	//system("pause");
    return 0;
}
