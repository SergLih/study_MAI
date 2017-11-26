#include <iostream>
#include <cstdlib>
#include <cstring>
#include <typeinfo>

#include "pentagon.h"
#include "octagon.h"
#include "hexagon.h"
#include "storage.h"

int main(int argc, char** argv) {
	std::cout << "Use 'help' or 'h' to get help." << std::endl;
	const int size = 16;
	char s[size];


	TStorage<Figure> storage;

	std::shared_ptr<Figure> ptr_fig = nullptr, ptr2 = nullptr;
	while (1) {
		std::cin.getline(s, size);
		std::cin.clear();
		std::cin.sync();
		if (strcmp(s, "quit") == 0 || strcmp(s, "exit") == 0 || strcmp(s, "q") == 0) {
			break;
		}
		else if (strcmp(s, "print") == 0 || strcmp(s, "p") == 0) {
			std::cout << storage << std::endl;
		}
		else if (strcmp(s, "append") == 0 || strcmp(s, "add") == 0) {
			std::cout << "Which figure do you want to append? (pent/hex/oct[agon]): ";
			std::cin.getline(s, size);
			std::cin.clear();
			std::cin.sync();

			if (strcmp(s, "pentagon") == 0 || strcmp(s, "pent") == 0) {
				storage.Insert(std::shared_ptr<Pentagon>(new Pentagon(std::cin)));
			}
			else if (strcmp(s, "hexagon") == 0 || strcmp(s, "hex") == 0) {
				storage.Insert(std::make_shared<Hexagon>(std::cin));
			}
			else if (strcmp(s, "octagon") == 0 || strcmp(s, "oct") == 0) {
				storage.Insert(std::make_shared<Octagon>(std::cin));
			}
			else {
				std::cout << "Invalid choice. The figure has not been created! " << std::endl;
				continue;
			}
		}
		else if (strcmp(s, "delete") == 0 || strcmp(s, "del") == 0) {
			std::cout << "Which criterion do you want to use? (a[rea]/t[ype]): ";
			std::cin.getline(s, size);
			std::cin.clear();
			std::cin.sync();

			if (strcmp(s, "area") == 0 || strcmp(s, "a") == 0) {
				double maxSq;
				std::cout << "Enter the square threshold under which figures will be deleted: ";
				std::cin >> maxSq;
				RemoveCriteriaByMaxSquare critMaxSq(maxSq);
				storage.DeleteByCriteria(critMaxSq);
			}
			else if (strcmp(s, "type") == 0 || strcmp(s, "t") == 0) 
			{
				std::cout << "Which type of figure do you want to delete? (pent/hex/oct[agon]): ";
				std::cin.getline(s, size);
				std::cin.clear();
				std::cin.sync();

				if (strcmp(s, "pentagon") == 0 || strcmp(s, "pent") == 0) {
					RemoveCriteriaByFigureType critFigType("Pentagon");
					storage.DeleteByCriteria(critFigType);
				}
				else if (strcmp(s, "hexagon") == 0 || strcmp(s, "hex") == 0) {
					RemoveCriteriaByFigureType critFigType("Hexagon");
					storage.DeleteByCriteria(critFigType);
				}
				else if (strcmp(s, "octagon") == 0 || strcmp(s, "oct") == 0) {
					RemoveCriteriaByFigureType critFigType("Octagon");
					storage.DeleteByCriteria(critFigType);
				}
				else {
					std::cout << "Invalid choice. The figure has not been created! " << std::endl;
					continue;
				}
			}
		}
		else if (strcmp(s, "help") == 0 || strcmp(s, "h") == 0) {
			std::cout<<"\n\nadd                    insert figure into the storage";
			std::cout << "\np[rint]                print contents of the storage";
			std::cout << "\ndel[ete]               delete figures from the storage based on criteria\n";
		}
	}

	return 0;
}
