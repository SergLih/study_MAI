#include <iostream>
#include <cstdlib>
#include <cstring>

#include "pentagon.h"
#include "hexagon.h"
#include "octagon.h"

int main(int argc, char** argv) {
    std::cout << "Use 'help' or 'h' to get help." << std::endl;
    const int size = 16;
    char s[size];

    Figure * ptr_fig = nullptr;
    while (1) {
        std::cin.getline(s, size);
        std::cin.clear();
        std::cin.sync();
        if (strcmp(s, "create") == 0 || strcmp(s, "cr") == 0) {
            std::cout << "Which figure do you want to create? (pent/hex/oct[agon]): ";
            std::cin.getline(s, size);
            std::cin.clear();
            std::cin.sync();

            if (strcmp(s, "pentagon") == 0 || strcmp(s, "pent") == 0) {
                if (ptr_fig != nullptr) delete ptr_fig;
                ptr_fig = new Pentagon(std::cin);
            } else if (strcmp(s, "hexagon") == 0 || strcmp(s, "hex") == 0) {
                if (ptr_fig != nullptr) delete ptr_fig;
                ptr_fig = new Hexagon(std::cin);
            } else if (strcmp(s, "octagon") == 0 || strcmp(s, "oct") == 0) {
                if (ptr_fig != nullptr) delete ptr_fig;
                ptr_fig = new Octagon(std::cin);
            } else {
                std::cout << "Invalid choice. The figure has not been created! " << std::endl;
            }
        } else if (strcmp(s, "print") == 0 || strcmp(s, "pr") == 0) {
            if(ptr_fig == nullptr) {
                std::cout << "The figure doesn't exist." << std::endl;
            } else {
                ptr_fig->Print();
            }
        } else if (strcmp(s, "square") == 0 || strcmp(s, "sq") == 0) {
            if(ptr_fig == nullptr) {
                std::cout << "The figure doesn't exist." << std::endl;
            } else {
                std::cout << ptr_fig->Square() << std::endl;
            }
        } else if (strcmp(s, "quit") == 0 || strcmp(s, "exit") == 0 || strcmp(s, "q") == 0) {
            if (ptr_fig != nullptr) {
                delete ptr_fig;
            }
            break;
        } else if (strcmp(s, "help") == 0 || strcmp(s, "h") == 0) {
            std::cout << "\n\ncr[eate]             create new figure";
              std::cout << "\npr[int]              print <side> of the figure";
              std::cout << "\nsq[uare]             compute square of the figure";
              std::cout << "\nq[uit] or exit       exit the program\n\n";
          } 
    }

    return 0;
}
