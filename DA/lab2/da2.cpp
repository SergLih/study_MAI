#include "rbtree.h" 

const int TYPE_SIZE = 4;
const int NAME_SIZE = 20;

int main(int argv, char **argc) {
	char type[TYPE_SIZE] = { '\0' };
	char name[NAME_SIZE] = { '\0' };
	TKey temp_str;
	TVal number = 0;
	TRBTree *dict = new TRBTree();
	if (dict == nullptr) {
		std::cout << "ERROR: Couldn't create dictionary\n";
		exit(EXIT_SUCCESS);
	}
	char action;
	while (std::cin >> action) {
		if (action == '+') {
			std::cin >> temp_str;
			std::cin >> number;
			if (dict->Insert(temp_str, number))
				std::cout << "OK\n";
			else
				std::cout << "Exist\n";
		}
		else if (action == '-') {
			std::cin >> temp_str;
			if (dict->Delete(temp_str))
				std::cout << "OK\n";
			else
				std::cout << "NoSuchWord\n";
		}
		else if (action == '!') {
			std::cin >> type;
			if (type[0] == 'S') {
				std::cin >> name;
				if (dict->Save(name))
					std::cout << "OK\n";
				else
					std::cout << "ERROR: Couldn't create file\n";
			}
			else if (type[0] == 'L') {
				std::cin >> name;
				if (dict->Load(name))
					std::cout << "OK\n";
				else
					std::cout << "ERROR: Cannot read file\n";
			}
		}
		else {
			ungetc(action, stdin);
			std::cin >> temp_str;
			if (dict->Search(temp_str, number))
				std::cout << "OK: " << number << "\n";
			else
				std::cout << "NoSuchWord\n";
		}
	}
	delete dict;
	return 0;
}
