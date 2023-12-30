#include "suffixtree.h"
#include <string>
#include <iostream>
//#define DEBUG1

std::vector<int> FindBrute(std::string s, std::string pattern) {
	std::vector<int> result;
	size_t pos = s.find(pattern, 0);
	while (pos != std::string::npos) {
		result.push_back(pos);
		pos = s.find(pattern, pos + 1);
	}
	return result;
}

void PrintResult(std::vector<int> result, int cntPattern) {
	if (!result.empty()) {
		std::cout << cntPattern << ": ";
		for (int i = 0; i < result.size(); ++i) {
			std::cout << result[i] + 1;
			if (i < result.size() - 1) {
				std::cout << ", ";
			}
		}
		std::cout << '\n';
	}
}

int main(void) {
	std::string s, pattern;
	std::cin >> s;
	TSuffixTree tree(s + "{");

#ifdef DEBUG1
	tree.TreePrint();
#endif // DEBUG1

	TSuffixArray arr(tree);
#ifdef DEBUG1
	arr.Print();
#endif // DEBUG1


	for (int i = 1; std::cin >> pattern; i++) {
#ifdef DEBUG1
		std::cout << "=================\n";
		std::vector<int> result_brute = FindBrute(s, pattern);
		PrintResult(result_brute, i);
#endif // DEBUG1

		std::vector<int> result = arr.Find(pattern);
		PrintResult(result, i);
	}

	return 0;
}
