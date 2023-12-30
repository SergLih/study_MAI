#include "ttrie.h"

int main() {
	TTrie tree;
	std::string input, text = "";

	std::getline(std::cin, input);
	while (input != "") {
		ConvertToLowerAndCountWords(input);
		tree.Insert(input);
		std::getline(std::cin, input);
	}

	tree.BuildFailAndOutLinks();

	LineStartsMap lineStarts;
	size_t lineNumber = 1;
	size_t wordNumber = 1;
	size_t wordsCountInLine;
	while (std::getline(std::cin, input)) {
		wordsCountInLine = ConvertToLowerAndCountWords(input);
		if (wordsCountInLine > 0) {
			text += input + " ";
			lineStarts[wordNumber] = lineNumber;
			wordNumber += wordsCountInLine;
		}
		lineNumber++;
	}
	
	tree.Search(text, lineStarts);

	return 0;
}

