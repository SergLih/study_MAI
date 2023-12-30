#include "ttrie.h"

size_t ConvertToLowerAndCountWords(std::string &str) {
	size_t wordCount = 0;
	bool insideWord = false;
	for (size_t i = 0; i < str.length(); i++) {
		str[i] = tolower(str[i]);
		if (isalpha(str[i])) {
			if (!insideWord) {
				insideWord = true;
				wordCount++;
			}
		} else {
			if (insideWord)
				insideWord = false;
		}
	}
	return wordCount;
}

TTrie::TTrie() {
	root = new TTrieNode();
}

void TTrie::Insert(std::string &str) {
	std::stringstream line(str);
	std::string word;

	TTrieNode * curNode = root;
	size_t curLength = 0;
	while (line >> word) {
		curLength++;
		std::map<std::string, TTrieNode *>::iterator it_res
			= curNode->children.find(word);

		//слово нашлось среди ребер -- проходим по нему
		if (it_res != curNode->children.end()) {
			curNode = it_res->second;
		}
		else { //не нашлось: нужно делать ответвление	
			curNode->isLeaf = false;
			TTrieNode * new_node = new TTrieNode(curNode);
			curNode->children[word] = new_node;
			curNode = new_node;
		}
	}
	curNode->patternId = nextPatternId++;
	curNode->patternLength = curLength;
}

void TTrie::Search(const std::string & text, const LineStartsMap &lineStarts) {
	//псевдокод на стр. 60 в англ. Гасфилде
	std::istringstream iss(text);
	std::string word;

	size_t wordNumber = 0;
	TTrieNode * w = root;
	
	bool getNextWord = true;
	do {
		if (getNextWord) {
			if (!(iss >> word))
				break;
			else {
				wordNumber++;
			}
		}

		std::map<std::string, TTrieNode *>::iterator it_res;
		if ((it_res = w->children.find(word)) != w->children.end()) {
			TTrieNode * wprime = it_res->second;
			if (wprime->patternId != 0) {
				size_t totalWordNumber = wordNumber - wprime->patternLength + 1;
				LineStartsMap::const_iterator it = lineStarts.lower_bound(totalWordNumber);
				std::cout << it->second << ", " << 1 + totalWordNumber - it->first << ", " << wprime->patternId << std::endl;
			}

			TTrieNode * cur = wprime;
			while (cur->outLink != nullptr) {
				cur = cur->outLink;
				size_t totalWordNumber = wordNumber - cur->patternLength + 1;
				LineStartsMap::const_iterator it = lineStarts.lower_bound(totalWordNumber);
				std::cout << it->second << ", " << 1 + totalWordNumber - it->first << ", " << cur->patternId << std::endl;
			}
			w = wprime;
			getNextWord = true;
		}
		else {
			if (w == root)
				getNextWord = true;
			else {
				w = w->failLink;
				getNextWord = false;
			}
		}
	} while (true);
}

void TTrie::BuildFailAndOutLinks() {
	std::map<std::string, TTrieNode *>::iterator it;
	//по умолчанию всем вершинам ставим связи ошибок в корень
	for (it = root->children.begin(); it != root->children.end(); ++it) {
		it->second->failLink = root;
	}
	root->failLink = root;
	std::queue<TTrieNode *> q;
	q.push(root);
	while (!q.empty()) {
		TTrieNode *v = q.front();	//обозначения как стр. 86 в Гасфилде
		q.pop();

		if (v != root && v->parent != root) {
			TTrieNode * vprime = v->parent;
			std::string x;
			for (it = vprime->children.begin();
				it != vprime->children.end(); ++it) {
				if (it->second == v) {
					x = it->first;
					break;
				}
			}

			TTrieNode * w = vprime->failLink;
			//цикл до тех пор пока не найдем удачное ребро или не окажемся в корне
			while (w->children.find(x) == w->children.end() && w != root) {
				w = w->failLink;
			}

			if ((it = w->children.find(x)) != w->children.end()) {
				TTrieNode * wprime = it->second;
				v->failLink = wprime;
				if (wprime->patternId != 0) {
					v->outLink = wprime;
				}
				else if (wprime->outLink != nullptr) {
					v->outLink = wprime->outLink;
				}
			}
			else {
				v->failLink = root;
			}
		}

		if (!v->isLeaf) {
			for (it = v->children.begin(); it != v->children.end(); ++it) {
				q.push(it->second);
			}
		}
	}
}

void TTrie::DeleteNodeRec(TTrieNode ** node) {
	std::map<std::string, TTrieNode *>::iterator it;
	for (it = (*node)->children.begin(); it != (*node)->children.end(); ++it) {
		DeleteNodeRec(&(it->second));
	}
	delete *node;
	*node = nullptr;
}

TTrie::~TTrie() {
	DeleteNodeRec(&root);
}

size_t TTrieNode::nextNodeId = 0;
size_t TTrie::nextPatternId = 1;

TTrieNode::TTrieNode(TTrieNode * parentNode) {
	isLeaf = true;
	failLink = nullptr;
	outLink = nullptr;
	id = nextNodeId++;
	this->parent = parentNode;
	patternId = 0;
	patternLength = 0;
}

TTrieNode::~TTrieNode() {}
