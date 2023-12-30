#include "suffixtree.h"

TSuffixTree::TSuffixTree(std::string text) {
	this->s = text;
	root = new TNode(0, 0, false);
	
	suffixesToInsertCount = 0;
	actLength = 0;
	actNode = suffixLinkStart =  root;
	
	for (int i = 0; i < text.length(); i++) {
		TreeExtend(i);
	}
}

void TSuffixTree::NodePrint(TNode *node, int dpth, int curLength) {
	if (node != root) {
		for (int i = 0; i < dpth - 1; ++i) {
			std::cout << "    ";
		}
		if (dpth > 0) {
			std::cout << "|---";
		}


		int len = node->end == -1 ? node->EgdeLength(curLength) + 1 : node->EgdeLength(curLength);
		for (int i = node->start; i < node->start + len; i++) {
			std::cout << s[i];
		}
		std::cout << " " << node->id;
		if (node->suffixLink)
			std::cout << " --> " << node->suffixLink->id;
		std::cout << std::endl;
	}

    if(node->edges) {
	    for (int i = 0; i < ALPHABET_SIZE; i++) {
		    if (node->edges[i]) {
			    NodePrint(node->edges[i], dpth + 1, curLength);
		    }
	    }
	}
}

void TSuffixTree::TreePrint(int curLength) {
	if (curLength == -1)
		curLength = s.size();
	NodePrint(root, 0, curLength);
}

void TSuffixTree::PrintDebugInfo(int pos) {
	std::cout << "Pos: " << pos << "| Rem: " << suffixesToInsertCount << " | " << "Active: id " <<
		actNode->id << " edge#: " << actEdge << " len: " << actLength <<
		"\nedge: " << s[actEdge] << "\n";
	std::cout << "current sfx lnk st:" << (suffixLinkStart ? suffixLinkStart->id : -1) << "\n";
	TreePrint(pos);
	std::cout << std::endl;
}

void TSuffixTree::TreeExtend(int pos) {
	suffixLinkStart = nullptr;
	suffixesToInsertCount++;
	TNode * nextNode = nullptr;

#ifdef DEBUG
	std::cout << std::string(80, '=') << "\n";
	PrintDebugInfo(pos);
#endif // DEBUG

	do {

		if (actLength == 0)
			actEdge = pos;
		nextNode = actNode->edges[c(s[actEdge])];

		if (nextNode == nullptr) {								//СЛУЧАЙ 1: ВСТАВКА НОВОГО ЛИСТА
			actNode->edges[c(s[actEdge])] = new TNode(pos, -1);			//вставляем новое ребро до листа, активную точку не меняем
			suffixesToInsertCount--;										//Вставили один суффикс (ответвлением от вершины)
			AddSuffixLink(actNode);											//Правило 2: суфф. стр. из акт. вершины будет вставлена на след. шагах
		} else if (s[nextNode->start + actLength] == s[pos]) {	//СЛУЧАЙ 2: ПРОДЛЕНИЕ ПО РЕБРУ
			actLength++;													//* Проходим на один символ по ребру
			AddSuffixLink(actNode);										//Правило 2 при продлении: суфф. стр. из акт. вершины будет вставлена на след. шагах 
			CheckActiveEdgeLength(pos);	//Проверяем, не попали ли в новую вершину
			break;															//Накопили один "сейчас не вставленный" суффикс ("лишний" ++ перед циклом!)
		} else {
			
			//СЛУЧАЙ 3: РАЗБИЕНИЕ РЕБРА И ВЕТВЛЕНИЕ
			TNode * splitNode = new TNode(nextNode->start, nextNode->start + actLength, false);	//вершина ветвления
			TNode * leafNode = new TNode(pos, -1);
			nextNode->start += actLength;	
			splitNode->edges[c(s[nextNode->start])] = nextNode;				//присоединяем два листа (новый и старый)    
			splitNode->edges[c(s[pos])] = leafNode;							//к вершине ветвления
			actNode->edges[c(s[actEdge])] = splitNode;					//ребро, ведущее из акт.точки, теперь входит в верш. ветвл.
			suffixesToInsertCount--;										//Вставили один суффикс (ответвлением "от ребра" после разбиения)
			AddSuffixLink(splitNode);										//Правило 2: суфф. стр. из вершины ветвл. будет вставлена на след. шагах 
		}

		//Правила 1 и 3, применяемые после вставки новой вершины ветвления или листа
		if (actNode == root && actLength) {									//Правило 1
			actLength--;													// * 
			actEdge = pos - suffixesToInsertCount + 1;						// *
		} else {															//Правило 3
			FollowSuffixLink();												// *
		}
		CheckActiveEdgeLength(pos);

#ifdef DEBUG
		PrintDebugInfo(pos);
#endif // DEBUG

	} while (suffixesToInsertCount > 0);
}

void TSuffixTree::CheckActiveEdgeLength(int pos) {	//функция для возможного перехода в новую активную вершину после сдвига активной точки / длины

	TNode* nextNode = actNode->edges[c(s[actEdge])];
	if (nextNode == nullptr || nextNode->end == -1) //продление ребер входящих в лист происходит автоматически
		return;
		
	int edgeLength;						//Если ребро кончилось, перешли в новый узел
	while (nextNode && (actLength >= (edgeLength = nextNode->EgdeLength(pos)))) {
		actLength -= edgeLength;		
		actNode = nextNode;
		if (actLength == 0) {
			break;
		} else {
			actEdge += edgeLength;
			nextNode = actNode->edges[c(s[actEdge])];
#ifdef DEBUG
			PrintDebugInfo(pos);
#endif // DEBUG
		}
	}
}

void TSuffixTree::FollowSuffixLink() {
	if (actNode->suffixLink != nullptr)
		actNode = actNode->suffixLink;
	else
		actNode = root;
}

int TSuffixTree::c(char letter) {
	return (letter - 'a');
}

void TSuffixTree::AddSuffixLink(TNode *node) {
	if (suffixLinkStart != nullptr && node != root && node != suffixLinkStart) {
		suffixLinkStart->suffixLink = node;
#ifdef DEBUG
		std::cout << "\tsfx link was added: " << suffixLinkStart->id << " --> " << node->id << "\n";
#endif // DEBUG
	}
	suffixLinkStart = node;
}

void TSuffixTree::DFS(TNode *node, std::vector<int> &result, int deep) {
	
	if (node->edges != nullptr) {
		result.push_back(s.length() - deep);
		return;
	}

	for (int i = 0; i < ALPHABET_SIZE; i++) {
		if (node->edges[i]) {
			int tmp = deep;
			tmp += node->edges[i]->end - node->edges[i]->start;
			DFS(node->edges[i], result, tmp);
		}
	}

}

TSuffixArray::TSuffixArray(TSuffixTree &tree) {
	tree.ToSuffixArrayRec(tree.root, 0, arr);
	s = tree.s;
}

void TSuffixTree::ToSuffixArrayRec(TNode *node, int depth, std::vector<int> &arr) {
	if (node->end != -1) {
		for (int i = 0; i < ALPHABET_SIZE; i++) {
			TNode * nextNode = node->edges[i];
			if (nextNode) {
				ToSuffixArrayRec(node->edges[i], depth + nextNode->EgdeLength(s.size()), arr);	//увеличивая текущую длину суффикса
			}
		}
	} else {											//Если узел является листом, значит мы дошли до конца суффикса -- надо положить его в массив.
		arr.push_back(s.size() - depth);
	}
}

std::vector<int> TSuffixArray::Find(std::string pattern) {
	this->pattern = pattern;
	std::vector<int> result;

	int left = 0;
	int right = arr.size();
#ifdef DEBUG
	std::cout << "BS: " << left << " " << right << std::endl;
#endif //DEBUG
	for (size_t i = 0; i < pattern.size(); i++) {
		left = LowerBound(left, right, i);		//первый суффикс, который лексиграфически <= pattern по i-й букве
		right = UpperBound(left, right, i);		//первый суффикс, который лексиграфически > pattern
#ifdef DEBUG
		std::cout << "BS: " << left << " " << right << "\n\t"
			      << s.substr(arr[left]) << "\n\t" << s.substr(arr[right-1]) << std::endl;
#endif //DEBUG
	}

	if (right - left > 0) {
		for (size_t j = left; j < right; j++) {
			result.push_back(arr[j]);
		}
	}

	std::sort(result.begin(), result.end());
	return result;
}

void TSuffixArray::Print() {
	std::cout << "\n";
	for (size_t i = 0; i < arr.size(); i++) {
		std::cout << arr[i] << "\t" << s.substr(arr[i]) << "\n";
	}
	std::cout << "\n";
}

int TSuffixArray::ComparePatternWithSuffix(int i_char, int i_sfx) {	//номер буквы образца, ноиер суффикса в суффиксном массиве (отсортированном)
	int len_sfx = s.size() - arr[i_sfx];
	if (len_sfx < i_char) {
		return 1;
	} else {
		#ifdef DEBUG
			std::cout << "\t\tCMP: " << pattern[i_char] << s[arr[i_sfx] + i_char] << " " << std::endl;
		#endif // DEBUG

		
		return s[arr[i_sfx] + i_char] - pattern[i_char];
	}
}


int TSuffixArray::LowerBound(int left, int right, int i) {
#ifdef DEBUG
	std::cout << "\tLB: " << left << " " << right << std::endl;
#endif // DEBUG

	while (right - left > 0) {
		int mid = left + (right - left) / 2;
		int cmp = ComparePatternWithSuffix(i, mid);
		if (cmp < 0) {
			left = mid + 1;
		} else {
			right = mid;
		}

#ifdef DEBUG
	std::cout << "\tLB: " << left << " " << right << std::endl;
#endif // DEBUG
	}
	return left;
}


int TSuffixArray::UpperBound(int left, int right, int i) {
#ifdef DEBUG
	std::cout << "\tUB: " << left << " " << right << std::endl;
#endif //DEBUG

	while (right - left > 0) {
		int mid = left + (right - left) / 2;
		int cmp = ComparePatternWithSuffix(i, mid);
		if (cmp <= 0) {
			left = mid + 1;
		} else {
			right = mid;
		}
#ifdef DEBUG
	std::cout << "\tUB: " << left << " " << right << std::endl;
#endif //DEBUG
	}
	return left;
}

int TNode::nextId = 0;

TNode::TNode(int start, int end, bool isLeaf) {
	this->id = TNode::nextId++;
	this->start = start;
	this->end = end;
	this->suffixLink = nullptr;

	if (!isLeaf) {
		this->edges = new TNode*[ALPHABET_SIZE];
		for (int i = 0; i < ALPHABET_SIZE; i++) {
			this->edges[i] = nullptr;
		}
	} else {
		this->edges = nullptr;
	}
}

void TSuffixTree::TreeDestroy(TNode *node) {
	if (node->edges) {
		for (int i = 0; i < ALPHABET_SIZE; i++) {
			if (node->edges[i])
				TreeDestroy(node->edges[i]);
		}
	}
	delete node;
	node = nullptr;
}

TNode::~TNode() { 
	if (edges) {
		delete[] edges;
		edges = nullptr;
	}
}

TSuffixTree::~TSuffixTree() {
    TreeDestroy(root);
}

int TNode::EgdeLength(size_t curLength) {
	return (end == -1) ? curLength - start : end - start;
}
