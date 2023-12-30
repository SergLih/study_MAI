#ifndef TTRIE_HPP
#define TTRIE_HPP

#include <iostream>
#include <cstdlib>
#include <sstream>
#include <map>
#include <string>
#include <queue>
#include <functional>

typedef std::map<size_t, size_t, std::greater<size_t> > LineStartsMap;

size_t ConvertToLowerAndCountWords(std::string &str);

class TTrie;

class TTrieNode {
private:
	friend class TTrie;

	std::map<std::string, TTrieNode *> children;
	bool isLeaf;
	TTrieNode *failLink;
	TTrieNode *outLink;
	size_t patternId;
	size_t patternLength;
	TTrieNode *parent;

	size_t id;
	static size_t nextNodeId;
public:
	TTrieNode(TTrieNode * parentNode = nullptr);
	virtual ~TTrieNode();
};

class TTrie {
private:
	friend class TTrieNode;

	TTrieNode *root;
	static size_t nextPatternId;

public:
	TTrie();
	void  Search(const std::string & text, const LineStartsMap &lineStarts);
	void  BuildFailAndOutLinks();
	void  Insert(std::string &str);
    void  DeleteNodeRec(TTrieNode **node);
	virtual ~TTrie();
};

#endif // TTRIE_HPP
