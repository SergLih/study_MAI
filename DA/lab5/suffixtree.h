#ifndef SUFFIXTREE_H
#define SUFFIXTREE_H

#define ALPHABET_SIZE 27

//#define DEBUG

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>


class TSuffixArray;

class TNode
{
public:

	TNode** edges;
	int start, end;
	TNode *suffixLink;
	TNode(int start, int end, bool isLeaf = true);
	~TNode();
	int EgdeLength(size_t curLength);

	int id;
	static int nextId;
};

class TSuffixTree
{
public:
	TSuffixTree(std::string str);
	
	void TreePrint(int curLength = -1);
	void PrintDebugInfo(int pos);
	~TSuffixTree();
	friend TSuffixArray;

private:
	std::string s;
	TNode *root;
	int suffixesToInsertCount;
	TNode *suffixLinkStart, *actNode;
	int actLength;
	int actEdge = 0;
	void NodePrint(TNode *node, int dpth, int curLength);
	void AddSuffixLink(TNode *node);
	void TreeExtend(int pos);
	void TreeDestroy(TNode *node);
	void CheckActiveEdgeLength(int pos);
	void FollowSuffixLink();
	int c(char c);
	void DFS(TNode *node, std::vector<int> &result, int deep);
	void ToSuffixArrayRec(TNode * node, int depth, std::vector<int>& result);
};

class TSuffixArray
{
public:
	TSuffixArray(TSuffixTree &tree);
	std::vector<int> Find(std::string pattern);
	~TSuffixArray() {};
	void Print();
private:
	std::string s;
	std::string pattern;
	std::vector<int> arr;

	int LowerBound(int left, int right, int i);
	int UpperBound(int left, int right, int i);
	int ComparePatternWithSuffix(int i_char, int i_sfx);
};


#endif
