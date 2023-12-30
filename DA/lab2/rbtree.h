#ifndef RBTREE_H
#define RBTREE_H

#include "main.h"
#include "mystring.h"

constexpr char const* FILE_SIGNATURE = "da2SergioRBTree_5f31c7cbccef4969bdace27bd7e82dd6";
const int FILE_SIGNATURE_LEN = 49;

typedef TString TKey;
typedef unsigned long long TVal;

int CompareKey(const TKey &k1, const TKey &k2);

void Serialize(const TVal &obj, std::ofstream & file);
bool Deserialize(TVal &obj, std::ifstream & file);

enum TColor : unsigned char { BLACK = 0, RED = 1 };
enum TNodeState : unsigned char {
	NO_CHILDREN = 0,
	NO_CHILDREN_RED = 1,
	HAS_RIGHT = 2,
	HAS_RIGHT_RED = 3,
	HAS_LEFT = 4,
	HAS_LEFT_RED = 5,
	HAS_BOTH = 6,
	HAS_BOTH_RED = 7,
	EMPTY_TREE = 8
};

class TRBTree {
private:
	struct TNode {
		TColor color;
		TNode *left;
		TNode *right;
		TNode *parent;
		TVal val;
		TKey key;
	};
	TNode *root;
	TNode nil;

	TNode* CreateNode(TNode *parent, const TKey &key, const TVal &number, TColor color = RED);
	TNode* Minimum(TNode *node);

	TNode * SearchNode(const TKey &key) const;
	void DestroyR(TNode* node);

	void RotateLeft(TNode *node);
	void RotateRight(TNode *node);

	void InsertFix(TNode *node);
	void DeleteFix(TNode *node);

	void Transplant(TNode *unode, TNode *vnode);

	void Serialize(TNode *node, std::ofstream &file) const;
	TNode* Deserialize(TNode *parent, std::ifstream &file);

public:
	TRBTree();
	bool Insert(const TKey &key, const TVal &number);
	bool Search(const TKey &key, TVal &result_val) const;
	bool Delete(const TKey &key);
	bool Save(const char *path);
	bool Load(const char *path);
	virtual ~TRBTree();
};

#endif
