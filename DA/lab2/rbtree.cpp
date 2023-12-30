#include "rbtree.h"

off_t filesize(const char *filename) {
	struct stat st;

	if (stat(filename, &st) == 0)
		return st.st_size;

	return -1;
}

TRBTree::TNode *TRBTree::CreateNode(TNode *parent, const TKey &key, const TVal &number, TColor color) {
	TNode *node = new TNode;
	if (node == nullptr) {
		std::cout << "ERROR: couldn't malloc a lot of memory for new node\n";
		exit(EXIT_SUCCESS);
	}
	node->val = number;
	node->key = key;
	node->color = color;
	node->left = &nil;
	node->right = &nil;
	node->parent = parent;
	return node;
}

TRBTree::TNode *TRBTree::Minimum(TNode *node) {
	while (node->left != &nil) {
		node = node->left;
	}
	return node;
}

void TRBTree::DestroyR(TNode *node) {
	if (node->left != &nil) {
		DestroyR(node->left);
	}
	if (node->right != &nil) {
		DestroyR(node->right);
	}
	delete node;
	node = &nil;
}

void TRBTree::RotateLeft(TNode *node) {
	TNode *tmp = node->right;
	node->right = tmp->left;
	if (tmp->left != &nil) {
		tmp->left->parent = node;
	}
	tmp->parent = node->parent;
	if (node->parent == &nil) {
		this->root = tmp;
	}
	else if (node == node->parent->left) {
		node->parent->left = tmp;
	}
	else {
		node->parent->right = tmp;
	}
	tmp->left = node;
	node->parent = tmp;
}

void TRBTree::RotateRight(TNode *node) {
	TNode *tmp = node->left;
	node->left = tmp->right;
	if (tmp->right != &nil) {
		tmp->right->parent = node;
	}
	tmp->parent = node->parent;
	if (node->parent == &nil) {
		this->root = tmp;
	}
	else if (node == node->parent->left) {
		node->parent->left = tmp;
	}
	else {
		node->parent->right = tmp;
	}
	tmp->right = node;
	node->parent = tmp;
}

void TRBTree::InsertFix(TNode *node) {
	while (node != this->root && node->parent->color == RED) {
		if (node->parent == node->parent->parent->left) {
			TNode *tmp = node->parent->parent->right;
			if (tmp->color == RED) {
				node->parent->color = BLACK;
				tmp->color = BLACK;
				node->parent->parent->color = RED;
				node = node->parent->parent;
			}
			else {
				if (node == node->parent->right) {
					node = node->parent;
					RotateLeft(node);
				}
				node->parent->color = BLACK;
				node->parent->parent->color = RED;
				RotateRight(node->parent->parent);
			}
		}
		else {
			TNode *tmp = node->parent->parent->left;
			if (tmp->color == RED) {
				node->parent->color = BLACK;
				tmp->color = BLACK;
				node->parent->parent->color = RED;
				node = node->parent->parent;
			}
			else {
				if (node == node->parent->left) {
					node = node->parent;
					RotateRight(node);
				}
				node->parent->color = BLACK;
				node->parent->parent->color = RED;
				RotateLeft(node->parent->parent);
			}
		}
	}
	this->root->color = BLACK;
}

void TRBTree::DeleteFix(TNode *node) {
	while (node != this->root && node->color == BLACK) {
		if (node == node->parent->left) {
			TNode *temp = node->parent->right;
			if (temp->color == RED) {
				temp->color = BLACK;
				node->parent->color = RED;
				RotateLeft(node->parent);
				temp = node->parent->right;
			}
			if (temp->left->color == BLACK && temp->right->color == BLACK) {
				temp->color = RED;
				node = node->parent;
			}
			else {
				if (temp->right->color == BLACK) {
					temp->left->color = BLACK;
					temp->color = RED;
					RotateRight(temp);
					temp = node->parent->right;
				}
				temp->color = node->parent->color;
				temp->right->color = node->parent->color = BLACK;
				RotateLeft(node->parent);
				node = this->root;
			}
		}
		else {
			TNode *temp = node->parent->left;
			if (temp->color == RED) {
				temp->color = BLACK;
				node->parent->color = RED;
				RotateRight(node->parent);
				temp = node->parent->left;
			}
			if (temp->right->color == BLACK && temp->left->color == BLACK) {
				temp->color = RED;
				node = node->parent;
			}
			else {
				if (temp->left->color == BLACK) {
					temp->right->color = BLACK;
					temp->color = RED;
					RotateLeft(temp);
					temp = node->parent->left;
				}
				temp->color = node->parent->color;
				temp->left->color = node->parent->color = BLACK;
				RotateRight(node->parent);
				node = this->root;
			}
		}
	}
	node->color = BLACK;
}

void TRBTree::Transplant(TNode * unode, TNode * vnode) {
	if (unode->parent == &nil)
		root = vnode;
	else if (unode == unode->parent->left)
		unode->parent->left = vnode;
	else
		unode->parent->right = vnode;
	vnode->parent = unode->parent;
}

void Serialize(const TVal & obj, std::ofstream & file) {
	file.write((char*)&obj, sizeof(TVal));
}

bool Deserialize(TVal & obj, std::ifstream & file) {
	if (file.read((char *)&obj, sizeof(TVal)))
		return true;
	return false;
}

void TRBTree::Serialize(TNode *node, std::ofstream &file) const {
	if (node == &nil) {
		TNodeState state = EMPTY_TREE;
		file.write((char *)&state, sizeof(TNodeState));
		return;
	}

	TNodeState state = (TNodeState)(int)node->color;
	if (node->left != &nil) {
		state = (TNodeState)(state | HAS_LEFT);
	}
	if (node->right != &nil) {
		state = (TNodeState)(state | HAS_RIGHT);
	}

	file.write((char *)&state, sizeof(TNodeState));

	//:: означает, что функция из внешнего верхнего пространства (не в этом классе, а лежит в другом классе или вообще в классе не лежит)
	::Serialize(node->key, file);
	::Serialize(node->val, file);
	if (state & HAS_LEFT) {				//наложение маски
		Serialize(node->left, file);
	}
	if (state & HAS_RIGHT) {
		Serialize(node->right, file);
	}
}

TRBTree::TNode *TRBTree::Deserialize(TNode *parent, std::ifstream &file) {
	TNodeState state;
	file.read((char *)&state, sizeof(TNodeState));
	if (state == EMPTY_TREE) {
		return &nil;
	}

	TKey key;
	TVal val;
	if (!::Deserialize(key, file)) {
		std::cerr << "ERROR: Problem with key parsing. Check the file format.\n";
		return nullptr;
	}
	if (!::Deserialize(val, file)) {
		std::cerr << "ERROR: Problem with value parsing. Check the file format.\n";
		return nullptr;
	}
	TNode *node = CreateNode(parent, key, val, TColor(state & RED));
	node->left = node->right = &nil;
	if (state & HAS_LEFT) {
		node->left = Deserialize(node, file);
		node->left->parent = node;
	}
	if (state & HAS_RIGHT) {
		node->right = Deserialize(node, file);
		node->right->parent = node;
	}
	return node;
}

bool TRBTree::Load(const char *path) {
	std::ifstream file;
	file.open(path, std::ios::in | std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "File does not exist\n";
		return false;
	}
	if (file.eof()) {
		std::cerr << "File is empty\n";
		file.close();
		return false;
	}
	if (filesize(path) < FILE_SIGNATURE_LEN) {
		std::cerr << "Invalid file format\n";
		file.close();
		return false;
	}
	char signature[FILE_SIGNATURE_LEN];
	file.read((char *)&signature, FILE_SIGNATURE_LEN);
	if (strcmp(signature, FILE_SIGNATURE) != 0) {
		std::cerr << "Invalid file format\n";
		file.close();
		return false;
	}

	if (root != &nil) {
		DestroyR(root);
	}
	TNode * tmp_node = Deserialize(&nil, file);
	file.close();
	if (tmp_node) {
		this->root = tmp_node;
		return true;
	}
	return false;
}

TRBTree::TRBTree() {
	this->nil.parent = this->nil.left = this->nil.right = nullptr;
	this->nil.color = BLACK;
	this->nil.val = 0;
	this->root = &nil;
}

bool TRBTree::Insert(const TKey &key, const TVal &number) {
	TNode *parent = &nil;
	TNode *current = this->root;
	int cmp = 0;
	while (current != &nil) {
		parent = current;
		cmp = CompareKey(key, current->key);
		if (cmp == 0)
			return false;
		else if (cmp < 0)
			current = current->left;     //к родителю присоединяем левую ветвь
		else
			current = current->right;    //к родителю присоединяем правую ветвь
	}
	TNode *newnode = CreateNode(parent, key, number);
	if (parent == &nil) {
		this->root = newnode;
	}
	else if (cmp < 0) {
		parent->left = newnode;
	}
	else {
		parent->right = newnode;
	}
	InsertFix(newnode);
	return true;
}

TRBTree::TNode * TRBTree::SearchNode(const TKey &key) const {
	TNode *out = root;
	while (out != &nil) {
		int cmp = CompareKey(key, out->key);
		if (cmp == 0)
			break;
		else if (cmp < 0)
			out = out->left;
		else
			out = out->right;
	}
	return out;
}

bool TRBTree::Search(const TKey &key, TVal &result_val) const {
	const TNode *out = SearchNode(key);

	if (out != &nil) {
		result_val = out->val;
		return true;
	}
	else {
		return false;
	}
}

bool TRBTree::Delete(const TKey &key) {
	TNode *znode = SearchNode(key), *ynode, *xnode;
	if (znode == &nil || znode == nullptr) {
		return false;
	}

	ynode = znode;
	TColor ycolor = ynode->color;
	if (znode->left == &nil) {
		xnode = znode->right;
		Transplant(znode, znode->right);
	}
	else if (znode->right == &nil) {
		xnode = znode->left;
		Transplant(znode, znode->left);
	}
	else {
		ynode = Minimum(znode->right);
		ycolor = ynode->color;
		xnode = ynode->right;
		if (ynode->parent == znode)
			xnode->parent = ynode;
		else {
			Transplant(ynode, ynode->right);
			ynode->right = znode->right;
			ynode->right->parent = ynode;
		}
		Transplant(znode, ynode);
		ynode->left = znode->left;
		ynode->left->parent = ynode;
		ynode->color = znode->color;
	}
	if (ycolor == BLACK) {
		DeleteFix(xnode);
	}

	delete znode;
	return true;
}

bool TRBTree::Save(const char *path) {
	std::ofstream file;
	file.open(path, std::ios::out | std::ios::binary);
	if (!file.is_open()) {
		return false;
	}
	file.write((char*)FILE_SIGNATURE, FILE_SIGNATURE_LEN);
	Serialize(this->root, file);
	file.close();
	return true;
}

TRBTree::~TRBTree() {
	if (this->root != &nil) {
		DestroyR(this->root);
	}
	this->root = nullptr;
}
