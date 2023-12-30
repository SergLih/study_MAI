#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <memory>
#include <mutex>

template <class T>
class TreeNode;

template <class T>
class TBinaryTree;

template <class T>
using TreeNodePtr = std::shared_ptr<TreeNode<T> >;		//C++2011 примерно как typedef

#endif