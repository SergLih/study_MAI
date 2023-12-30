//
// Created by sergey on 6/4/21.
//

#ifndef EXPAT_TEST_EXPRTREE_HPP
#define EXPAT_TEST_EXPRTREE_HPP

#include "TStack.h"
#include "tstring.h"
#include <iostream>
#include <string>
#include <string_view>
#include <regex>

class Node {
public:
    TString info;
    Node* left;
    Node* right;
};

class TExprTree{
private:
    int priority(TString);
    //void evaluateTree_rec(Node* start);
public:
    TStack<Node*> operators; // stack for operator pointer addresses
    TStack<Node*> treeNodes;
    TExprTree(string s);

    Node* makeNode(TString);
    void attachOperator();

    //void evaluateTree();
};

#endif //EXPAT_TEST_EXPRTREE_HPP

