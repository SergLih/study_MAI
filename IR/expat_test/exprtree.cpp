//
// Created by sergey on 6/4/21.
//

#include "exprtree.hpp"

TExprTree::TExprTree(string s) {
    TStack<TString> input; // stack for input string
    regex reg(R"([a-z][a-z\d]+|AND|OR|NOT|\(\))");
    sregex_token_iterator start(s.begin(), s.end(), reg), end;
    for (auto it = start; it != end; ++it) {
        cout << *it << endl;
        input.push(TString(*it));
    }


//pushes the contents of the string into the input stack
//        for (int i = 0; i < infix.length(); i++) {
//            input.push(infix[i]);
//        }
//        // While the input stack is not empty...
        while (!input.empty()) {
            TString temp(input.top());
            input.pop();
//            //If it is operand, make it into a Node, add it to output string.
////			if (isdigit(temp))
////				treeNodes.push(makeNode(temp));
//            //If it is Closing parenthesis, make into Node, push it on stack.
            if (temp == ")")
                operators.push(makeNode(temp));
            //If it is an operator, then
            else if ((temp == "OR") || (temp == "AND")) {
                bool pushed = false;
                while (!pushed) {
                    //If stack is empty, make Node and push operator on stack.
                    if (operators.empty()) {
                        operators.push(makeNode(temp));
                        pushed = true;
                    }
                        //If the _top of stack is closing parenthesis, make Node and push operator on stack.
                    else if (operators.top()->info == ")") {
                        operators.push(makeNode(temp));
                        pushed = true;
                    }
                        //If it has same or higher priority than the _top of stack, make Node and push operator on stack.
                    else if ((priority(temp) > priority(TString(operators.top()->info))) ||
                             (priority(temp) == priority(operators.top()->info))) {
                        operators.push(makeNode(temp));
                        pushed = true;
                    }
                        //Else pop the operator from the stack, perform attachOperator and add it to treeNodes. repeat step 5.
                    else {
                        attachOperator();
                    }
                }
            }
            //If it is a opening parenthesis, pop operators from stack and perform attachOperator
            //until a closing parenthesis is encountered. pop and discard the closing parenthesis.
            else if (temp == "(") {
                while (!(operators.top()->info == ")")) {
                    attachOperator();
                }
                operators.pop(); // ')' is popped and discarded
            } else { //узел-слово
                treeNodes.push(makeNode(temp));
            }

        }
        //If there is no more input, unstack the remaining operators and perform attachOperator
        while (!operators.empty()) {
            attachOperator();
        }
}

//Determines the priority of an operator
int TExprTree::priority(TString op) {
    if ((op =="OR")/* || (op =='-')*/)
        return 1;
    if (/*(op =='/') ||*/ (op =="AND"))
        return 2;
}

//Places a char from the input stack into a new treenode
Node* TExprTree::makeNode(TString info) {
    Node* childnode;
    childnode = new Node;
    childnode->info = info;
    childnode->left = NULL;
    childnode->right = NULL;
    return childnode;
}

//pops an operator from a stack
//Builds a tree node with the _top two nodes in the
//treenode stacks as its left and right children.
void TExprTree::attachOperator(){
    Node* operatornode = operators.top();
    operators.pop();
    operatornode->left = treeNodes.top();
    treeNodes.pop();
    operatornode->right = treeNodes.top();
    treeNodes.pop();
    treeNodes.push(operatornode);
}


