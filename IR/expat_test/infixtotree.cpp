//Tacuma Solomon
//Computer Science III
//Infix to Expression Tree Conveter and Evaluator

//Algorithm to insert an expression in Infix  notation
//( e.g. 2 * 3 + (6 / 4) - 8 ) to a binary tree.
//In addition to the above requirement, you must implement
//an evaluation of the expression tree.
//This solution includes class templates.

//Precondition: Must be a single digit numbers

#include<iostream>
#include<string>
#include "TStack.h"
using namespace std;

struct node;

int priority(char);
node* makeNode(char);
void attachOperator(TStack<node*>&, TStack<node*>&); 
//pops operator off of the operators stack, 
//pops 2 operands of treenode stack and attaches them to operator
int evaluateTree(node*);



//int main(){
//
//
//
//		int answer = evaluateTree(treenodes.top());
//		cout << endl << "Evaluation: " << answer << endl;
//		cout << endl;
//		cout << "Would  you like to convert another expression? (y/n)";
//		cin >> again;
//	}
//	cout << endl;
//	Footer();
//	system("pause");
//	return 0;
//}


//Determines the priority of an operator
int priority(char op){
	if ((op =='+') || (op =='-'))
		return 1;
	if ((op =='/') || (op =='*'))
		return 2;
}

//Places a char from the input stack into a new treenode
node* makeNode(char info){
	node* childnode;
	childnode = new node;
	childnode->info = info;
	childnode->left = NULL;
	childnode->right = NULL;
	return childnode;
}

//pops an operator from a stack
//Builds a tree node with the _top two nodes in the
//treenode stacks as its left and right children. 
void attachOperator(TStack<node*>& treenodes, TStack<node*>& operators){
	node* operatornode = operators.top();
	operators.pop();
	operatornode->left = treenodes.top();
	treenodes.pop();
	operatornode->right = treenodes.top();
	treenodes.pop();
	treenodes.push(operatornode);
}

//Using a recursive function, the value of the expression is Calculated
int evaluateTree(node* treenode){
	int x,y,z;
	if ((treenode->info) == '+'||(treenode->info) == '-'||(treenode->info) == '*'||(treenode->info) == '/') {
		x = evaluateTree(treenode->left);
		y = evaluateTree(treenode->right);
		if (treenode->info=='+')
			z=x+y;
		else if (treenode->info=='-')
			z=x-y;
		else if (treenode->info=='*')
			z=x*y;
		else if (treenode->info=='/')
			z=x/y;
		return z;
	}
	else return treenode->info - '0';	
}

/*
-Infix to Expression Tree Creator-
-An expression tree is created from a user inputted infix expression-

Please enter an Infix Mathematical Expression
2*3/(2-1)+5*(4-1)

Evaluation: 21

Would  you like to convert another expression? (y/n)n



() Code by Tacuma Solomon
() Not for Redistribution or Reuse.

Press any key to continue . . .
*/