#include <iostream>
#include "exprtree.hpp"
using namespace std;

const int MAX_ITEMS = 20;
//   The user of this file must provied a file "TItem.h" that defines:
//       TItem : the class definition of the objects on the stack.
//       MAX_ITEMS: the maximum number of items on the stack.

//   Class specification for Stack ADT in file Stack1.h

class FullStack
// Exception class thrown by push when stack is full.
{};

class EmptyStack
// Exception class thrown by pop and top when stack is emtpy.
{};

template<class TItem>
class TStack
{
public:

   TStack();
   // Class constructor.
   bool is_full() const;
   // Function: Determines whether the stack is full.
   // Pre:  Stack has been initialized.
   // Post: Function value = (stack is full)
   bool empty() const;
   // Function: Determines whether the stack is empty.
   // Pre:  Stack has been initialized.
   // Post: Function value = (stack is empty)
   void push(TItem item);
   // Function: Adds newItem to the _top of the stack.
   // Pre:  Stack has been initialized.
   // Post: If (stack is full), FullStack exception is thrown;
   //     otherwise, newItem is at the _top of the stack.
   void pop();
   // Function: Removes _top item from the stack.
   // Pre:  Stack has been initialized.
   // Post: If (stack is empty), EmptyStack exception is thrown;
   //     otherwise, _top element has been removed from stack.
   TItem top();
   // Function: Returns a copy of _top item on the stack.
   // Pre:  Stack has been initialized.
   // Post: If (stack is empty), EmptyStack exception is thrown;
   //     otherwise, _top element has been removed from stack.
private:
   int _top;
   TItem  items[MAX_ITEMS];
};


template<class TItem>
TStack<TItem>::TStack()
{
  _top = -1;
}

template<class TItem>
bool TStack<TItem>::empty() const
{
  return (_top == -1);
}

template<class TItem>
bool TStack<TItem>::is_full() const
{
  return (_top ==  MAX_ITEMS-1);
}

template<class TItem>
void TStack<TItem>::push(TItem newItem) {
    if (is_full()) {
        cerr << "Stack is full!";
        return;
   }
  _top++;
  items[_top] = newItem;
}

template<class TItem>
void TStack<TItem>::pop()
{
  if(empty() ) {
      cerr << "Stack is empty!";
      return;
  }
  _top--;
}

template<class TItem>
TItem TStack<TItem>::top()
{
  if (empty())
    throw EmptyStack();
  return items[_top];
}    
