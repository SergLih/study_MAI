#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "list.h"
#include "iterator.h"


Node *NodeCreate(char * string_value)
{
    Node *res = (Node *)malloc(sizeof(Node));
    if (string_value != NULL) {
		size_t len = strlen(string_value);
		res->value = (char *)malloc(sizeof(char) * (len+1));
		strcpy(res->value, string_value);
		res->next = NULL;
	} else {//если был передан NULL, то передается пустая строка
		res->value = (char *)malloc(sizeof(char));
		*(res->value) = '\0';
		res->next = res;
    }

    return res;
}

void NodeDestroy(Node **node)
{
	free((*node)->value);
	free(*node);
	*node = NULL;
}

List *ListCreate()
{
    List *list = (List *)malloc(sizeof(List));
    list->head = NULL;
    list->barrier = NodeCreate(NULL);
    list->size = 0;
    return list;
}

void ListDestroy(List **list)
{
	Iterator iter, iter2;
	IteratorStart(&iter, **list);
	IteratorStart(&iter2, **list);
	
	while(IteratorNext(&iter)) {
		NodeDestroy(&(iter2.node));
		iter2.node = iter.node;
	}
	
	NodeDestroy(&((*list)->barrier));
    free(*list);
    *list = NULL;
    
}

void ListAddNode(List *list, char * string_value)  //вставка в конец
{
	Node * newNode = NodeCreate(string_value);
	list->barrier->next->next = newNode;// связываем старый "последний" эл-т с новым последним
	list->barrier->next = newNode;	// из барьера ставим указатель на новый последний
	newNode->next = list->barrier;  // ставим у последнего next на барьер
	if(list->size == 0)
		list->head = newNode;
	list->size++;
}

void ListPrint(List *list)
{
    if (list->size == 0) {
        printf("List is empty\n");
    } else {
    	Iterator iter;
    	IteratorStart(&iter, *list);
        printf("List:\n");
        do
        	printf("%s ", IteratorFetch(&iter));	
        while (IteratorNext(&iter));
        printf("\n");
    }
}


void ListInsert(List *list, int index, char * value)
{							  //вставка в конец
    if (index>=list->size) {//+защита от индекса > размер списка 
        ListAddNode(list, value);	
        return;
    } else if(index <= 0)		{//вставка в начало + защита от отриц.индексов 
    	if(list->size == 0) {	//если список пуст
    		ListAddNode(list, value);	//вставка "в конец" пустого
        	return;						// = "в начало"     списка
    	} else {
            Node *new_node = NodeCreate(value);
            new_node->next = list->head;
            list->head = new_node;
        }
    } else {
        Node *new_node = NodeCreate(value);
        Iterator iter;
        IteratorStart(&iter, *list);
        for (int j = 0; j < index - 1 && IteratorNext(&iter); j++)
            ;
        new_node->next = iter.node->next;	//соединение нового узла
        iter.node->next = new_node;			//с соседними
    }
    list->size++;
}


bool ListDeleteNode(List *list, Node *node_to_delete)
{
    if (list->size == 0 || node_to_delete == NULL) {
        return false;
    }
    if (list->size == 1) {                //Если удаляем единственный элемент
        list->head = list->barrier;
    } else if (node_to_delete == list->head) {  //Если удаляем первый элемент
        list->head = list->head->next;			//обновляем указ-ль на начало
    } else {	//Во всех остальных случаях нужно получить указ-ль на предыдущий
    			//по отношению к удаляемому элементу  
    	Iterator iter;
    	IteratorStart(&iter, *list);
    	while(iter.node->next != node_to_delete) {	//Крутим цикл пока не дойдем до 
    		if(IteratorNext(&iter) == false)//предыд. по отношению к удаляемому элементу
   				return false;		//(если нет такого узла в списке,сразу ошибка)
   		}
   		iter.node->next = node_to_delete->next;
    }
    NodeDestroy(&node_to_delete);
    list->size--;
    return true;
}


Node *ListFindValue(List *list, char * value_to_find)
{
	if(list == NULL || value_to_find == NULL || list->size == 0)
		return NULL; 

	Iterator iter;
    IteratorStart(&iter, *list);
    do
    {
        if(strcmp(value_to_find, IteratorFetch(&iter)) == 0) {
        	return iter.node;
        }
    }
    while(IteratorNext(&iter));
    return NULL;
}


Node *ListFindNumber(List *list, int number)
{
    if ( number < 0 || number >= list->size) {
        return NULL;
    }
    Iterator iter;
    IteratorStart(&iter, *list);
    for (int j = 0; j < number; ++j) {
        IteratorNext(&iter);
    }
    return iter.node;
}

void ListExchange(List *list)
{
    if (list->size == 0 || list->size == 1) {
        return;
    }
    Iterator first;
    Iterator med;
    Iterator last;
    IteratorStart(&first, *list);
    IteratorStart(&med, *list);
    for (int j = 1; j < list->size / 2 ; ++j) {
        IteratorNext(&med);
    }
    last = med;
    IteratorNext(&last);
   	
    med.node->next = list->barrier;
    list->head = last.node;
    while (last.node->next != list->barrier)
    	IteratorNext(&last);
    
    last.node->next = first.node;
    list->barrier->next = med.node;
}
