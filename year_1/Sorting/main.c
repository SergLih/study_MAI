#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "queue.h"

#define QUEUE_DEFAULT_SIZE 100

void queue_reverse(Queue q)
{
    if (queue_is_empty(q))
        return;

    Item n = queue_get(q);
    queue_reverse(q);
    queue_put(q, n);
}

void queue_print_r(Queue q)
{
	if (queue_is_empty(q))
		return;
    Item n = queue_get(q);
    printf("%d ", n);
	queue_print_r(q);
    queue_put(q, n);
}

void queue_print(Queue q)
{
	if (queue_is_empty(q))
	{
		printf("queue is empty\n");
		return;
	}
	queue_print_r(q);
	queue_reverse(q);
	printf("\n");
}

void queue_merge(Queue q1, Queue q2, Queue q_res) //res для добавления отсортированного отрезка к очереди
{

    while(!queue_is_empty(q1) && !queue_is_empty(q2))
        if(queue_first(q1) <= queue_first(q2))
            queue_put(q_res, queue_get(q1));
        else
            queue_put(q_res, queue_get(q2));

    while (!queue_is_empty(q1))
        queue_put(q_res, queue_get(q1));

    while (!queue_is_empty(q2))
        queue_put(q_res, queue_get(q2));
}

void queue_merge_sort(Queue *q)
{
    if (*q == NULL || queue_is_empty(*q))
        return;
    Queue q_left, q_right;
    q_left  = queue_create(QUEUE_DEFAULT_SIZE);
    q_right = queue_create(QUEUE_DEFAULT_SIZE);

    int k = 0;      //razmer ocheredi
    while (!queue_is_empty(*q)) {
        queue_put(q_right, queue_get(*q));
        k++;
    }
    if(k == 1)
    {
        queue_put(*q, queue_get(q_right));
        queue_destroy(&q_left);
        queue_destroy(&q_right);
    	return;
    }

    for(int i = 0; i < k / 2; i++) {
        queue_put(q_left, queue_get(q_right));
    }

    //printf("\tq_left:  "); queue_print(q_left);
    //printf("\tq_right: "); queue_print(q_right);

    queue_merge_sort(&q_left);
    queue_merge_sort(&q_right);

    //printf("\tq_left:  "); queue_print(q_left);
    //printf("\tq_right: "); queue_print(q_right);

    queue_merge(q_left, q_right, *q);
    queue_destroy(&q_left);
    queue_destroy(&q_right);
}

int main(void)
{
	Queue q1 = NULL;
    int val = 0, cap = 0;
    char s[6];

    printf("\nНаберите команду `help`, чтобы узнать команды, используемые в программе\n\n");
    printf("Для того, чтобы работать с программой, необходимо сначала создать очередь, для этого обратитесь к помощи команды `help`.\n\n");
    while (1) {
    	scanf("%6s", s);
		if (!strcmp(s, "put") || !strcmp(s, "p")) {
    		while(scanf("%d", &val))
    			if(queue_put(q1, val))
                    fprintf(stderr, "\nНевозможно добавить цифру %d в очередь, т.к. она переполнена.\n\n", val);
    	} else if (!strcmp(s, "get") || !strcmp(s, "g")) {
    		printf("Из очереди выведен элемент %d\n\n", queue_get(q1));
    	} else if (!strcmp(s, "print") || !strcmp(s, "pr")) {
    		if(queue_is_empty(q1)) {
                fprintf(stderr, "\nОчередь пуста.\n\n");
                continue;
            }
            queue_print(q1);
            queue_reverse(q1);
            printf("\n\n");
    	} else if (!strcmp(s, "create") || !strcmp(s, "c")) {
    		if (q1 != NULL) {
    			queue_destroy(&q1);
    			q1 = NULL;
    		}
    		scanf("%d", &cap);
    		q1 = queue_create(cap);
    	} else if (!strcmp(s, "sort") || !strcmp(s, "s")) {
            queue_merge_sort(&q1);
    	} else if (!strcmp(s, "help")) {
            printf("\n`сreate numb` и `c numb` === создает очередь размером numb.\n");
            printf("`put n1 n2 ...` и `p n1 n2 ...` === добавляет в очередь элементы n1, n2 ...\n");
            printf("`get` и `g` === выводит первый элемент из очереди.\n");
            printf("`print` и `pr` === печатает очередь.\n");
            printf("`sort` === сортирует очередь методом слияния.\n");
            printf("`quit` и `q` === заканчивает работу программы.\n\n");
    	} else if (!strcmp(s, "quit") || !strcmp(s, "q")) {
    		if (q1 != NULL) {
    			queue_destroy(&q1);
    			q1 = NULL;
    		}
    		break;
    	} else {
            printf("\n\nВведены некорректные данные. Воспользуйтесь командой `help`, чтобы подробнее узнать команды.\n\n");
        }
    }

    return 0;
}
