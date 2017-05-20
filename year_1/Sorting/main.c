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
    Queue qq[2];
    Queue q1, q2;

    Item v;
    int i = 0; //переключатель рабочих очередей
    bool all; 
    
    qq[0] = *q;
    qq[1] = queue_create(QUEUE_DEFAULT_SIZE);
    do {
        if (queue_is_empty(qq[i])) {
            i = 1 - i;
            all = true;
        } else {
            all = false;
        }
        v = queue_get(qq[i]);
        q1 = queue_create(QUEUE_DEFAULT_SIZE);
        q2 = queue_create(QUEUE_DEFAULT_SIZE);
         
        while (!queue_is_empty(qq[i])){
            if (v > queue_first(qq[i])) {   //отрезок заканчивается, как только заканчивается инверсия(неупорядоченная пара)
                break;
            }
            queue_put(q1, v);               //Очередной элемент отрезка добавляется в хвост (последний элемент) активной очереди
            v = queue_get(qq[i]);           //Из рабочей очереди извлекается следующий
        }
        
        queue_put(q1, v);
        
        if (!queue_is_empty(qq[i])) {
            all = false; 
            v = queue_get(qq[i]);
            while (!queue_is_empty(qq[i])){
                if (v > queue_first(qq[i])) {   //отрезок заканчивается, как только заканчивается инверсия(неупорядоченная пара)
                    break;
                }
                queue_put(q2, v);               //Очередной элемент отрезка добавляется в хвост (последний элемент) активной очереди
                v = queue_get(qq[i]);           //Из рабочей очереди извлекается следующий
            }
            queue_put(q2, v);
            queue_merge(q1, q2, qq[1-i]);       //сливаем очереди, при этом записываем в конец результата слияния
        } else {                                //Поскольку q2 пуста, то слияние как таковое не выполняется. 
            while (!queue_is_empty(q1)) {       //Вместо этого упорядоченный отрезок из q1 доливается в qq[1 - i]
                queue_put(qq[1 - i], queue_get(q1));
            }
        }
        
        queue_destroy(&q1);
        queue_destroy(&q2);
    }
    while (!all);
    *q = qq[1 - i]; //отсортированная очередь из приемника помещается на место исходной
    queue_destroy(&qq[i]);
}

int main(void)
{
	Queue q1 = NULL;
    int val = 0, cap = 0;
    char s[7];

    printf("\nНаберите команду `help`, чтобы узнать команды, используемые в программе\n\n");
    printf("Для того, чтобы работать с программой, необходимо сначала создать очередь, для этого обратитесь к помощи команды `help`.\n\n");
    while (1) {
    	scanf("%6s", s);
    	if (!strcmp(s, "create") || !strcmp(s, "c")) {
    		scanf("%d", &cap);
    		q1 = queue_create(cap);
    	} else if (!strcmp(s, "put") || !strcmp(s, "p")) {
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
    	} else if (!strcmp(s, "new")) {
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
            printf("`new numb` === создает новую очередь размером numb.\n");
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
