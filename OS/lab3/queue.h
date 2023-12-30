#ifndef _QUEUE_
#define _QUEUE_

#include <limits.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

//#define MAX_QUEUE_SIZE 10000

typedef int q_item_t;

typedef struct _queue {
    q_item_t *data;
    int front;
    int rear;
    int curCount;
    int maxCount;
} Queue;

Queue *  q_create(int maxCount);
void     q_destroy(Queue **q);
q_item_t q_peek(Queue *q);
bool     q_is_empty(Queue *q);
bool     q_is_full(Queue *q);
int      q_size(Queue *q);
void     q_push(Queue *q, q_item_t new_item);
q_item_t q_pop(Queue *q);
bool     q_is_in_queue(Queue *q, q_item_t item_to_search);
void     q_print(Queue *q);

#endif
