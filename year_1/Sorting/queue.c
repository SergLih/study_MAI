#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "queue.h"

typedef struct _queue {
    int head;
    int last;
    int size;
    Item *data;
} _queue;

Queue queue_create(int queue_size)
{
    Queue q = (Queue) malloc(sizeof(struct _queue));

    q->size = queue_size + 1;
    q->data = (Item *) malloc(sizeof(Item) * q->size);

    q->head = q->size;
    q->last = 0;

    return q;
}

int queue_put(Queue q, Item value)
{
    if ((q->last + 1) % q->size == q->head % q->size)
        return QUEUE_ERROR;
    
    q->data[q->last++] = value;
    q->last %= q->size;
    
    return QUEUE_SUCCESS;
}

bool queue_is_empty(Queue q)
{
    return (q->head % q->size) == q->last;
}

Item queue_get(Queue q)
{
    q->head %= q->size;
    return q->data[q->head++];
}

void queue_destroy(Queue *q)
{
    free((*q)->data);
    free(*q);
}

Item queue_first(Queue q)
{
    return q->data[q->head % q->size];
}

Item queue_last(Queue q)
{
    return q->data[q->last % q->size];
}
