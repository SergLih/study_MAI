#ifndef _QUEUE_H
#define _QUEUE_H

#define QUEUE_SUCCESS 0
#define QUEUE_ERROR -1

typedef int Item;
typedef struct _queue *Queue;

Queue queue_create(int queue_size);
int queue_put(Queue q, Item value);
Item queue_get(Queue q);
bool queue_is_empty(Queue q);
void queue_destroy(Queue *q);

Item queue_first(Queue q);
Item queue_last(Queue q);

#endif
