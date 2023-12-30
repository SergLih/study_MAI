#include "queue.h"

Queue * q_create(int maxCount) {
    Queue * q = (Queue*) malloc(sizeof(Queue));
    q->data = (q_item_t*) malloc(sizeof(q_item_t)*maxCount);
    q->front = 0;
    q->rear = -1;
    q->curCount = 0;
    q->maxCount = maxCount;
}

void q_destroy(Queue **q) {
    free((*q)->data);
    free((*q));
    *q = NULL;
}

q_item_t q_peek(Queue *q) {
   return q->data[q->front];
}

bool q_is_empty(Queue *q) {
   return q->curCount == 0;
}

bool q_is_full(Queue *q) {
   return q->curCount == q->maxCount;
}

int q_size(Queue *q) {
   return q->curCount;
}  

void q_push(Queue *q, q_item_t new_item) {

   if (!q_is_full(q)) {
	
      if (q->rear == q->maxCount-1) {
         q->rear = -1;            
      }       

      q->data[++q->rear] = new_item;
      q->curCount++;
   }
}

q_item_t q_pop(Queue *q) {
   q_item_t data = q->data[q->front++];
	
   if (q->front == q->maxCount) {
      q->front = 0;
   }
	
   q->curCount--;
   return data;  
}

bool q_is_in_queue(Queue *q, q_item_t item_to_search) {
    if (q->front <= q->rear) {
        for (int i = q->front; i<=q->rear; i++)
            if (q->data[i] == item_to_search)
                return true;
    } else if (!q_is_empty(q)) {
        for (int i = q->front; i<q->maxCount; i++)
            if (q->data[i] == item_to_search)
                return true;
        for (int i = 0; i<=q->rear; i++)
            if (q->data[i] == item_to_search)
                return true;
    }
    return false;
}

void q_print(Queue *q) {
    if (q->front <= q->rear) {
        for (int i = q->front; i <= q->rear; i++)
            printf("%3d", q->data[i]);
    } else if (!q_is_empty(q)) {
        for (int i = q->front; i < q->maxCount; i++)
            printf("%3d", q->data[i]);
        for (int i = 0; i <= q->rear; i++)
            printf("%3d", q->data[i]);
    }
    printf("\n");
    printf("%d || %d | %d | ", q->curCount, q->front, q->rear);
    for (int i = 0; i<q->maxCount; i++)
        printf("%3d", q->data[i]);
    printf("\n");
}
