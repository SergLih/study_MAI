#include "queue.h"


typedef struct _params {
    int from; //текущая вершина (начало ребра)
    int to;   //конец ребра
} Params;


int ** matrix; 
int n_ver;
int * dist;
Queue * qVer;
int max_n_threads;
int used_n_threads;
pthread_t *threads;
Params * params;

int ** input_from_adj_matr(int *n_ver) {
    int x, y, w;
    scanf("%d", n_ver);
    int ** matrix = (int**) malloc(sizeof(int*)* *n_ver);
    for (int i = 0; i < *n_ver; i++) {
        matrix[i] = (int *) malloc(sizeof(int) * *n_ver);
        for (int j = 0; j < *n_ver; j++) {
            scanf("%d", &matrix[i][j]);
        }
    }
    return matrix;
} 


/*int ** input_from_adj_list(int *n_ver)
{
    int n_edges = 0, x, y, w;
    scanf("%d %d\n", n_ver, &n_edges);
    int ** matrix = (int**) malloc(sizeof(int*)* *n_ver);
    for(int i = 0; i < *n_ver; i++) {
        matrix[i] = (int *) malloc(sizeof(int) * *n_ver);
        for (int j = 0; j < *n_ver; j++) {
            matrix[i][j] = 0;
        }
    }
    
    for(int i = 0; i < n_edges; i++) {
        scanf("%d %d %d", &x, &y, &w);
        matrix[x][y] = w;
    }
    
    return matrix;
}*/

void free_adj_matr(int ** matrix, int n_ver) {
    for (int i = 0; i < n_ver; i++) 
        free(matrix[i]);
    free(matrix);
}


void print_adj_matr(int **matrix, int n_ver) {
    printf("Adjacency matrix:\n");
    for (int i = 0; i < n_ver; i++) {
        for (int j = 0; j < n_ver; j++) {
            printf("%3d ", matrix[i][j]);
        }
        printf("\n");
    }
}

void print_dist() {
    for (int i = 0; i < n_ver; i++)
        printf("%4d", i);
    printf("\n");
    for (int i = 0; i < n_ver; i++) 
        if (dist[i] < INT_MAX)
            printf("%4d", dist[i]); 
        else
            printf("   -");
    printf("\n");
}

void * bfs_step(void *dummyPtr) {
    Params *p = (Params *)dummyPtr;
    if (dist[p->from] + matrix[p->from][p->to] < dist[p->to]) {
        dist[p->to] = dist[p->from] + matrix[p->from][p->to];
        if (!q_is_in_queue(qVer, p->to)) {
            if (q_is_full(qVer)) {
                fprintf(stderr, "Error: overflow of queue for adjacent edges\n");
                exit(EXIT_FAILURE);        
            }
            q_push(qVer, p->to);
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "%s\n", "Usage: ThreadsNumber");
        exit(EXIT_FAILURE);
    }
    
    max_n_threads = atoi(argv[1]);
    if (max_n_threads < 0) {
        fprintf(stderr, "%s\n", "Usage: ThreadsNumber must be positive");
        exit(EXIT_FAILURE);
    }
    
    
    threads = (pthread_t *) malloc(max_n_threads * sizeof(pthread_t));
    params = (Params *) malloc(max_n_threads * sizeof(Params));
    
    int start, end;
    matrix = input_from_adj_matr(&n_ver);
    scanf("%d %d", &start, &end);
    if (start < 0 || start >= n_ver || end < 0 || end >= n_ver) {
        fprintf(stderr, "Error: incorrect vertex number\n");
        exit(EXIT_FAILURE);        
    }
    
    qVer = q_create(n_ver);
    
    dist = (int*) malloc(sizeof(int)*n_ver);
    for (int i = 0; i < n_ver; i++)
        dist[i] = INT_MAX;
    dist[start] = 0;
    
    //print_dist();
    //print_adj_matr(matrix, n_ver);
     
    q_push(qVer, start);
    while (!q_is_empty(qVer)) {
        int cur_ver = q_pop(qVer);
        //printf("Processing vertex %d\n", cur_ver);
        Queue * qAdj = q_create(n_ver);
        
        for (int j = 0; j < n_ver; j++) 
            if (matrix[cur_ver][j] != 0) { 
                if (q_is_full(qAdj)) {
                    fprintf(stderr, "Error: overflow of queue for adjacent edges\n");
                    exit(EXIT_FAILURE);        
                }
                q_push(qAdj, j);
            }
        
        while (!q_is_empty(qAdj)) {
            int used_n_threads = 0;
            for (int i = 0; i < max_n_threads && !q_is_empty(qAdj); i++) {
                params[i].from = cur_ver;
                params[i].to = q_pop(qAdj);
                //printf("Processing edge %d->%d\n", cur_ver, params[i].to);
                pthread_create(&threads[i], NULL, bfs_step, (void *) &params[i]);
                used_n_threads++;
            }
            //printf("used thr %d\n", used_n_threads);
            for (int k = 0; k < used_n_threads; k++) {
                pthread_join(threads[k], NULL);
            }
        }
        //print_dist();
        //printf("Q:"); q_print(qVer);
        q_destroy(&qAdj);
    }
    
    printf("Shortest path in the graph from %d to %d: %d\n", start, end, dist[end]);
    //print_dist();
 
    q_destroy(&qVer);
    free_adj_matr(matrix, n_ver);
    free(dist);
    free(threads);
    free(params);
    return 0;
}
