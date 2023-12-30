#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
//#include <unistd.h>
#include <pthread.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <time.h>

#define CSC(call)  					\
do {								\
	cudaError_t res = call;			\
	if (res != cudaSuccess) {		\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);					\
	}								\
} while(0)

double *arr;
int n;
int threadsCount;

pthread_t *threads;		//динамич.массив потоков


typedef struct _params {
	int globalThreadIdx; //номер потока
} Params;


__global__ void kernel(double *arr, int n) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	while(idx < n) {
		arr[idx] *= arr[idx];
		idx += offset;
	}
}

void* cpu_kernel(void *dummyPtr){
    Params *p = (Params *)dummyPtr;
    int idx = p->globalThreadIdx;
	while(idx < n) {
		arr[idx] *= arr[idx];
		idx += threadsCount;
	}
	return NULL;
}


int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "%s\n","Usage: N BlockNumber ThreadsNumber");
        exit(EXIT_FAILURE);
    }
    
    n = atoi(argv[1]); 
	arr = (double *)malloc(sizeof(double) * n);
	int blocksCount = atoi(argv[2]);
	int threadsInBlocksCount = atoi(argv[3]);
	threadsCount = blocksCount * threadsInBlocksCount;
	for(int i = 0; i < n; i++)
		arr[i] = i;

	double *dev_arr;
	
	//выделение памяти на видеокарте
	CSC(cudaMalloc(&dev_arr, sizeof(double) * n));
	//копирование данных из оперативной памяти на видеокарту
	CSC(cudaMemcpy(dev_arr, arr, sizeof(double) * n, cudaMemcpyHostToDevice));

	cudaEvent_t start, end;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&end));
	CSC(cudaEventRecord(start));

	kernel<<<blocksCount, threadsInBlocksCount>>>(dev_arr, n);
	CSC(cudaGetLastError());

	CSC(cudaEventRecord(end));
	CSC(cudaEventSynchronize(end));
	float t;
	CSC(cudaEventElapsedTime(&t, start, end));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(end));

	fprintf(stderr, "GPU: ready\nWorking time:     %f sec.\n", t);

	//копирование результата отключено, т.к. во времени не учитывается, а нужно
	//еще раз использовать исходный массив
	//CSC(cudaMemcpy(arr, dev_arr, sizeof(double) * n, cudaMemcpyDeviceToHost));
	CSC(cudaFree(dev_arr));

	//for(i = 0; i < n; i++)
	//	printf("%0.10e ", arr[i]);
	//printf("\n");
	//free(arr);
	
	
    time_t start0 = clock();
    threads = (pthread_t *) malloc(threadsCount * sizeof(pthread_t));
    Params * params = (Params *) malloc(threadsCount * sizeof(Params));
    for(int i = 0; i < threadsCount; i++) {
    	params[i].globalThreadIdx = i;
    	pthread_create(&threads[i], NULL, cpu_kernel, (void *) &params[i]);
    } 
    time_t end0 = clock();
    
    
    for(int i = 0; i < threadsCount; i++) {
    	pthread_join(threads[i], NULL);
    } 
    fprintf(stderr, "CPU: ready\n");
    fprintf(stderr, "Working time:     %f sec.\n", (double)(end0 - start0) / (double)CLOCKS_PER_SEC);
    

    free(threads);
    free(params);
    return 0;
}

























