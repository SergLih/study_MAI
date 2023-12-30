#include <stdio.h>
#include <stdlib.h>

#define CSC(call)  					\
do {								\
	cudaError_t res = call;			\
	if (res != cudaSuccess) {		\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);					\
	}								\
} while(0)


__global__ void kernel(double *arr, int n) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	while(idx < n) {
		arr[idx] *= arr[idx];
		idx += offset;
	}
}

int main() {
	int i, n = 0;
	scanf("%d", &n);
	double *arr = (double *)malloc(sizeof(double) * n);
	for(i = 0; i < n; i++)
		scanf("%lf", &arr[i]);

	double *dev_arr;

	//выделение памяти на видеокарте
	CSC(cudaMalloc(&dev_arr, sizeof(double) * n));
	//копирование данных из оперативной памяти на видеокарту
	CSC(cudaMemcpy(dev_arr, arr, sizeof(double) * n, cudaMemcpyHostToDevice));

	/*cudaEvent_t start, end;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&end));
	CSC(cudaEventRecord(start));*/

	kernel<<<256, 256>>>(dev_arr, n);
	CSC(cudaGetLastError());

	/*CSC(cudaEventRecord(end));
	CSC(cudaEventSynchronize(end));
	float t;
	CSC(cudaEventElapsedTime(&t, start, end));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(end));*/

	//printf("time = %f\n", t);

	CSC(cudaMemcpy(arr, dev_arr, sizeof(double) * n, cudaMemcpyDeviceToHost));
	CSC(cudaFree(dev_arr));

	for(i = 0; i < n; i++)
		printf("%0.10e ", arr[i]);
	printf("\n");
	free(arr);
	return 0;
}
