#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
//#include <unistd.h>
#include <pthread.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <time.h>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#define CSC(call) do { \
	cudaError_t res = call;	\
	if (res != cudaSuccess) { \
		fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(res)); \
		exit(0); \
	} \
} while (0)

typedef struct _params {
	int threadIdx;
	int step;
	int maxPos;
} Params;

pthread_t * threads;
int blocksCount;
int threadsInBlocksCount;
int threadCountCPU;
double *h_a;
int n;


void print(double *a, int n)
{
	for (int row = 0; row < n; row ++) {
		for (int col = 0; col < n; col ++) {
			printf("%le\t", a[col * n + row]);
		}
		printf("\n");
	}
	printf("\n");
}

void print0(double *a, int n, int step)
{
	for (int row = 0; row < n; row ++) {
		for (int col = 0; col < n; col ++) {
			if(row > col && col < step)
				printf("%le\t", 0.0);
			else
				printf("%le\t", a[col * n + row]);
		}
		printf("\n");
	}
	printf("\n");
}

__global__ void gaussElim(int n, double *a, int step) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;

	for (int row = idx; row < n; row += offsetx) {
		for (int col = idy; col < n; col += offsety) {
			if (row <= step || col <= step) {
				continue;
			}

			double coef = a[step * n + row] / a[step * n + step];
			a[col * n + row] -= coef * a[col * n + step];
		}
	}
}

void* cpuGaussElim(void *dummyPtr){
	Params *p = (Params *)dummyPtr;
	int idx = p->threadIdx;
	int offset = threadCountCPU;
	int step = p->step;

	for (int z = idx; z < n*n; z += offset) {
		int row = z % n;
		int col = z / n;
		if (row <= step || col <= step) {
			continue;
		}

		double coef = h_a[step * n + row] / h_a[step * n + step];
		h_a[col * n + row] -= coef * h_a[col * n + step];
	}
	return NULL;
}

__global__ void swapRows(int n, double *a, int step, int maxPos) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	for (int col = idx; col < n; col += offsetx) {
		if (col < step) {
			continue;
		}
		double tmp = a[col * n + step];
		a[col * n + step] = a[col * n + maxPos];
		a[col * n + maxPos] = tmp;
	}
}

void* cpuSwapRows(void *dummyPtr){
    Params *p = (Params *)dummyPtr;
	int idx = p->threadIdx;
	int offsetx = threadCountCPU;
	int step = p->step;
	int maxPos = p->maxPos;
	for (int col = idx; col < n; col += offsetx) {
		if (col < step) {
			continue;
		}
		double tmp = h_a[col * n + step];
		h_a[col * n + step] = h_a[col * n + maxPos];
		h_a[col * n + maxPos] = tmp;
	}
	return NULL;
}


struct cmpAbsDoubles {
	__host__ __device__ bool operator()(double a, double b) {
		return fabs(a) < fabs(b);
	}
};


double detOfDiagonalMatrix(int replacements)	//h_a
{
	int neg_count = 0;
	double sumlog = 0;
	bool zero = false;
	for(int i = 0; i < n; i++)
	{
		double x = h_a[n*i + i];
		if(x < 0)
			neg_count++;
		else if(fabs(x) <= 1e-7) {
			zero = true;
			break;
		}
		sumlog += log(fabs(x));
	}
	return zero ? 0 : exp(sumlog) * (((neg_count + replacements) % 2) == 0 ? 1 : -1);
}


int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "%s\n","Usage: BlockNumber ThreadsNumber");
        exit(EXIT_FAILURE);
    }

	blocksCount = atoi(argv[1]);
	threadsInBlocksCount = atoi(argv[2]);
	threadCountCPU = blocksCount * threadsInBlocksCount;
	threadCountCPU = threadCountCPU * threadCountCPU;

	scanf("%d", &n);

	h_a  = (double*)malloc(sizeof(double)*n*n);

	for (int row = 0; row < n; row++) {
		for (int col = 0; col < n; col++) {
			scanf("%le", &h_a[col * n + row]);
		}
	}

	double *devCols;
	CSC(cudaMalloc(&devCols, sizeof(double) * n * n));
	CSC(cudaMemcpy(devCols, h_a, sizeof(double) * n * n, cudaMemcpyHostToDevice));

	int replacements = 0;
	time_t start0 = clock();
	threads = new pthread_t[threadCountCPU];
	Params * params = new Params[threadCountCPU];
	for(int i = 0; i < threadCountCPU; i++)
		params[i].threadIdx = i;
	for (int step = 0; step < n - 1; step++) {
		for(int i = 0; i < threadCountCPU; i++)
				params[i].step = step;
		double * hPtr = h_a + step * (n + 1);
		//thrust::host_ptr<double> devPtr = thrust::device_pointer_cast(devCols + step * (n + 1));
		double * h_maxPtr = thrust::max_element(hPtr, hPtr + (n - step), cmpAbsDoubles());
		//thrust::device_ptr<double> maxPtr = thrust::max_element(devPtr, devPtr + (n - step), cmpAbsDoubles());
		int maxPos = &h_maxPtr[0] - &hPtr[0] + step;
		//printf("step %d maxPos %d\n", step, maxPos);
		if(maxPos != step){
			replacements++;
			for(int i = 0; i < threadCountCPU; i++){
				params[i].maxPos = maxPos;
				pthread_create(&threads[i], NULL, cpuSwapRows, (void *) &params[i]);
			}
			for(int i = 0; i < threadCountCPU; i++) {
		    	pthread_join(threads[i], NULL);
		    } 
		}
		//printf("step %d swap\n", step);
		//print(h_a, n);

		for(int i = 0; i < threadCountCPU; i++)
		{
			pthread_create(&threads[i], NULL, cpuGaussElim, (void *) &params[i]);
		}
		for(int i = 0; i < threadCountCPU; i++) {
	    	pthread_join(threads[i], NULL);
	    } 
	    //printf("step %d elim\n", step);
	    //print(h_a, n);

		//gaussElim<<<dim3(blocksCount, blocksCount), dim3(threadsInBlocksCount, threadsInBlocksCount)>>>(n, devCols, step);
		// print0(a, n, step);
	
	}
	time_t end0 = clock();
	//print(h_a, n);
	//print0(h_a, n, n);

    double res = detOfDiagonalMatrix(replacements);
    
    fprintf(stderr, "CPU: ready\n");
    fprintf(stderr, "Working time:     %f sec.  Result: %.10le\n", 
    	(double)(end0 - start0) / (double)CLOCKS_PER_SEC, res);


	replacements = 0;
	int maxPos;
	cudaEvent_t start, end;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&end));
	CSC(cudaEventRecord(start));
	for (int step = 0; step < n - 1; step++) {
		thrust::device_ptr<double> devPtr = thrust::device_pointer_cast(devCols + step * (n + 1));
		thrust::device_ptr<double> maxPtr = thrust::max_element(devPtr, devPtr + (n - step), cmpAbsDoubles());
		maxPos = &maxPtr[0] - &devPtr[0] + step;
		//printf("step %d maxPos %d\n", step, maxPos);
		if(maxPos != step){
			replacements++;
			swapRows<<<blocksCount*blocksCount, threadsInBlocksCount*threadsInBlocksCount>>>(n, devCols, step, maxPos);
			CSC(cudaGetLastError());
		}
		//CSC(cudaMemcpy(h_a, devCols, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
		//printf("step %d swap\n", step);
		//print(h_a, n);

		gaussElim<<<dim3(blocksCount, blocksCount), dim3(threadsInBlocksCount, threadsInBlocksCount)>>>(n, devCols, step);

		//CSC(cudaMemcpy(h_a, devCols, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
		//printf("step %d elim\n",step);
		//print(h_a, n);
		// print0(a, n, step);
	}
	CSC(cudaGetLastError());

	CSC(cudaEventRecord(end));
	CSC(cudaEventSynchronize(end));
	float t;
	CSC(cudaEventElapsedTime(&t, start, end));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(end));

	CSC(cudaMemcpy(h_a, devCols, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
	//print(h_a, n);
	//print0(h_a, n, n);
	res = detOfDiagonalMatrix(replacements);

	fprintf(stderr, "GPU: ready\nWorking time:     %f sec.  Result: %.10le\n", t/1000, res);
	
	CSC(cudaFree(devCols));

	free(h_a);
	delete[] threads;
	delete[] params;
	return 0;
}
