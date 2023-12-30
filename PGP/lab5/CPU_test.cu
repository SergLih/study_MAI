#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

#define BLOCK_SIZE 256
#define LOG_NUM_BANKS 5

#define CF_OFFS(x) (x >> LOG_NUM_BANKS)
#define CF_CORR(x) (x + (x >> LOG_NUM_BANKS))

#define CSC(call) do { \
	cudaError_t pixels = call;	\
	if (pixels != cudaSuccess) { \
		fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(pixels)); \
		exit(0); \
	} \
} while (0)

__global__ void scan_cum_sum(int *arr_count, int *prefix_sum) {
	__shared__ int temp[CF_CORR(2*BLOCK_SIZE)];

	int tid = threadIdx.x;
	
	int n = BLOCK_SIZE;
	int ai = tid;
	int bi = tid + (n /2);

	temp[CF_CORR(ai)] = arr_count[ai]; 
	temp[CF_CORR(bi)] = arr_count[bi]; 
	int offset = 1;
	for ( int d = n>>1; d > 0; d >>= 1 ) {
		__syncthreads ();
		if (tid < d) {
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;
			//ai += CF_OFFS(ai);
			//bi += CF_OFFS(bi);
			temp[CF_CORR(bi)] += temp[CF_CORR(ai)];
		}
		offset <<= 1;
	}
	if (tid == 0)
		temp[CF_CORR(n-1)] = 0;
	for (int d = 1; d < n; d <<= 1 ) {
		offset >>= 1;
		__syncthreads ();
		if (tid < d) {
			int ai = CF_CORR(offset * (2 * tid + 1)-1);
			int bi = CF_CORR(offset * (2 * tid + 2)-1);
			int t      = temp[ai];
			temp[ai]   = temp [bi];
			temp [bi] += t;
		}
	}

	__syncthreads ();

	if (tid < BLOCK_SIZE - 1)
		prefix_sum[tid] = temp[CF_CORR(tid + 1)];
	else
		prefix_sum[tid] = temp[CF_CORR(tid)] + arr_count[BLOCK_SIZE - 1];
}

__global__ void histogram_count_values(unsigned char *arr, int size, int *arr_count) {
	__shared__ int temp[BLOCK_SIZE];
	for (int i = threadIdx.x; i < BLOCK_SIZE; i += blockDim.x)  {
		temp[i] = 0;
		arr_count[i] = 0;
	}
	__syncthreads();
	int offset = blockDim.x * gridDim.x;
	for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i+=offset) {
	    atomicAdd(&temp[arr[i]], 1);
	}
	__syncthreads();
	for (int i = threadIdx.x; i < BLOCK_SIZE; i += blockDim.x) {
	 	atomicAdd(&(arr_count[i]), temp[i]);
	}
}

__global__ void count_sort_gpu(unsigned char* arr_in, unsigned char* arr_out, int* arr_count, int size) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = gridDim.x * blockDim.x;

	//please work!
	for (int i = tid; i < size; i += offset) {
		arr_out[atomicSub(&arr_count[arr_in[i]], 1) - 1] = arr_in[i];
	}
}

void count_sort_cpu(unsigned char *arr, int n) {
    int counts[BLOCK_SIZE];
    unsigned char *aux = (unsigned char*)malloc(sizeof(unsigned char) * n);
    for (int i = 0; i < BLOCK_SIZE; i++) {
        counts[i] = 0;
    }
    for (int i = 0; i < n; i++) {
        counts[arr[i]]++; 
    }
    for (int i = 1; i < BLOCK_SIZE; i++) {
        counts[i] += counts[i - 1];
    }
    for (int i = n; i >= 0; i--) {
        aux[--counts[arr[i]]] = arr[i];
    }
    for (int i = 0; i < n; i++) {
        arr[i] = aux[i];
    }
    free(aux);
}

int main() {
	int n;

	freopen(NULL, "rb", stdin);
	fread(&n, sizeof(int), 1 , stdin);
	unsigned char *data = (unsigned char*)malloc(sizeof(unsigned char) * n);
	fread(data, sizeof(unsigned char), n, stdin);
	fclose(stdin);
    cerr << n << endl;
    if (n == 1) {
		freopen(NULL, "wb", stdout);
		fwrite(data, sizeof(unsigned char), n, stdout);
		fclose(stdout);
        return 0;
    }
	

	//cpu version 	
    unsigned char *data_cpu = (unsigned char*)malloc(sizeof(unsigned char) * n);
	memcpy(data_cpu, data, sizeof(unsigned char) * n);
	time_t start0 = clock();
    count_sort_cpu(data_cpu, n);

    time_t end0 = clock();
    
    fprintf(stderr, "CPU: ready\n");
    fprintf(stderr, "Working time:     %f sec.", 
        (double)(end0 - start0) / (double)CLOCKS_PER_SEC);

    //gpu version

	unsigned char *arr_in;
	cudaMalloc(&arr_in, sizeof(unsigned char) * n);
	cudaMemcpy(arr_in, data, sizeof(unsigned char) * n, cudaMemcpyHostToDevice);
	int *arr_count, *h_arr_count;
	h_arr_count = (int*)malloc(sizeof(int) * BLOCK_SIZE);
	cudaMalloc(&arr_count, sizeof(int) * BLOCK_SIZE);
	int *prefix_sum;
	cudaMalloc(&prefix_sum, sizeof(int) * BLOCK_SIZE);
	
	unsigned char *arr_out;
	cudaMalloc(&arr_out, sizeof(unsigned char) * n);

	cudaEvent_t start, end;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&end));
	CSC(cudaEventRecord(start));

	histogram_count_values<<<128,128>>>(arr_in, n, arr_count);
	scan_cum_sum<<<1, BLOCK_SIZE>>>(arr_count, prefix_sum);
	cudaMemcpy(h_arr_count, prefix_sum, sizeof(int) * BLOCK_SIZE, cudaMemcpyDeviceToHost);
	//for (int i = 0; i < BLOCK_SIZE; i++) {
		//if (h_arr_count[i] != 0)
	//  	cerr << i << ":" << h_arr_count[i] << " ";
	//}
	//cerr << "\n";
	count_sort_gpu<<<128, 128>>>(arr_in, arr_out, prefix_sum, n);
	
	CSC(cudaEventRecord(end));
	CSC(cudaEventSynchronize(end));
	float t;
	CSC(cudaEventElapsedTime(&t, start, end));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(end));

    cudaMemcpy(data, arr_out, sizeof(unsigned char) * n, cudaMemcpyDeviceToHost);
	freopen(NULL, "wb", stdout);
	fwrite(data, sizeof(unsigned char), n, stdout);
	fclose(stdout);

	fprintf(stderr, "\nGPU: ready\nWorking time:     %f sec.\n", t/1000);

	/*for (int i = 0; i < n; i++) {
		if (data[i] != 0) {
			cerr << i << ":" << data[i] << ",";
		}
	}
	cerr << "\n";*/
	//freopen(NULL, "wb", stderr);
	//fwrite(data, sizeof(unsigned char), n, stderr);
	//fclose(stderr);
	
	cudaFree(prefix_sum);
	cudaFree(arr_count);
	cudaFree(arr_in);
	cudaFree(arr_out);
	free(data);
	free(h_arr_count);
}

