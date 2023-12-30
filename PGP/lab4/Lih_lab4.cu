#include <stdio.h>
#include <vector>
#include <math.h>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#define CSC(call) do { \
	cudaError_t pixels = call;	\
	if (pixels != cudaSuccess) { \
		fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(pixels)); \
		exit(0); \
	} \
} while (0)

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


struct cmpAbsDoubles {
	__host__ __device__ bool operator()(double a, double b) {
		return fabs(a) < fabs(b);
	}
};


int main() {

	int n;
	scanf("%d", &n);

	double *a  = (double*)malloc(sizeof(double)*n*n);

	for (int row = 0; row < n; row++) {
		for (int col = 0; col < n; col++) {
			scanf("%le", &a[col * n + row]);
		}
	}

	double *devCols;
	CSC(cudaMalloc(&devCols, sizeof(double) * n * n));
	CSC(cudaMemcpy(devCols, a, sizeof(double) * n * n, cudaMemcpyHostToDevice));

	int replacements = 0;
	int maxPos;
	for (int step = 0; step < n - 1; step++) {
		thrust::device_ptr<double> devPtr = thrust::device_pointer_cast(devCols + step * (n + 1));
		thrust::device_ptr<double> maxPtr = thrust::max_element(devPtr, devPtr + (n - step), cmpAbsDoubles());
		maxPos = &maxPtr[0] - &devPtr[0] + step;
		if(maxPos != step){
			replacements++;
			swapRows<<<256, 256>>>(n, devCols, step, maxPos);
			CSC(cudaGetLastError());
		}
		gaussElim<<<dim3(16, 16), dim3(16, 16)>>>(n, devCols, step);
		// print0(a, n, step);

		CSC(cudaGetLastError());
	}

	CSC(cudaMemcpy(a, devCols, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
	//print0(a, n, n);
	int neg_count = 0;
	double sumlog = 0;
	bool zero = false;
	for(int i = 0; i < n; i++)
	{
		double x = a[n*i + i];
		if(x < 0)
			neg_count++;
		else if(fabs(x) <= 1e-7) {
			zero = true;
			break;
		}
		sumlog += log(fabs(x));
	}
	double res = zero ? 0 : exp(sumlog) * (((neg_count + replacements) % 2) == 0 ? 1 : -1);
	printf("%.10le", res);

	CSC(cudaFree(devCols));

	free(a);

	return 0;
}
