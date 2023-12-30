#include <cstdio>

#define MAX_N_CLUSTERS 32
#define MAX_INT 4294967295
#define MAX_ULONG 18446744073709551616

__constant__ int dim[2];
__constant__ unsigned char n_clusters[1];
__constant__ float4 centroids[MAX_N_CLUSTERS];

typedef unsigned long long _ulong;
typedef uint uint;

unsigned char n_clu;
uint h_dim[2];
float4 h_centroids[MAX_N_CLUSTERS];

__device__ float dist(uchar4 p1, float4 p2)
{
	return (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y)
	       + (p1.z - p2.z)*(p1.z - p2.z);
}

__global__ void kernel(uchar4 *dst, ulonglong4 *clu_info, uint * cluster_reassignments) {
	
	uint w = dim[1], h = dim[0], n_clu = n_clusters[0];
  	const uint global_index = blockIdx.x * blockDim.x + threadIdx.x;
  	if(global_index >= w * h){
  		//printf("glob %u -- return\n", global_index);
  		return;
  	}
  	const uint offset = blockDim.x * gridDim.x;

	for(uint z = global_index; z < w*h; z += offset) {
		uchar4 p = dst[z]; 
		unsigned char i_best_clu = 0, i_prev_clu = p.w;
		float d_best_clu = dist(p, centroids[0]);

		for(unsigned char i_clu = 1; i_clu<n_clu; i_clu++)
		{
			float d = dist(p, centroids[i_clu]);
			if(d < d_best_clu)
			{
				d_best_clu = d;
				i_best_clu = i_clu;
			}
		}
		if(i_best_clu != i_prev_clu)
		{
			atomicInc(&cluster_reassignments[0], MAX_INT);
		}
		dst[z].w = i_best_clu;

		atomicAdd(&clu_info[i_best_clu].x, (unsigned long long)p.x);		//sum of red
		atomicAdd(&clu_info[i_best_clu].y, (unsigned long long)p.y);		//sum of green
		atomicAdd(&clu_info[i_best_clu].z, (unsigned long long)p.z);		//sum of blue
		atomicAdd(&clu_info[i_best_clu].w, (unsigned long long)1);	//count
		__syncthreads();
	}
}

void debug_print_centroids()
{
	printf("\ncentroids:\n");
	for (int i = 0; i < n_clu; i++) {
		printf("cluster %d: %f %f %f\n", i, h_centroids[i].x, 
								  h_centroids[i].y,
								  h_centroids[i].z);
	}
}


int main() {
	uint blocksCount = 8192, threadsCount = 256;
	uint w, h;
	char infilename[255], outfilename[255];
	scanf("%s", infilename);
	scanf("%s", outfilename);
	scanf("%hhu", &n_clu);
	uint coords[n_clu][2];
	for (int i = 0; i < n_clu; i++) {
			scanf("%u %u", &coords[i][1], &coords[i][0]);
	}

	FILE *in = fopen(infilename, "rb");
	fread(&w, sizeof(int), 1 , in);
	fread(&h, sizeof(int), 1 , in);
	h_dim[1] = w;
	h_dim[0] = h;
	uchar4 *data = (uchar4*)malloc(sizeof(uchar4) * h * w);
	fread(data, sizeof(uchar4), h * w, in);
	fclose(in);

    for (int i = 0; i < n_clu; i++) {
          uchar4 t = data[coords[i][0]*w + coords[i][1]];
          h_centroids[i] = make_float4(t.x, t.y, t.z, t.w);
    }
	//debug_print_centroids();

	cudaMemcpyToSymbol(n_clusters, &n_clu, sizeof(unsigned char));
	cudaMemcpyToSymbol(dim, h_dim, sizeof(int)*2);
	cudaMemcpyToSymbol(centroids, h_centroids, sizeof(float4) * MAX_N_CLUSTERS);
	
	ulonglong4 *h_clu_info = (ulonglong4*)malloc(sizeof(ulonglong4) *n_clu);

	uchar4 *res;
	ulonglong4 *clu_info;
	uint *cluster_reassignments;

	cudaMalloc(&res, sizeof(uchar4) * h * w);
	cudaMemcpy(res, data, sizeof(uchar4) * h * w, cudaMemcpyHostToDevice);

	cudaMalloc(&clu_info, sizeof(ulonglong4) * n_clu);
	cudaMalloc((void**)&cluster_reassignments, sizeof(uint));
	
	int i_iter = 0;
	uint h_cluster_reassignments = 0;
	do {
		i_iter++;
		cudaMemset(cluster_reassignments, 0, sizeof(uint));
		cudaMemset(clu_info, 0, sizeof(ulonglong4) * n_clu);

		kernel<<<blocksCount, threadsCount>>>(res, clu_info, cluster_reassignments);

		cudaMemcpy(h_clu_info, clu_info, sizeof(ulonglong4) * n_clu, cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_cluster_reassignments, cluster_reassignments, sizeof(uint), cudaMemcpyDeviceToHost);
		//printf("iter %d, cluster reassignments = %d\n", i_iter, h_cluster_reassignments);
		for (int i = 0; i < n_clu; i++) {
			float n = h_clu_info[i].w;
			if(n==0)
				n = 1;
			//printf("iter %d, cluster %d, n = %d\n", i_iter, i, n);
			h_centroids[i] = make_float4(h_clu_info[i].x / n, h_clu_info[i].y / n, h_clu_info[i].z / n, 0);
		}
		cudaMemcpyToSymbol(centroids, h_centroids, sizeof(float4) * n_clu);
	
	}
	while(h_cluster_reassignments > 0 /*&& i_iter < 7*/);

	cudaMemcpy(data, res, sizeof(uchar4) * h * w, cudaMemcpyDeviceToHost);

	FILE *out = fopen(outfilename, "wb");
	fwrite(&w, sizeof(int), 1, out);
	fwrite(&h, sizeof(int), 1, out);
	fwrite(data, sizeof(uchar4), w * h, out);
	fclose(out);

	cudaFree(res);
	cudaFree(clu_info);
	cudaFree(cluster_reassignments);

	free(data);
	free(h_clu_info);

	cudaDeviceReset();
	return 0;
}
