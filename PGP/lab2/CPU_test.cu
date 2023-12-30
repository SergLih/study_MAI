#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
//#include <unistd.h>
#include <pthread.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <time.h>

#define CSC(call) do { \
	cudaError_t res = call;	\
	if (res != cudaSuccess) { \
		fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(res)); \
		exit(0); \
	} \
} while (0)

typedef struct _pixel {
	unsigned char x;
	unsigned char y;
	unsigned char z;
	unsigned char w;	
} Pixel;

int w_in, h_in, w_out, h_out;
int thrCntX, thrCntY;
Pixel ** pixels;
Pixel ** pixels_out; 
pthread_t *threads;		//динамич.массив потоков

typedef struct _params {
	int threadIdxX; 
	int threadIdxY; 
} Params;




texture<uchar4, 2, cudaReadModeElementType> tex;

__global__ void kernel(uchar4 *dst, int w_in, int h_in, int w_out, int h_out) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int x_out, y_out, i, j;
	int cw = w_in / w_out;
	int ch = h_in / h_out;

	for (x_out = idx; x_out < w_out; x_out += offsetx) {
		for (y_out = idy; y_out < h_out; y_out += offsety) {
			int x_in = x_out * cw, y_in = y_out * ch; 
			double sx = 0., sy = 0., sz = 0.;
			int n = 0;
			for(i = 0; i < cw; i++) {
				for(j = 0; j < ch; j++) {
					if(x_in + i >= w_in || y_in + j >= h_in)
						continue;
					n++;
					uchar4 t = tex2D(tex, x_in+i, y_in+j);
					sx += t.x;
					sy += t.y;
					sz += t.z;
				}
			}
			sx /= n;
			sy /= n;
			sz /= n;
			dst[y_out * w_out + x_out] = make_uchar4(sx, sy, sz, 0);
		}
	}
}

void* cpu_kernel(void *dummyPtr){
    Params *p = (Params *)dummyPtr;
    int idx = p->threadIdxX;
	int idy = p->threadIdxY;
	int offsetx = thrCntX;
	int offsety = thrCntY;
	int x_out, y_out, i, j;
	int cw = w_in / w_out;
	int ch = h_in / h_out;
	for (y_out = idy; y_out < h_out; y_out += offsety) {
			for (x_out = idx; x_out < w_out; x_out += offsetx) {
			int x_in = x_out * cw, y_in = y_out * ch; 
			int sx = 0, sy = 0, sz = 0;
			int n = 0;
			for(j = 0; j < ch; j++) {
				for(i = 0; i < cw; i++) {
					if(x_in + i >= h_in || y_in + j >= w_in)
						continue;
					n++;
					Pixel t = pixels[y_in+j][x_in+i];
					sx += t.x;
					sy += t.y;
					sz += t.z;
					//printf("\t[%d %d] %x %x %x | %d %d %d\n", idx, idy, t.x, t.y, t.z, sx, sy, sz);
				}
			}
			sx /= n;
			sy /= n;
			sz /= n;
			pixels_out[y_out][x_out] = Pixel({(unsigned char)sx, 
											 (unsigned char)sy, 
											 (unsigned char)sz, 0});
			//printf("[%d %d] %d %d: %x %x %x\n", idx, idy, x_out, y_out, (unsigned char)sx,
			// (unsigned char)sy, (unsigned char)sz);
		}
	}
	return NULL;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "%s\n","Usage: BlockNumber ThreadsNumber NewSizeX NewSizeY");
        exit(EXIT_FAILURE);
    }

	int blocksCount = atoi(argv[1]);
	int threadsInBlocksCount = atoi(argv[2]);
	int threadCountX = blocksCount * threadsInBlocksCount;
	thrCntX = thrCntY = threadCountX;
	w_out = atoi(argv[3]);
	h_out = atoi(argv[4]);
	char infilename[255], outfilename_cpu[255], outfilename_gpu[255];
	//scanf("%s", infilename);
	//scanf("%s", outfilename);
	//scanf("%d %d", &w_out, &h_out);
	strcpy(infilename, "big.data");
	strcpy(outfilename_cpu, "big_cpu.data");
	strcpy(outfilename_gpu, "big_gpu.data");
	
	FILE *in = fopen(infilename, "rb");
	fread(&w_in, sizeof(int), 1 , in);
	fread(&h_in, sizeof(int), 1 , in);
	printf("Original size: %d x %d\n", w_in, h_in);

	uchar4 *data = (uchar4*)malloc(sizeof(uchar4) * h_in * w_in);
	fread(data, sizeof(uchar4), h_in * w_in, in);
	fseek(in, sizeof(int)*2, SEEK_SET);

	pixels = new Pixel*[h_in];
	for(int i = 0; i < h_in; i++) {
		pixels[i] = new Pixel[w_in];
		fread(pixels[i], sizeof(Pixel), w_in, in);
	}
	fclose(in);
	pixels_out = new Pixel*[h_out];
	for(int i = 0; i < h_out; i++) {
		pixels_out[i] = new Pixel[w_out];
	}
	time_t start0 = clock();
	threads = new pthread_t[threadCountX*threadCountX];
	Params * params = new Params[threadCountX*threadCountX];
	for(int i = 0; i < threadCountX; i++)
	{
		for(int j = 0; j < threadCountX; j++){
			int z = i*threadCountX + j;
			params[z].threadIdxX = i;
			params[z].threadIdxY = j;
			pthread_create(&threads[z], 
				NULL, cpu_kernel, (void *) &params[z]);
		}
	}
	time_t end0 = clock();
    
    
    for(int i = 0; i < threadCountX*threadCountX; i++) {
    	pthread_join(threads[i], NULL);
    } 
    fprintf(stderr, "CPU: ready\n");
    fprintf(stderr, "Working time:     %f sec.\n", (double)(end0 - start0) / (double)CLOCKS_PER_SEC);
    


	cudaArray *arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	cudaMallocArray(&arr, &ch, w_in, h_in);
	cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * h_in * w_in, cudaMemcpyHostToDevice);

	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.channelDesc = ch;
	tex.filterMode = cudaFilterModePoint;
	tex.normalized = false; 

	cudaBindTextureToArray(tex, arr, ch);
	uchar4 *dev_data;
	cudaMalloc(&dev_data, sizeof(uchar4) * h_out * w_out);
	
	cudaEvent_t start, end;
	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&end));
	CSC(cudaEventRecord(start));

	kernel<<<dim3(blocksCount, blocksCount), 
			 dim3(threadsInBlocksCount, threadsInBlocksCount)>>>(
			 	dev_data, w_in, h_in, w_out, h_out);
	CSC(cudaGetLastError());

	CSC(cudaEventRecord(end));
	CSC(cudaEventSynchronize(end));
	float t;
	CSC(cudaEventElapsedTime(&t, start, end));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(end));

	fprintf(stderr, "GPU: ready\nWorking time:     %f sec.\n", t/1000);

	cudaMemcpy(data, dev_data, sizeof(uchar4) * h_out * w_out, cudaMemcpyDeviceToHost);

	FILE *out = fopen(outfilename_gpu, "wb");
	fwrite(&w_out, sizeof(int), 1, out);
	fwrite(&h_out, sizeof(int), 1, out);
	fwrite(data, sizeof(uchar4), w_out * h_out, out);
	fclose(out);

	cudaUnbindTexture(tex);
	cudaFreeArray(arr);
	cudaFree(dev_data);


	/////////////////////////////////////////////////////////////////////

	FILE *out2 = fopen(outfilename_cpu, "wb");
	fwrite(&w_out, sizeof(int), 1, out2);
	fwrite(&h_out, sizeof(int), 1, out2);
	for(int i = 0; i < h_out; i++) 
		fwrite(pixels_out[i], sizeof(Pixel), w_out, out2);
	fclose(out2);

	for(int i = 0; i < h_in; i++)
		delete[] pixels[i];
	for(int i = 0; i < h_out; i++) {
		delete[] pixels_out[i];
	}
	delete[] pixels;
	delete[] pixels_out;

	delete[] threads;
	delete[] params;


	free(data);

	return 0;
}
