#include <stdio.h>

#define CSC(call) do { \
	cudaError_t res = call;	\
	if (res != cudaSuccess) { \
		fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(res)); \
		exit(0); \
	} \
} while (0)

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

int main() {
	int w_in, h_in, w_out, h_out;
	char infilename[255], outfilename[255];
	scanf("%s", infilename);
	scanf("%s", outfilename);
	scanf("%d %d", &w_out, &h_out);
	FILE *in = fopen(infilename, "rb");
	fread(&w_in, sizeof(int), 1 , in);
	fread(&h_in, sizeof(int), 1 , in);


	uchar4 *data = (uchar4*)malloc(sizeof(uchar4) * h_in * w_in);
	fread(data, sizeof(uchar4), h_in * w_in, in);
	fclose(in);

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
	kernel<<<dim3(16, 16), dim3(16, 16)>>>(dev_data, w_in, h_in, w_out, h_out);
	cudaMemcpy(data, dev_data, sizeof(uchar4) * h_out * w_out, cudaMemcpyDeviceToHost);

	FILE *out = fopen(outfilename, "wb");
	fwrite(&w_out, sizeof(int), 1, out);
	fwrite(&h_out, sizeof(int), 1, out);
	fwrite(data, sizeof(uchar4), w_out * h_out, out);
	fclose(out);

	cudaUnbindTexture(tex);
	cudaFreeArray(arr);
	cudaFree(dev_data);
	free(data);

	return 0;
}
