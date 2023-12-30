#include "ssaa.h"

Vec3f ** pixels_in;
Vec3f ** pixels_out;
int threadCountX;
int w_in, h_in, w_out, h_out;
pthread_t *ssaa_threads;
SSAA_params * ssaa_params;

void cpu_ssaa(std::vector <Vec3f> &data, int width, int height, int coef, int n_threads_cpu) {
    if(width*height != data.size()){
        std::cerr << "Expected data of size " << width << "*" << "height\n";
        exit(1);
    }

    w_in = width;
    h_in = height;
    threadCountX = n_threads_cpu;

    pixels_in = new Vec3f*[height];
    for (int i = 0; i < height; ++i) {
        pixels_in[i] = new Vec3f[width];
        for (int j = 0; j < width; ++j) {
            pixels_in[i][j] = data[i*width + j];
        }
    }
    w_out = width / coef;
    h_out = height / coef;

    pixels_out = new Vec3f*[h_out];
    for (int i = 0; i < h_out; ++i) {
        pixels_out[i] = new Vec3f[w_out];
    }

    ssaa_threads = new pthread_t[threadCountX*threadCountX];
    ssaa_params = new SSAA_params[threadCountX*threadCountX];
    for(int i = 0; i < threadCountX; i++)
    {
        for(int j = 0; j < threadCountX; j++){
            int z = i*threadCountX + j;
            ssaa_params[z].threadIdxX = i;
            ssaa_params[z].threadIdxY = j;
            pthread_create(&ssaa_threads[z],
                           NULL, cpu_ssaa_kernel, (void *) &ssaa_params[z]);
        }
    }

    for(int i = 0; i < threadCountX*threadCountX; i++) {
        pthread_join(ssaa_threads[i], NULL);
    }
    delete[] ssaa_params;
    delete[] ssaa_threads;
    data.clear();
    data.resize(w_out * h_out);
    int k = 0;
    for (int i = 0; i < h_out; ++i) {
        for (int j = 0; j < w_out; ++j) {
            data[k++] = pixels_out[i][j];
        }
    }

    for (int i = 0; i < height; ++i) {
        delete[] pixels_in[i];
    }
    delete[] pixels_in;
    for (int i = 0; i < h_out; ++i) {
        delete[] pixels_out[i];
    }
    delete[] pixels_out;
}

void* cpu_ssaa_kernel(void *dummyPtr){
    SSAA_params *p = (SSAA_params *)dummyPtr;
    int idx = p->threadIdxX;
    int idy = p->threadIdxY;
    int offsetx = threadCountX;
    int offsety = threadCountX;
    int x_out, y_out, i, j;
    int cw = w_in / w_out;
    int ch = h_in / h_out;
    for (x_out = idx; x_out < h_out; x_out += offsetx) {
        for (y_out = idy; y_out < w_out; y_out += offsety) {
            int x_in = x_out * ch, y_in = y_out * cw;
            Vec3f s;
            int n = 0;
            for(i = 0; i < ch; i++) {
                for(j = 0; j < cw; j++) {
                    if(x_in + i >= h_in || y_in + j >= w_in)
                        continue;
                    n++;
                    s = s + pixels_in[x_in+i][y_in+j];
                }
            }
            pixels_out[x_out][y_out] = s * (1.0/n);
        }
    }
    return NULL;
}